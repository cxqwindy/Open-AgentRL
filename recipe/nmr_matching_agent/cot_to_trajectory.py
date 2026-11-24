#!/usr/bin/env python3
"""Convert structured CoT traces -> Hermes multi-turn TOOL-CALLING trajectories (tool-SFT cold start).

The structured CoT (cot_traces_v4/v6.jsonl, ~86% have steps[]) uses an `action` vocabulary that
maps 1:1 to our deterministic matching tools:
    count_carbons / count_hydrogens -> count_atoms(element=C/H)
    analyze_c_nmr / analyze_h_nmr   -> spectral_match()
    check_substructure              -> check_substructure(pattern=<inferred>) | functional_group_scan()
    conclude                        -> <answer>{...}</answer>

CRITICAL: the CoT `observation` fields are Gemini PROSE (often wrong -- e.g. "at least 23 carbons"
when RDKit says 22). We DISCARD that prose as the tool result and instead RE-RUN the deterministic
tool on the real candidate+spectrum, so the trajectory's <tool_response> carries the exact JSON the
model will see at rollout/eval time (train/inference parity). Gemini's prose is kept only as the
assistant's <think> reasoning.

DATA-QUALITY FILTER: if the deterministic evidence flatly CONTRADICTS the CoT's final answer
(e.g. count_atoms says the counts are consistent but the trace concluded "No" solely on a count
mismatch), we can optionally drop the trace. We log the contradiction rate; if >30% the CoT is too
noisy to cold-start from and we should lean on RL (see plan risks).

Output: JSONL of {messages:[...], meta:{...}} where messages is a system/user/assistant/tool/...
Hermes conversation, directly consumable by an SFT trainer (LLaMA-Factory sharegpt-style or verl SFT).

Usage:
    python cot_to_trajectory.py \
        --cot /fs_mol/cxq/SpecMaster_jp/cot_output/cot_traces_v4.jsonl \
        --pred_cache recipe/nmr_matching_agent/prediction_cache.json \
        --out recipe/nmr_matching_agent/data/tool_sft_v4.jsonl \
        [--limit 20]   # for eyeballing
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recipe.nmr_matching_agent.nmr_matching_runtime import NmrMatchingRuntime  # noqa: E402
from recipe.nmr_matching_agent.named_smarts import KEYWORD_TO_NAME  # noqa: E402
from recipe.nmr_matching_agent.prompt import build_system_prompt  # noqa: E402


def _to_scorer_peaks(c_flat, h_peaks_cot):
    """cot: C peaks are flat floats; H peaks are dicts with 'delta'/'nH'. Convert to runtime record form."""
    c = [{"delta (ppm)": float(v)} for v in (c_flat or [])]
    h = []
    for p in (h_peaks_cot or []):
        d = p.get("delta")
        if d is None:
            continue
        h.append({"delta": float(d), "centroid": float(d), "nH": p.get("nH")})
    return c, h


def _infer_substructure_pattern(text: str) -> Optional[str]:
    """Map a check_substructure step's prose to a named group, via KEYWORD_TO_NAME."""
    low = (text or "").lower()
    # longest keyword first to prefer specific matches (e.g. 'trimethylsilyl' before 'silyl')
    for kw in sorted(KEYWORD_TO_NAME, key=len, reverse=True):
        if kw in low:
            return KEYWORD_TO_NAME[kw]
    return None


# action -> (tool_name, args_builder(step)->dict) ; None tool_name means "answer/skip"
def _map_step_to_toolcall(step: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    action = (step.get("action") or "").strip()
    obs_text = f"{step.get('observation','')} {step.get('conclusion','')}"
    if action in ("count_carbons",):
        return {"name": "count_atoms", "arguments": {"element": "C"}}
    if action in ("count_hydrogens",):
        return {"name": "count_atoms", "arguments": {"element": "H"}}
    if action in ("count_carbons_and_hydrogens",):
        return {"name": "count_atoms", "arguments": {"element": "C"}}  # C first; H can be a separate step
    if action in ("analyze_c_nmr", "analyze_h_nmr"):
        return {"name": "spectral_match", "arguments": {}}
    if action in ("check_substructure", "analyze_substructure", "analyze_structure"):
        pat = _infer_substructure_pattern(obs_text)
        if pat:
            return {"name": "check_substructure", "arguments": {"pattern": pat}}
        return {"name": "functional_group_scan", "arguments": {}}
    if action in ("conclude",):
        return None  # handled as the answer
    return None


async def _exec_tool(runtime: NmrMatchingRuntime, tool_call: Dict[str, Any]) -> str:
    res = await runtime.run_tool_call_async({"function": tool_call})
    return res.get("content", "")


def _dedup_toolcalls(calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drop consecutive identical tool calls (e.g. two analyze_h_nmr -> one spectral_match)."""
    out = []
    seen_keys = set()
    for c in calls:
        key = json.dumps(c, sort_keys=True)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        out.append(c)
    return out


def convert_one(rec: Dict[str, Any], pred_cache: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    reasoning = rec.get("reasoning") or {}
    steps = reasoning.get("steps")
    if not steps:
        return None  # only convert structured traces
    inp = rec.get("input") or {}
    cand = inp.get("candidate_smiles")
    if not cand:
        return None

    c_peaks, h_peaks = _to_scorer_peaks(inp.get("c_nmr_peaks"), inp.get("h_nmr_peaks"))
    record = {
        "candidate_smiles": cand,
        "c_nmr_peaks": c_peaks,
        "h_nmr_peaks": h_peaks,
        "nmr_solvent": inp.get("nmr_solvent"),
        "nmr_frequency": inp.get("nmr_frequency"),
    }
    runtime = NmrMatchingRuntime(record, prediction_cache=pred_cache, allow_remote_predict=False)

    # Build the tool-call sequence from the CoT steps (skip conclude), dedup consecutive repeats.
    raw_calls = []
    think_bits = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        tc = _map_step_to_toolcall(step)
        if tc is not None:
            raw_calls.append(tc)
        # collect gemini prose as think material
        obs = step.get("observation")
        if obs:
            think_bits.append(str(obs))
    calls = _dedup_toolcalls(raw_calls)

    # user case (JSON, matching the rollout prompt shape)
    user_case = {
        "candidate_smiles": cand,
        "c_nmr_peaks": c_peaks,
        "h_nmr_peaks": h_peaks,
        "nmr_solvent": inp.get("nmr_solvent"),
        "nmr_frequency": inp.get("nmr_frequency"),
    }
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": json.dumps(user_case, ensure_ascii=False)},
    ]

    # Execute each tool, build assistant(think+tool_call)/tool(observation) turns.
    loop = asyncio.get_event_loop()
    evidence_summary = []
    for i, tc in enumerate(calls):
        think = think_bits[i] if i < len(think_bits) else "Let me verify this with a tool."
        assistant = f"<think>{think}</think>\n<tool_call>{json.dumps(tc, ensure_ascii=False)}</tool_call>"
        messages.append({"role": "assistant", "content": assistant})
        obs_json = loop.run_until_complete(_exec_tool(runtime, tc))
        messages.append({"role": "tool", "name": tc["name"], "content": obs_json})
        try:
            evidence_summary.append(json.loads(obs_json))
        except Exception:
            pass

    # final answer
    final_answer = reasoning.get("final_answer")
    confidence = reasoning.get("confidence", 0.9)
    key_evidence = reasoning.get("key_evidence", "")
    ans_obj = {"final_answer": final_answer, "confidence": confidence, "key_evidence": key_evidence}
    messages.append({
        "role": "assistant",
        "content": f"<think>Synthesizing the tool evidence.</think>\n<answer>{json.dumps(ans_obj, ensure_ascii=False)}</answer>",
    })

    # contradiction check: does the deterministic evidence conflict with the CoT answer?
    contradiction = _detect_contradiction(final_answer, evidence_summary)

    return {
        "messages": messages,
        "meta": {
            "candidate_smiles": cand,
            "gt_smiles": rec.get("gt_smiles", ""),
            "label": rec.get("label"),
            "negative_type": rec.get("negative_type", "positive"),
            "final_answer": final_answer,
            "n_tool_calls": len(calls),
            "reasoning_level": rec.get("reasoning_level"),
            "contradiction": contradiction,
        },
    }


def _detect_contradiction(final_answer: Optional[str], evidence: List[Dict[str, Any]]) -> Optional[str]:
    """Flag traces whose deterministic evidence points the OTHER way from the CoT answer.

    Conservative: only flag the clearest cases.
      * answer 'No' but every count_atoms match=True AND spectral_match similarity high (>0.6):
        the deterministic evidence supports a match, contradicting a 'No'.
      * answer 'Yes' but a count_atoms shows match=False (fewer candidate atoms than signals):
        a hard structural contradiction against a 'Yes'.
    """
    counts = [e for e in evidence if e.get("element") in ("C", "H")]
    specs = [e for e in evidence if "similarity" in e]
    any_count_mismatch = any(e.get("match") is False for e in counts)
    all_count_ok = counts and all(e.get("match") in (True, None) for e in counts)
    high_sim = any((e.get("ok") and e.get("similarity", 0) > 0.6) for e in specs)

    if final_answer == "Yes" and any_count_mismatch:
        return "yes_but_count_mismatch"
    if final_answer == "No" and all_count_ok and high_sim:
        return "no_but_evidence_supports_match"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cot", required=True)
    ap.add_argument("--pred_cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="convert only first N structured traces (eyeball)")
    ap.add_argument("--drop_contradictions", action="store_true",
                    help="exclude traces whose deterministic evidence contradicts the CoT answer")
    args = ap.parse_args()

    with open(args.pred_cache) as f:
        pred_cache = json.load(f)
    print(f"[cache] {len(pred_cache)} predictions")

    n_total = n_structured = n_converted = n_contra = 0
    out_rows = []
    with open(args.cot) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_total += 1
            rec = json.loads(line)
            if not (rec.get("reasoning") or {}).get("steps"):
                continue
            n_structured += 1
            row = convert_one(rec, pred_cache)
            if row is None:
                continue
            if row["meta"]["contradiction"]:
                n_contra += 1
                if args.drop_contradictions:
                    continue
            out_rows.append(row)
            n_converted += 1
            if args.limit and n_converted >= args.limit:
                break

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[convert] total={n_total} structured={n_structured} converted={n_converted}")
    rate = 100.0 * n_contra / max(n_structured, 1)
    print(f"[convert] contradictions={n_contra} ({rate:.1f}% of structured) "
          f"{'-- DROPPED' if args.drop_contradictions else '-- kept'}")
    if rate > 30:
        print("[WARN] contradiction rate >30%: CoT is noisy; lean toward RL over SFT cold-start.")
    print(f"[write] {len(out_rows)} trajectories -> {args.out}")


if __name__ == "__main__":
    main()
