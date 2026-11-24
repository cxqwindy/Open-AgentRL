#!/usr/bin/env python3
"""Convert Path-A teacher CoT (distilled from Gemini/GPT-5.5/DeepSeek) into Hermes tool
trajectories compatible with data/nmr_matching_tool_sft_v4.json.

Key idea: the teacher already reasoned over the pre-computed TOOL_LOG numbers (see
scripts/distill_teacher_cot.py). We do NOT try to re-parse the teacher's free-form step order
into a sequence of tool calls. Instead we EMIT a canonical, tool-grounded trajectory:

    system_prompt
    user(candidate + spectrum)
    assistant: <think>hypothesis (from first sentence of teacher CoT, stripped of unsupported
        integer counts)</think><tool_call>count_atoms C</tool_call>
    tool(count_atoms C) -> real JSON from TOOL_LOG
    assistant: <think>{previous obs summary} + next-step announce</think><tool_call>count_atoms H</tool_call>
    tool(count_atoms H) -> real JSON
    assistant: <think>...</think><tool_call>spectral_match</tool_call>
    tool(spectral_match) -> real JSON
    assistant: <think>...</think><tool_call>functional_group_scan</tool_call>
    tool(functional_group_scan) -> real JSON
    assistant: <think>{aggregator citing every observation}
        + teacher's key_evidence sentence</think><answer>{final_answer, confidence, key_evidence}</answer>

Every <think> block that mentions a number will have that number available inside a prior
<tool_response> (by construction), so the v5 provenance validator passes it.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recipe.nmr_matching_agent.prompt import build_system_prompt  # noqa: E402
from recipe.nmr_matching_agent.cot_to_trajectory_v5 import (  # noqa: E402
    _strip_unsupported_integer_counts,
    _post_tool_think,
    _final_think,
    _validate_think_number_provenance,
)


_FINAL_LINE_RE = re.compile(r"FINAL:\s*(\{.*?\})\s*$", re.DOTALL)


def parse_teacher_final(content: str) -> Dict[str, Any]:
    """Extract the final answer JSON that the teacher was told to emit as a FINAL: {...} line."""
    m = _FINAL_LINE_RE.search(content or "")
    if not m:
        # fallback: last JSON-looking object anywhere in the text
        idx = (content or "").rfind("{")
        end = (content or "").rfind("}")
        if idx >= 0 and end > idx:
            try:
                obj = json.loads(content[idx:end + 1])
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass
        return {"final_answer": "Unknown", "confidence": 0.0, "key_evidence": ""}
    try:
        return json.loads(m.group(1))
    except Exception:
        return {"final_answer": "Unknown", "confidence": 0.0, "key_evidence": ""}


_TOOL_MENTION_RE = re.compile(
    r"\b(count_atoms|spectral_match|check_substructure|functional_group_scan|tool_log|"
    r"returned|tool call|the tool)\b",
    re.IGNORECASE,
)


def first_hypothesis_from_teacher(content: str) -> str:
    """Extract a hypothesis sentence about the candidate STRUCTURE (no tool-return references)
    from the teacher CoT, to serve as the pre-first-tool <think>. Skip any paragraph that
    mentions a tool by name -- those reference outputs that haven't been produced yet in our
    reconstructed trajectory. Aggressively strip any numeric assertions so it passes
    provenance."""
    if not content:
        return ""
    # split on paragraphs OR sentence-ish boundaries so a big first paragraph doesn't stop us
    chunks: List[str] = []
    for para in re.split(r"\n\s*\n", content.strip()):
        chunks.extend(re.split(r"(?<=[.!?])\s+", para))
    for s in chunks:
        s = s.strip()
        if not s:
            continue
        if re.match(r"^(FINAL|Answer|Conclusion)[:\s]", s, re.IGNORECASE):
            continue
        if _TOOL_MENTION_RE.search(s):
            continue  # skip anything that already cites a tool_log return value
        # strip leading numbering ("1.", "Step 1:", "**Step 1:**")
        s = re.sub(r"^\**\s*(step\s+\d+[:.]|\d+\.\s*|\d+\)\s*)\s*", "", s, flags=re.IGNORECASE)
        s = s.replace("**", "").strip()
        cleaned = _strip_unsupported_integer_counts(s[:400])
        if cleaned and len(cleaned) > 20:
            return cleaned
    # graceful fallback: no clean sentence -> a safe boilerplate hypothesis
    return "Assess the candidate SMILES against the experimental peaks by running atom counts first, then spectral similarity"


def build_trajectory(row: Dict[str, Any], teacher_row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Produce the Hermes trajectory row for one teacher distillation record.

    row         : pool row from tool_log_pool.jsonl (has `input` + `tool_log`)
    teacher_row : record from distill_{safe}.jsonl (has `content` + `reasoning_content` + `pred`)
    """
    tl = row.get("tool_log") or {}
    inp = row["input"]
    if not inp.get("candidate_smiles"):
        return None

    # Build the canonical 4-tool sequence (skip spectral_match if it wasn't ok).
    calls_and_obs: List[Any] = []
    calls_and_obs.append(({"name": "count_atoms", "arguments": {"element": "C"}},
                          tl.get("count_atoms_C", {})))
    calls_and_obs.append(({"name": "count_atoms", "arguments": {"element": "H"}},
                          tl.get("count_atoms_H", {})))
    spec = tl.get("spectral_match", {})
    if spec.get("ok"):
        calls_and_obs.append(({"name": "spectral_match", "arguments": {}}, spec))
    fg = tl.get("functional_group_scan", {})
    if fg and not fg.get("error"):
        calls_and_obs.append(({"name": "functional_group_scan", "arguments": {}}, fg))

    if not calls_and_obs:
        return None

    # Compose user message (same JSON shape v5 rewriter uses).
    user_case = {
        "candidate_smiles": inp["candidate_smiles"],
        "c_nmr_peaks": inp.get("c_nmr_peaks") or [],
        "h_nmr_peaks": inp.get("h_nmr_peaks") or [],
        "nmr_solvent": inp.get("nmr_solvent"),
        "nmr_frequency": inp.get("nmr_frequency"),
    }
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": json.dumps(user_case, ensure_ascii=False)},
    ]

    # Assemble the assistant/tool turns with tool-grounded <think>.
    evidence: List[Any] = []
    hypothesis = first_hypothesis_from_teacher(teacher_row.get("content") or "")

    prev_tool_name: Optional[str] = None
    prev_obs_json: Optional[str] = None
    for i, (call, obs) in enumerate(calls_and_obs):
        tool_name = call["name"]
        if i == 0:
            think = (hypothesis + f" Verify with {tool_name}." if hypothesis
                     else f"Begin verification with {tool_name}.")
        else:
            think = _post_tool_think(prev_tool_name or "prev_tool",
                                     prev_obs_json or "{}", tool_name)
        assistant = f"<think>{think}</think>\n<tool_call>{json.dumps(call, ensure_ascii=False)}</tool_call>"
        messages.append({"role": "assistant", "content": assistant})
        obs_json = json.dumps(obs, ensure_ascii=False)
        messages.append({"role": "tool", "name": tool_name, "content": obs_json})
        evidence.append((tool_name, obs))
        prev_tool_name = tool_name
        prev_obs_json = obs_json

    # Final <answer> uses the teacher's parsed FINAL: JSON (or the harness-parsed pred as
    # fallback). Key_evidence sentence comes from the teacher directly for stylistic diversity.
    teacher_final = parse_teacher_final(teacher_row.get("content") or "")
    final_answer = teacher_final.get("final_answer") or teacher_row.get("pred") or "Unknown"
    confidence = teacher_final.get("confidence", 0.95)
    key_evidence = str(teacher_final.get("key_evidence", ""))[:400]
    ans_obj = {"final_answer": final_answer, "confidence": confidence,
               "key_evidence": key_evidence}
    final_think_txt = _final_think(evidence, final_answer, key_evidence=key_evidence)
    messages.append({
        "role": "assistant",
        "content": f"<think>{final_think_txt}</think>\n<answer>{json.dumps(ans_obj, ensure_ascii=False)}</answer>",
    })

    violations = _validate_think_number_provenance(messages)

    return {
        "messages": messages,
        "meta": {
            "candidate_smiles": inp["candidate_smiles"],
            "gt_smiles": row.get("gt_smiles"),
            "label": row.get("label"),
            "negative_type": row.get("negative_type"),
            "_class": row.get("_class"),
            "teacher": teacher_row.get("teacher"),
            "teacher_pred": teacher_row.get("pred"),
            "final_answer": final_answer,
            "n_tool_calls": len(calls_and_obs),
            "provenance_violations": violations,
            "source": "path_A_distill",
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True,
                    help="tool_log_pool.jsonl with `tool_log` field per row")
    ap.add_argument("--distill", nargs="+", required=True,
                    help="one or more distill_{safe_model}.jsonl teacher outputs; if >1 the "
                    "first is treated as primary and subsequent as adjudicators for disagreement")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rejects_out", default=None)
    ap.add_argument("--require_teacher_correct", action="store_true",
                    help="drop rows where the teacher's final_answer disagrees with the "
                    "ground-truth label (Yes iff label==1)")
    ap.add_argument("--drop_provenance_violations", action="store_true", default=True)
    ap.add_argument("--keep_provenance_violations", dest="drop_provenance_violations",
                    action="store_false")
    args = ap.parse_args()

    with open(args.pool) as f:
        pool = [json.loads(l) for l in f]
    print(f"[pool] {len(pool)} candidates", flush=True)

    # Primary teacher: distill[0]. Adjudicators: distill[1:] (indexed by `i`).
    primary_by_i: Dict[int, Dict[str, Any]] = {}
    with open(args.distill[0]) as f:
        for line in f:
            r = json.loads(line)
            primary_by_i[r["i"]] = r
    print(f"[primary teacher] rows in cache: {len(primary_by_i)}", flush=True)

    adjudicators_by_i: List[Dict[int, Dict[str, Any]]] = []
    for p in args.distill[1:]:
        d = {}
        with open(p) as f:
            for line in f:
                r = json.loads(line)
                d[r["i"]] = r
        adjudicators_by_i.append(d)
        print(f"[adjudicator {p}] rows in cache: {len(d)}", flush=True)

    kept: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    n_prov = n_no_teacher = n_teacher_wrong = n_swap_gpt = n_swap_ds = 0

    def _valid(rec):
        return rec is not None and rec.get("content")

    for i, pool_row in enumerate(pool):
        gt = "Yes" if pool_row.get("label") == 1 else "No"

        primary = primary_by_i.get(i)
        # Build full teacher pool in preference order: primary, then adjudicators.
        teacher_pool: List[Dict[str, Any]] = []
        if _valid(primary):
            teacher_pool.append(primary)
        for adj_map in adjudicators_by_i:
            adj = adj_map.get(i)
            if _valid(adj):
                teacher_pool.append(adj)

        if not teacher_pool:
            n_no_teacher += 1
            continue

        # Pick the first teacher whose pred matches GT; fall through to primary if none matches.
        chosen = None
        for k, t in enumerate(teacher_pool):
            if t.get("pred") == gt:
                chosen = t
                if k > 0:  # bookkeeping: which teacher we swapped in
                    tname = (t.get("teacher") or "").lower()
                    if "gpt" in tname:
                        n_swap_gpt += 1
                    else:
                        n_swap_ds += 1
                break

        if chosen is None:
            if args.require_teacher_correct:
                n_teacher_wrong += 1
                continue
            chosen = teacher_pool[0]  # keep the primary CoT (even if wrong)

        traj = build_trajectory(pool_row, chosen)
        if traj is None:
            continue
        if traj["meta"]["provenance_violations"]:
            n_prov += 1
            if args.drop_provenance_violations:
                rejected.append(traj)
                continue
        kept.append(traj)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    if args.rejects_out and rejected:
        Path(args.rejects_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.rejects_out, "w") as f:
            for r in rejected:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_pool = len(pool)
    print(f"[stats] pool={n_pool}  kept={len(kept)}  rejected_prov={n_prov}  "
          f"swapped_to_gpt={n_swap_gpt}  swapped_to_ds={n_swap_ds}  "
          f"no_teacher={n_no_teacher}  all_teachers_wrong={n_teacher_wrong}", flush=True)
    print(f"[write] {len(kept)} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
