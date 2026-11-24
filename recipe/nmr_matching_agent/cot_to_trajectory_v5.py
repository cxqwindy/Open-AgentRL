#!/usr/bin/env python3
"""v5: tool-grounded <think> rewriter + provenance validator.

Difference from cot_to_trajectory.py (v4):
- Pre-tool <think> before the FIRST tool call = hypothesis WITHOUT unsupported integer counts
  (uses Gemini prose only to state the "what to verify" claim, strips concrete numbers).
- Pre-tool <think> before subsequent tool calls = generated FROM the previous <tool_response>
  JSON so it quotes the actual returned numbers (e.g. "count_atoms returned {C: 22}. matches...").
- Final <answer> <think> = aggregator over ALL <tool_response> observations in this trajectory,
  citing each observed numeric field explicitly.
- Provenance validator: every integer atom-count and similarity value in any <think> block MUST
  appear in some PRIOR <tool_response> of the same trajectory (per user choice: "middle"
  strictness — delta-ppm values are whitelisted).

The v4 -> v5 diff is entirely inside the two think-builders and the new validator; the tool
sequence, execution, and Hermes packaging are unchanged.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recipe.nmr_matching_agent.nmr_matching_runtime import NmrMatchingRuntime  # noqa: E402
from recipe.nmr_matching_agent.named_smarts import KEYWORD_TO_NAME  # noqa: E402
from recipe.nmr_matching_agent.prompt import build_system_prompt  # noqa: E402

# reuse the v4 mapping + peak conversion helpers so behaviour stays parity
from recipe.nmr_matching_agent.cot_to_trajectory import (  # noqa: E402
    _to_scorer_peaks,
    _infer_substructure_pattern,
    _map_step_to_toolcall,
    _dedup_toolcalls,
    _exec_tool,
    _detect_contradiction,
)


# ---------------------------------------------------------------------------
# Tool-grounded think builders
# ---------------------------------------------------------------------------
_STRIP_INT_COUNT_RE = re.compile(
    r"\b(?:at\s+least\s+|exactly\s+|only\s+)?\d{1,3}\s*"
    r"(?:carbons?|hydrogens?|nitrogens?|oxygens?|protons?|signals?|matches?|groups?|"
    r"H\s+atoms?|C\s+atoms?|N\s+atoms?|O\s+atoms?)\b",
    re.IGNORECASE,
)
_STRIP_SIM_RE = re.compile(r"(?<![0-9.])(0?\.\d{2,4})(?![0-9])")
# Teacher CoT often quotes TOOL_LOG numbers as key=value tokens
# ("candidate_count=18", "similarity=0.5563"). These need scrubbing when they appear in
# text destined for a pre-tool <think> (before any tool has executed in the trajectory).
_STRIP_KV_RE = re.compile(
    r"\b(?:candidate_count|experimental_(?:distinct_signals|total_nH)|"
    r"n_pred_[chCH]|n_exp_[chCH]|n_matches|n_groups|similarity|c_sim|h_sim)\s*=\s*[\d.]+\b",
    re.IGNORECASE,
)
# NMR integration in prose (kept because pre-tool teacher often writes "6H singlet at 2.93 ppm")
_STRIP_NMR_INT_RE = re.compile(
    r"(?<![A-Za-z0-9])\d{1,3}H(?=[\s,(]+(?:singlet|doublet|triplet|multiplet|quartet|"
    r"integrating|integr\.|s\b|d\b|t\b|m\b|q\b|dd\b|dt\b|td\b|ddd\b))",
    re.IGNORECASE,
)


def _strip_unsupported_integer_counts(text: str) -> str:
    """Remove phrases like 'at least 23 carbons', 'sum of 17 protons', kv notations
    ('candidate_count=18', 'similarity=0.85'), NMR integrations ('6H singlet'), or bare
    similarity floats. Used for text that will land in a <think> before a tool has returned
    those exact numbers -- so the training distribution never asserts an unsourced quantity.

    Kept exhaustive so stripped prose passes the same rules the validator enforces.
    """
    if not text:
        return ""
    out = _STRIP_KV_RE.sub("[to be verified]", text)
    out = _STRIP_INT_COUNT_RE.sub("[to be verified]", out)
    out = _STRIP_NMR_INT_RE.sub("[to be verified]", out)
    out = _STRIP_SIM_RE.sub("[to be verified]", out)
    out = re.sub(r"\s+,", ",", out)
    out = re.sub(r"\s{2,}", " ", out).strip()
    return out


def _first_pre_tool_think(gemini_prose: Optional[str], planned_tool: str) -> str:
    """<think> before the very first tool call. Take Gemini's hypothesis (feature-level),
    strip unsupported integer atom counts, and announce the next tool."""
    hypothesis = _strip_unsupported_integer_counts(gemini_prose or "")
    if not hypothesis:
        hypothesis = "Inspect the candidate SMILES and experimental peaks to form a hypothesis."
    return f"{hypothesis} Verify with {planned_tool}."


def _summarize_obs(tool_name: str, obs: Dict[str, Any]) -> str:
    """Turn a single <tool_response> JSON into a short natural sentence that quotes its actual
    return values. This becomes the anchor of the next <think>."""
    if tool_name == "count_atoms":
        el = obs.get("element", "?")
        cand = obs.get("candidate_count")
        if el == "C":
            exp = obs.get("experimental_distinct_signals")
            m = obs.get("match")
            verdict = ("consistent (candidate C >= signals)" if m
                       else "CONTRADICTION (fewer candidate carbons than signals)")
            return f"count_atoms(C) returned candidate_count={cand}, experimental_signals={exp} — {verdict}"
        # H
        exp = obs.get("experimental_total_nH")
        m = obs.get("match")
        if exp is None:
            return f"count_atoms(H) returned candidate_count={cand}, experimental_total_nH=unavailable"
        verdict = "counts match" if m else "MISMATCH in total proton count"
        return f"count_atoms(H) returned candidate_count={cand}, experimental_total_nH={exp} — {verdict}"
    if tool_name == "spectral_match":
        if not obs.get("ok"):
            return "spectral_match returned ok=false (prediction unavailable)"
        sim = obs.get("similarity")
        c_sim = obs.get("c_sim")
        h_sim = obs.get("h_sim")
        n_pc, n_ec = obs.get("n_pred_c"), obs.get("n_exp_c")
        n_ph, n_eh = obs.get("n_pred_h"), obs.get("n_exp_h")
        c_absent = obs.get("c_data_absent", False)
        base = (f"spectral_match returned similarity={sim}, c_sim={c_sim}, h_sim={h_sim}, "
                f"n_pred_c={n_pc} vs n_exp_c={n_ec}, n_pred_h={n_ph} vs n_exp_h={n_eh}")
        if c_absent:
            base += "; c_data_absent=true — n_exp_c=0 so c_sim=0.0 is uninformative, rely on h_sim only"
        return base
    if tool_name == "check_substructure":
        present = obs.get("present_in_candidate")
        nm = obs.get("resolved_name") or obs.get("pattern")
        n = obs.get("n_matches")
        v = "present" if present else "NOT present"
        return f"check_substructure({nm}) returned {v} in candidate (n_matches={n})"
    if tool_name == "functional_group_scan":
        groups = obs.get("groups_present", [])
        n = obs.get("n_groups", 0)
        return f"functional_group_scan returned {n} groups: {groups}"
    if tool_name == "peak_assignment":
        return (f"peak_assignment({obs.get('nucleus')}) returned n_matched={obs.get('n_matched')}, "
                f"n_unmatched_exp={obs.get('n_unmatched_exp')}, "
                f"n_unmatched_pred={obs.get('n_unmatched_pred')}, "
                f"n_large_residual={obs.get('n_large_residual')}, "
                f"max_abs={obs.get('residual_stats', {}).get('max_abs')}")
    # unknown tool
    return f"{tool_name} returned: {json.dumps(obs, ensure_ascii=False)[:120]}"


def _post_tool_think(prev_tool_name: str, prev_obs_json: str, next_tool: str) -> str:
    """<think> that references the previous tool's returned numbers, then announces next tool."""
    try:
        obs = json.loads(prev_obs_json)
    except Exception:
        obs = {}
    anchor = _summarize_obs(prev_tool_name, obs)
    return f"{anchor}. Next: call {next_tool} to gather further evidence."


# def _extract_numbers_from_evidence(evidence: List[Tuple[str, Dict[str, Any]]]) -> Tuple[set, list]:
#     """Extract all numbers from tool responses to build a whitelist."""
#     numbers = set()
#     sim_values = []  # Track similarity values for fuzzy matching

#     for _, obs in evidence:
#         obs_str = json.dumps(obs)
#         for match in re.finditer(r'\b(\d+\.?\d*)\b', obs_str):
#             num_str = match.group(1)
#             numbers.add(num_str)

#             # Track similarity values (0.XXXX) for fuzzy matching
#             if '.' in num_str and num_str.startswith('0.'):
#                 try:
#                     val = float(num_str)
#                     if 0 <= val <= 1:
#                         sim_values.append(val)
#                         numbers.add(f'{val:.3f}')
#                         numbers.add(f'{val:.2f}')
#                 except:
#                     pass

#     return numbers, sim_values


# def _sanitize_key_evidence(text: str, allowed_numbers: set, sim_values: list) -> str:
#     """Keep numbers from tool outputs; be permissive with similarity values."""
#     def replace_number(match):
#         num = match.group(0)

#         # Exact match in whitelist
#         if num in allowed_numbers:
#             return num

#         if '.' in num:
#             try:
#                 val = float(num)
#                 # If it's a valid similarity value and we have ANY similarity in tools, keep it
#                 if 0 <= val <= 1 and sim_values:
#                     return num
#             except:
#                 pass

#         # Remove if no match
#         return ""

#     text = re.sub(r'\b\d+\.?\d*\b', replace_number, text)
#     text = re.sub(r'\s{2,}', ' ', text)
#     text = re.sub(r'\s+([,;.])', r'\1', text)
#     return text.strip(" ,;.")


def _final_think(
    evidence: List[Tuple[str, Dict[str, Any]]],
    final_answer: Optional[str],
    key_evidence: str = "",
) -> str:
    """Aggregator <think> before the final <answer>. Cites every observation numerically."""
    if not evidence:
        return f"No tool evidence collected; defaulting to structural inspection. Final: {final_answer}."
    bits = [_summarize_obs(name, obs) for name, obs in evidence]
    base = " ; ".join(bits)
    if key_evidence:
        # Keep key_evidence as-is - teacher wrote it after seeing tool outputs
        # Minor rounding differences (0.997 vs 0.9977) are not hallucinations
        return f"{base}. {key_evidence}. Therefore final_answer = {final_answer}."
    return f"{base}. Therefore final_answer = {final_answer}."

# ---------------------------------------------------------------------------
# Provenance validator
# ---------------------------------------------------------------------------
# Integer atom counts asserted in <think>: "22 carbons", "17 H atoms", ...
_INT_COUNT_RE = re.compile(
    r"\b(\d{1,3})\s*(carbons?|hydrogens?|nitrogens?|oxygens?|C\s+atoms?|H\s+atoms?|"
    r"protons?|signals?|matches?|groups?)\b",
    re.IGNORECASE,
)
# NMR integration notation: "6H singlet", "3H triplet", "1H (dd)", "2H, m" -- classic
# hallucination surface where the model asserts a proton count for a peak w/o tool support.
# Match: bare integer immediately preceded by non-alphanumeric (so we don't catch H3 inside CH3),
# followed by an H, then an NMR multiplicity word (singlet/doublet/triplet/multiplet or one of
# s/d/t/m/q/dd/dt/td/ddd) or an "integrating" cue.
_NMR_INTEGRATION_RE = re.compile(
    r"(?<![A-Za-z0-9])(\d{1,3})H(?=[\s,(]+(?:singlet|doublet|triplet|multiplet|quartet|"
    r"integrating|integr\.|s\b|d\b|t\b|m\b|q\b|dd\b|dt\b|td\b|ddd\b))",
    re.IGNORECASE,
)
# Similarity / sub-similarity floats in <think>: 0.463, 0.7902, .60 ...
_SIM_RE = re.compile(r"(?<![0-9.])(0?\.\d{2,4})(?![0-9])")
# key=value integer / float forms our own template emits, must also be sourced
_KV_INT_RE = re.compile(r"\b(?:candidate_count|experimental_(?:distinct_signals|total_nH)|"
                        r"n_pred_[ch]|n_exp_[ch]|n_matches|n_groups)\s*=\s*(\d{1,3})\b")
_KV_FLOAT_RE = re.compile(r"\b(?:similarity|c_sim|h_sim)\s*=\s*(0?\.\d{2,4})\b")


def _extract_think_blocks(assistant_content: str) -> List[str]:
    return re.findall(r"<think>(.*?)</think>", assistant_content, flags=re.DOTALL)


def _validate_think_number_provenance(messages: List[Dict[str, str]]) -> List[str]:
    """Return list of violation descriptions. Empty means pass.

    Rule (middle strictness):
      - integer atom counts and similarity floats used inside <think> must be substring-findable
        inside SOME <tool_response> JSON that appeared BEFORE the think.
      - delta-ppm floats (e.g. "3.84 ppm") are whitelisted -- they come from input spectrum.
    """
    violations: List[str] = []
    prior_tool_blob = ""
    for i, m in enumerate(messages):
        if m["role"] == "tool":
            prior_tool_blob += "\n" + (m.get("content") or "")
            continue
        if m["role"] != "assistant":
            continue
        content = m.get("content") or ""
        for think in _extract_think_blocks(content):
            # Drop delta-ppm spans before we look for bare integers / floats to avoid false
            # positives (e.g. "3.84 ppm" contains "84").
            scrubbed = re.sub(r"\b\d{1,3}\.\d{1,2}\s*ppm\b", "[ppm]", think, flags=re.IGNORECASE)
            # 1) integer counts: number must appear in prior tool JSON
            for num, _unit in _INT_COUNT_RE.findall(scrubbed):
                if num not in prior_tool_blob:
                    violations.append(f"msg{i}: integer '{num} {_unit}' unsourced")
            # 1b) NMR integration notation ("6H singlet"): integer must be sourced too
            for num in _NMR_INTEGRATION_RE.findall(scrubbed):
                if num not in prior_tool_blob:
                    violations.append(f"msg{i}: NMR integration '{num}H' unsourced")
            # 2) key=value integers our own template emits
            for num in _KV_INT_RE.findall(scrubbed):
                if num not in prior_tool_blob:
                    violations.append(f"msg{i}: kv-int '{num}' unsourced")
            # 3) similarity / c_sim / h_sim floats
            for f in _SIM_RE.findall(scrubbed):
                # tool JSON stores floats without leading zero-strip variance; try both forms
                if f in prior_tool_blob:
                    continue
                # tool JSONs write "0.4632" style — normalize
                norm = f if f.startswith("0.") else "0" + f
                if norm in prior_tool_blob:
                    continue
                violations.append(f"msg{i}: similarity/float '{f}' unsourced")
            for f in _KV_FLOAT_RE.findall(scrubbed):
                if f not in prior_tool_blob and ("0" + f) not in prior_tool_blob:
                    violations.append(f"msg{i}: kv-float '{f}' unsourced")
    return violations


# ---------------------------------------------------------------------------
# Main converter (v5)
# ---------------------------------------------------------------------------
def convert_one(rec: Dict[str, Any], pred_cache: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    reasoning = rec.get("reasoning") or {}
    steps = reasoning.get("steps")
    if not steps:
        return None
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

    # Collect tool calls and Gemini prose per step.
    raw_calls: List[Dict[str, Any]] = []
    step_prose: List[str] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        tc = _map_step_to_toolcall(step)
        if tc is not None:
            raw_calls.append(tc)
        obs = step.get("observation") or ""
        step_prose.append(str(obs))
    calls = _dedup_toolcalls(raw_calls)
    if not calls:
        return None

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

    loop = asyncio.get_event_loop()
    evidence: List[Tuple[str, Dict[str, Any]]] = []
    prev_obs_json: Optional[str] = None
    prev_tool_name: Optional[str] = None
    first_hypothesis_prose = step_prose[0] if step_prose else ""

    for i, tc in enumerate(calls):
        tool_name = tc["name"]
        if i == 0:
            think = _first_pre_tool_think(first_hypothesis_prose, tool_name)
        else:
            think = _post_tool_think(prev_tool_name or "prev_tool", prev_obs_json or "{}", tool_name)
        assistant = f"<think>{think}</think>\n<tool_call>{json.dumps(tc, ensure_ascii=False)}</tool_call>"
        messages.append({"role": "assistant", "content": assistant})
        obs_json = loop.run_until_complete(_exec_tool(runtime, tc))
        messages.append({"role": "tool", "name": tool_name, "content": obs_json})
        try:
            obs = json.loads(obs_json)
        except Exception:
            obs = {}
        evidence.append((tool_name, obs))
        prev_obs_json = obs_json
        prev_tool_name = tool_name

    # final <answer>
    final_answer = reasoning.get("final_answer")
    confidence = reasoning.get("confidence", 0.9)
    key_evidence = reasoning.get("key_evidence", "")
    ans_obj = {"final_answer": final_answer, "confidence": confidence, "key_evidence": key_evidence}
    final_think = _final_think(evidence, final_answer)
    messages.append({
        "role": "assistant",
        "content": f"<think>{final_think}</think>\n<answer>{json.dumps(ans_obj, ensure_ascii=False)}</answer>",
    })

    contradiction = _detect_contradiction(final_answer, [e[1] for e in evidence])
    provenance_violations = _validate_think_number_provenance(messages)

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
            "provenance_violations": provenance_violations,
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cot", required=True)
    ap.add_argument("--pred_cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rejects_out", default=None,
                    help="where to dump rows that failed provenance / contradiction (jsonl)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--drop_contradictions", action="store_true")
    ap.add_argument("--drop_provenance_violations", action="store_true", default=True,
                    help="drop rows whose <think> has unsourced numbers (default: on)")
    ap.add_argument("--keep_provenance_violations", dest="drop_provenance_violations",
                    action="store_false")
    args = ap.parse_args()

    with open(args.pred_cache) as f:
        pred_cache = json.load(f)
    print(f"[cache] {len(pred_cache)} predictions", flush=True)

    n_total = n_structured = n_converted = 0
    n_contra = n_prov = 0
    kept: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
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

            reason: Optional[str] = None
            if row["meta"]["contradiction"]:
                n_contra += 1
                if args.drop_contradictions:
                    reason = f"contradiction:{row['meta']['contradiction']}"
            if row["meta"]["provenance_violations"]:
                n_prov += 1
                if args.drop_provenance_violations and reason is None:
                    reason = "provenance_violation"

            if reason:
                row["meta"]["reject_reason"] = reason
                rejected.append(row)
            else:
                kept.append(row)
                n_converted += 1
            if args.limit and n_converted >= args.limit:
                break

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    if args.rejects_out:
        Path(args.rejects_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.rejects_out, "w") as f:
            for r in rejected:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[convert] total={n_total} structured={n_structured} kept={n_converted}", flush=True)
    print(f"[filter ] contradictions={n_contra} ({100.0*n_contra/max(n_structured,1):.1f}%)", flush=True)
    print(f"[filter ] provenance_violations={n_prov} ({100.0*n_prov/max(n_structured,1):.1f}%)", flush=True)
    print(f"[write  ] {len(kept)} -> {args.out}   rejects={len(rejected)}", flush=True)


if __name__ == "__main__":
    main()
