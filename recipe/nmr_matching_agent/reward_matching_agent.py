"""Reward function for multi-turn NMR MATCHING tool agent GRPO (M4).

Task shape (see recipe/nmr_matching_agent/README.md — M3 SFT already hit 81.4% on
test_1000). The rollout produces a full Hermes-format transcript:
    <think>...</think>
    <tool_call>{"name":"...","arguments":{...}}</tool_call>
    <tool_response>...</tool_response>
    ... (up to `max_turns`) ...
    <answer>{"final_answer":"Yes"|"No","confidence":<float>,"key_evidence":"..."}</answer>

`solution_str` is the concatenated assistant+tool text; ground truth carries
{label:0/1, negative_type, spectral_sim, candidate_smiles, gt_smiles}.

Reward decomposition (per rollout, all summed):
  R_correct           +1 if parsed final_answer matches label; WRONG_ANSWER_REWARD if wrong
                      (parseable); UNKNOWN_ANSWER_REWARD if not parseable.
  R_format           +R_FORMAT_WEIGHT if the transcript contains a valid <answer>{...}</answer>
                      with a "final_answer" field.
  R_tool_use         +R_TOOL_USE_REWARD if at least one deterministic tool call happened.
                      Prevents the model from skipping tools entirely under RL pressure.
  R_tool_diversity   +R_TOOL_DIVERSITY_REWARD * (unique_tool_types - 1) capped at 2 extra
                      tool types beyond the first. Fights the SFT mode-collapse where the
                      model always emits (count_atoms, spectral_match, answer). Rewards a
                      third tool (check_substructure / functional_group_scan) being invoked.
  R_calib            R_CALIB_WEIGHT * (1 - |confidence - spectral_sim|) applied ONLY when
                      correct AND spectral_sim is non-degenerate. Same rationale as the
                      single-turn reward (see recipe/nmr_matching/reward_matching.py).

Set individual weights to 0.0 to disable a component.

verl entry point: `compute_score(data_source=, solution_str=, ground_truth=, extra_info=)`.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional, Set, Tuple

# ---- Reward weights (tune here) --------------------------------------------
R_FORMAT_WEIGHT = 0.2
R_TOOL_USE_REWARD = 0.2
R_TOOL_DIVERSITY_REWARD = 0.15   # per extra unique tool type beyond the first, cap at +0.3
R_TOOL_DIVERSITY_CAP_EXTRA = 2
R_CALIB_WEIGHT = 0.0             # keep 0 until spectral_sim signal is verified discriminative
WRONG_ANSWER_REWARD = 0.0
UNKNOWN_ANSWER_REWARD = -1.0
NO_TOOL_PENALTY = -0.4           # if zero tool calls, subtract this (encourages tool use)

TOOL_NAMES = {"count_atoms", "spectral_match", "check_substructure", "functional_group_scan", "peak_assignment"}

_ANSWER_RE = re.compile(r"<answer>\s*([\s\S]*?)\s*</answer>", re.IGNORECASE)
_TOOL_CALL_RE = re.compile(
    r'<tool_call>\s*(\{[\s\S]*?"name"[\s\S]*?\})\s*</tool_call>', re.IGNORECASE
)


def _parse_answer_block(text: str) -> Tuple[str, Optional[float], bool]:
    """Return (final_answer, confidence, has_valid_json_block).

    final_answer: "Yes" / "No" / "" (empty when unparseable).
    """
    m = _ANSWER_RE.search(text or "")
    if not m:
        return "", None, False
    body = m.group(1).strip()
    try:
        obj = json.loads(body)
    except Exception:
        a = re.search(r'"final_answer"\s*:\s*"(Yes|No)"', body)
        c = re.search(r'"confidence"\s*:\s*([0-9]*\.?[0-9]+)', body)
        ans = a.group(1) if a else ""
        conf = float(c.group(1)) if c else None
        return ans, conf, bool(a)
    ans = obj.get("final_answer", "") if isinstance(obj, dict) else ""
    conf = obj.get("confidence") if isinstance(obj, dict) else None
    if not isinstance(conf, (int, float)):
        conf = None
    return (ans if ans in ("Yes", "No") else ""), conf, isinstance(obj, dict)


def _collect_tools(text: str) -> Tuple[int, Set[str]]:
    """Return (n_tool_calls, {tool_name, ...}) parsed from the transcript."""
    tools: Set[str] = set()
    n = 0
    for m in _TOOL_CALL_RE.finditer(text or ""):
        try:
            j = json.loads(m.group(1))
        except Exception:
            continue
        name = str(j.get("name") or "")
        if name in TOOL_NAMES:
            tools.add(name)
            n += 1
    return n, tools


def compute_score(
    data_source: Optional[str] = None,
    solution_str: str = "",
    ground_truth: Optional[dict] = None,
    extra_info: Optional[dict] = None,
    **_kwargs: Any,
) -> dict:
    ground_truth = ground_truth or {}
    label = ground_truth.get("label")
    spectral_sim = ground_truth.get("spectral_sim")

    ans, conf, has_json = _parse_answer_block(solution_str)
    n_tool_calls, tool_types = _collect_tools(solution_str)

    # R_correct
    if ans == "":
        r_correct = UNKNOWN_ANSWER_REWARD
    else:
        pred_label = 1 if ans == "Yes" else 0
        r_correct = 1.0 if pred_label == label else WRONG_ANSWER_REWARD

    # R_format
    r_format = R_FORMAT_WEIGHT if has_json else 0.0

    # Tool-use signals
    if n_tool_calls == 0:
        r_tool_use = NO_TOOL_PENALTY
        r_tool_diversity = 0.0
    else:
        r_tool_use = R_TOOL_USE_REWARD
        extra_types = max(0, len(tool_types) - 1)
        extra_types = min(extra_types, R_TOOL_DIVERSITY_CAP_EXTRA)
        r_tool_diversity = R_TOOL_DIVERSITY_REWARD * extra_types

    # R_calib (only when correct AND signal available)
    r_calib = 0.0
    if R_CALIB_WEIGHT > 0.0 and r_correct == 1.0 and isinstance(conf, (int, float)) \
       and isinstance(spectral_sim, (int, float)):
        r_calib = R_CALIB_WEIGHT * (1.0 - abs(float(conf) - float(spectral_sim)))

    score = r_correct + r_format + r_tool_use + r_tool_diversity + r_calib
    return {
        "score": float(score),
        "r_correct": float(r_correct),
        "r_format": float(r_format),
        "r_tool_use": float(r_tool_use),
        "r_tool_diversity": float(r_tool_diversity),
        "r_calib": float(r_calib),
        "n_tool_calls": int(n_tool_calls),
        "n_tool_types": int(len(tool_types)),
        "pred_answer": ans or "Unknown",
    }
