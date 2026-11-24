#!/usr/bin/env python3
"""Convert tool-call trajectories (from cot_to_trajectory.py) -> LLaMA-Factory sharegpt tool format.

cot_to_trajectory.py emits {messages:[{role,content,...}], meta:{...}} where assistant turns carry
raw Hermes <think>/<tool_call>/<answer> tags. LLaMA-Factory's sharegpt tool format instead expects
role-typed turns and lets the `qwen3` template wrap tool calls in tags itself:

    {
      "conversations": [
        {"from": "system",        "value": "<system prompt>"},
        {"from": "human",         "value": "<user JSON>"},
        {"from": "gpt",           "value": "<think>...</think>"},          # reasoning only
        {"from": "function_call", "value": "{\"name\":...,\"arguments\":{...}}"},
        {"from": "observation",   "value": "<tool result JSON>"},
        ... (repeat function_call/observation) ...
        {"from": "gpt",           "value": "<final reasoning><answer>{...}</answer>"}
      ],
      "tools": "<JSON array of tool schemas>"
    }

We split each assistant <think>+<tool_call> turn into a `gpt` (think) + `function_call` turn, and map
`tool` turns to `observation`. The final <answer> turn becomes a plain `gpt` turn (the answer is the
target completion). System prompt + tool schemas are attached so the template renders identically to
RL rollout.

Usage:
    python trajectory_to_llamafactory.py \
        --in recipe/nmr_matching_agent/data/tool_sft_v4.jsonl \
        --out /fs_mol/cxq/Xmaster_AgentRL/repair_sft/LLaMA-Factory/data/nmr_matching_tool_sft_v4.json \
        [--val_out .../nmr_matching_tool_sft_v4_val.json --val_frac 0.02]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List

TOOLCALL_RE = re.compile(r"<tool_call>\s*({[\s\S]*?})\s*</tool_call>", re.IGNORECASE)
THINK_RE = re.compile(r"<think>([\s\S]*?)</think>", re.IGNORECASE)

# Tool schemas attached to every sample (OpenAI function form; LF renders them via the qwen3 template).
TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "name": "count_atoms",
        "description": "Count carbon or hydrogen atoms in the candidate (exact RDKit) vs the experimental NMR signal count.",
        "parameters": {"type": "object", "properties": {
            "element": {"type": "string", "enum": ["C", "H"], "description": "'C' or 'H'."}}, "required": ["element"]},
    },
    {
        "name": "spectral_match",
        "description": "Forward-predict the candidate spectrum and score similarity (0..1) vs the experimental spectrum.",
        "parameters": {"type": "object", "properties": {
            "sigma_c": {"type": "number"}, "sigma_h": {"type": "number"}}, "required": []},
    },
    {
        "name": "check_substructure",
        "description": "Check whether a named functional group or raw SMARTS is present in the candidate.",
        "parameters": {"type": "object", "properties": {
            "pattern": {"type": "string"}}, "required": ["pattern"]},
    },
    {
        "name": "functional_group_scan",
        "description": "Report all known functional groups present in the candidate.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "peak_assignment",
        "description": "One-to-one Hungarian assignment of predicted and experimental C or H signals with bidirectional unmatched peaks and residual statistics.",
        "parameters": {"type": "object", "properties": {
            "nucleus": {"type": "string", "enum": ["C", "H"]}}, "required": []},
    },
]
TOOLS_JSON = json.dumps(TOOL_SCHEMAS, ensure_ascii=False)


def convert_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Produce a STRICTLY ALTERNATING sharegpt conversation:
        system, human, [function_call, observation]*, gpt(answer)
    LF requires odd turns in (human, observation) and even turns in (gpt, function_call).
    A `<think>` for a tool step is embedded INSIDE the function_call value -- LF's FunctionFormatter
    (tool_format='qwen') extracts the <think>...</think> and prepends it to the rendered <tool_call>,
    so reasoning and the call live in one assistant turn (no separate gpt turn, which would break
    alternation)."""
    conv: List[Dict[str, str]] = []
    for m in messages:
        role = m["role"]
        content = m.get("content", "")
        if role == "system":
            conv.append({"from": "system", "value": content})
        elif role == "user":
            conv.append({"from": "human", "value": content})
        elif role == "tool":
            conv.append({"from": "observation", "value": content})
        elif role == "assistant":
            tc = TOOLCALL_RE.search(content)
            if tc:
                think = THINK_RE.search(content)
                try:
                    call_json = json.dumps(json.loads(tc.group(1)), ensure_ascii=False)
                except Exception:
                    call_json = tc.group(1)
                # embed the think (if any) with the call; LF's FunctionFormatter (qwen tool_format)
                # extracts <think>...</think> and prepends it to the rendered <tool_call>. The
                # delimiters MUST match the qwen3 template's thought_words exactly:
                # ("<think>\n", "\n</think>\n\n") -- otherwise the regex misses and the think leaks
                # into the JSON parser.
                if think and think.group(1).strip():
                    value = f"<think>\n{think.group(1).strip()}\n</think>\n\n{call_json}"
                else:
                    value = call_json
                conv.append({"from": "function_call", "value": value})
            else:
                # final answer turn (contains <answer>...); the gpt completion
                conv.append({"from": "gpt", "value": content})
    return conv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--val_out", default="")
    ap.add_argument("--val_frac", type=float, default=0.0)
    args = ap.parse_args()

    rows = []
    with open(args.inp) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            conv = convert_messages(obj["messages"])
            rows.append({"conversations": conv, "tools": TOOLS_JSON})

    n = len(rows)
    n_val = int(n * args.val_frac) if args.val_out and args.val_frac > 0 else 0
    val = rows[:n_val]
    train = rows[n_val:]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(train, f, ensure_ascii=False)
    print(f"[write] {len(train)} train samples -> {args.out}")
    if n_val:
        with open(args.val_out, "w") as f:
            json.dump(val, f, ensure_ascii=False)
        print(f"[write] {len(val)} val samples -> {args.val_out}")

    # print a sample for eyeballing
    print("\n=== sample conversation (roles) ===")
    for turn in train[0]["conversations"]:
        print(f"  {turn['from']:>14}: {turn['value'][:80]}")


if __name__ == "__main__":
    main()
