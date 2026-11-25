"""通过 HTTP 调用 v5 vLLM 服务的最小多轮客户端。

前提: vLLM 已在 --vllm-url (默认 http://localhost:8000/v1) 起了 v5 模型,
      served-model-name=nmr-matching。

用法:
  # 从 test 集拉 3 条跑
  python call_v5_vllm.py \
      --from-test /fs_mol/cxq/SpecMaster_jp/data/nmr_matching_test_2000.json --n 3

  # 单条案例:一段 alpaca 风格 JSON (candidate_smiles + c_nmr_peaks + h_nmr_peaks)
  python call_v5_vllm.py --input /tmp/case.json

  # curl 风格快速试:直接传 SMILES 和 peak 字符串
  python call_v5_vllm.py --smiles "CCOc1ccc(C=O)cc1" \
      --c-peaks "190.7,164.1,131.9,114.6,63.9,14.6" \
      --h-peaks "9.85(1H),7.82(2H),6.97(2H),4.10(2H),1.42(3H)"
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1]))  # Open-AgentRL/

from recipe.nmr_matching_agent.nmr_matching_runtime import NmrMatchingRuntime  # noqa: E402
from recipe.nmr_matching_agent.prompt import build_system_prompt  # noqa: E402
from recipe.nmr_matching_agent.standalone_infer_matching import (  # noqa: E402
    TOOL_SCHEMAS,
    attach_evaluation_meta,
    parse_action,
    prepare_eval_record,
)
from recipe.nmr_matching_agent.task_input import (  # noqa: E402
    has_experimental_signal,
    normalize_task_input,
    parse_c_peak_list,
    parse_h_peak_list,
    task_to_lossless_text,
)

DEFAULT_CACHE = str(_HERE / "prediction_cache.json")


async def chat_turn(client: httpx.AsyncClient, url: str, model: str,
                    messages: List[Dict[str, str]]) -> tuple:
    """Returns (text_for_parse, raw_msg_dict).

    When vLLM runs with --tool-call-parser hermes it returns structured tool_calls instead of
    raw <tool_call> text in content.  We reconstruct the <tool_call> block so parse_action still
    works, while keeping the original raw_msg for correct conversation history management
    (tool_calls field + tool_call_id on the follow-up tool response).
    """
    payload = {
        "model": model,
        "messages": messages,
        "tools": list(TOOL_SCHEMAS.values()),
        "temperature": 0,
        "max_tokens": 1024,
        "stop": ["</tool_call>", "</answer>"],
    }
    r = await client.post(f"{url}/chat/completions", json=payload, timeout=180)
    r.raise_for_status()
    j = r.json()
    choice = j["choices"][0]
    raw_msg = choice["message"]
    text = raw_msg.get("content") or ""

    # Structured tool_calls path (--tool-call-parser hermes active)
    tool_calls = raw_msg.get("tool_calls") or []
    if tool_calls:
        tc_fn = tool_calls[0]["function"]
        try:
            args_obj = json.loads(tc_fn["arguments"])
        except Exception:
            args_obj = {}
        text += f'<tool_call>{{"name": "{tc_fn["name"]}", "arguments": {json.dumps(args_obj)}}}</tool_call>'

    stop_reason = choice.get("stop_reason") or ""
    if isinstance(stop_reason, str) and stop_reason in ("</tool_call>", "</answer>"):
        text += stop_reason
    return text, raw_msg


def user_record_from_case(case: Dict[str, Any]) -> Dict[str, Any]:
    """Project a case onto the physical task, including an explicit total integral.

    ``nmr_frequency_nucleus`` stays unknown unless the case itself declares one.
    The spectrum predictor's default solvent and frequency are not changed here.
    """
    raw = {
        "candidate_smiles": case.get("candidate_smiles"),
        "c_nmr_peaks": case.get("c_nmr_peaks") if isinstance(case.get("c_nmr_peaks"), list) else [],
        "h_nmr_peaks": case.get("h_nmr_peaks") if isinstance(case.get("h_nmr_peaks"), list) else [],
        "nmr_solvent": case.get("nmr_solvent"),
        "nmr_frequency": case.get("nmr_frequency"),
    }
    if "nmr_frequency_nucleus" in case:
        raw["nmr_frequency_nucleus"] = case.get("nmr_frequency_nucleus")
    if "experimental_total_nH" in case:
        raw["experimental_total_nH"] = case.get("experimental_total_nH")
    return normalize_task_input(raw)


async def run_case(url: str, model: str, case: Dict[str, Any],
                   prediction_cache: Dict[str, Any],
                   max_turns: int = 8,
                   allow_remote_predict: bool = True) -> Dict[str, Any]:
    record = user_record_from_case(case)
    if not has_experimental_signal(record):
        raise ValueError("task has no experimental NMR peaks")
    runtime = NmrMatchingRuntime(record, prediction_cache=prediction_cache,
                                 allow_remote_predict=allow_remote_predict)
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": task_to_lossless_text(record)},
    ]
    transcript: List[Dict[str, Any]] = []
    async with httpx.AsyncClient() as client:
        for turn in range(1, max_turns + 1):
            text, raw_msg = await chat_turn(client, url, model, messages)
            action = parse_action(text)
            row = {"turn": turn, "assistant_text": text, "action_type": action["type"]}
            # Append original msg so tool_calls field is preserved for vLLM's next-turn formatting
            messages.append(raw_msg)
            if action["type"] == "answer":
                try:
                    parsed = json.loads(action["answer"])
                except Exception:
                    parsed = {"final_answer": action["answer"], "confidence": None,
                              "key_evidence": None, "_parse_error": True}
                row["answer"] = parsed
                transcript.append(row)
                return {"final": parsed, "turns": turn, "transcript": transcript,
                        "tool_sequence": [r.get("tool") for r in transcript if r.get("tool")]}
            if action["type"] != "tool":
                row["fail"] = action["type"]
                transcript.append(row)
                return {"final": None, "turns": turn, "transcript": transcript,
                        "fail_reason": action["type"]}
            name = action.get("name", "")
            args = action.get("arguments") or {}
            row["tool"] = name
            row["arguments"] = args
            obs = await runtime.run_tool_call_async(
                {"function": {"name": name, "arguments": args}})
            obs_text = obs.get("content") if isinstance(obs, dict) else str(obs)
            row["observation_preview"] = (obs_text or "")[:200]
            transcript.append(row)
            # Include tool_call_id to match vLLM's structured tool_calls in the assistant turn
            tool_call_id = ((raw_msg.get("tool_calls") or [{}])[0].get("id") or "call_0")
            messages.append({"role": "tool", "tool_call_id": tool_call_id,
                             "name": name, "content": obs_text or ""})
    return {"final": None, "turns": max_turns, "transcript": transcript,
            "fail_reason": "max_turns"}


def parse_h_peaks_arg(s: str) -> List[Dict[str, Any]]:
    """Parse ``9.85(1H),7.82(2H)``. An empty parenthesis is an unknown integral, not 1."""
    return parse_h_peak_list(s, "--h-peaks")


def build_case_from_flags(args) -> Dict[str, Any]:
    c_peaks = parse_c_peak_list(args.c_peaks, "--c-peaks") if args.c_peaks else []
    h_peaks = parse_h_peaks_arg(args.h_peaks) if args.h_peaks else []
    case: Dict[str, Any] = {
        "candidate_smiles": args.smiles,
        "c_nmr_peaks": c_peaks,
        "h_nmr_peaks": h_peaks,
        "nmr_solvent": args.solvent,
        "nmr_frequency": args.frequency,
    }
    if getattr(args, "experimental_total_nH", None) is not None:
        case["experimental_total_nH"] = args.experimental_total_nH
    return case


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vllm-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="nmr-matching")
    ap.add_argument("--cache-path", default=DEFAULT_CACHE)
    ap.add_argument("--max-turns", type=int, default=8)
    ap.add_argument("--allow-remote-predict", action="store_true", default=True)
    ap.add_argument("--no-remote-predict", dest="allow_remote_predict", action="store_false")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--from-test", help="alpaca test json (uses `input` field)")
    src.add_argument("--input", help="single JSON case file")
    src.add_argument("--smiles", help="candidate SMILES (使用 --c-peaks / --h-peaks 传峰)")
    ap.add_argument("--c-peaks", default="", help="逗号分隔 13C 位移,配合 --smiles")
    ap.add_argument("--h-peaks", default="",
                    help="逗号分隔 1H,格式 '9.85(1H),7.82(2H)',配合 --smiles")
    ap.add_argument("--solvent", default="CDCl3")
    ap.add_argument("--frequency", type=float, default=400.0,
                    help="Caller default only. It is not a measured nucleus assignment.")
    ap.add_argument("--experimental-total-nH", dest="experimental_total_nH", type=float, default=None)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--output", default="")
    args = ap.parse_args()

    cache = json.load(open(args.cache_path))
    print(f"[cache] loaded {len(cache)} entries from {args.cache_path}")

    def _prepare(rec: Dict[str, Any], index: int) -> Dict[str, Any]:
        item = prepare_eval_record(rec, index)
        # Historical v5 fallback: some rows stored the gold string only in output.
        # A present _label of 0 must win; output is not copied onto the result row.
        if "_label" not in rec and "label" not in rec and "output" in rec:
            item["evaluation_meta"]["_label"] = rec.get("output")
        return item

    items: List[Dict[str, Any]] = []
    if args.from_test:
        data = json.load(open(args.from_test))
        items = [_prepare(rec, i) for i, rec in enumerate(data[: args.n])]
    elif args.input:
        items = [_prepare(json.load(open(args.input)), 0)]
    else:
        items = [_prepare(build_case_from_flags(args), 0)]

    results = []
    for i, item in enumerate(items):
        case = item["inference_input"]
        t0 = time.time()
        res = asyncio.run(run_case(args.vllm_url, args.model, case, cache,
                                    max_turns=args.max_turns,
                                    allow_remote_predict=args.allow_remote_predict))
        res["latency_s"] = round(time.time() - t0, 2)
        res = attach_evaluation_meta(res, item)
        final = res.get("final") or {}
        print(f"\n=== case {i+1}/{len(items)} ===")
        print(f"  candidate      : {case['candidate_smiles'][:70]}")
        print(f"  gt / neg_type  : {res.get('_label')} / {res.get('_negative_type')}")
        print(f"  final_answer   : {final.get('final_answer')}")
        print(f"  confidence     : {final.get('confidence')}")
        print(f"  key_evidence   : {(final.get('key_evidence') or '')[:120]}")
        print(f"  turns          : {res['turns']}")
        print(f"  tool_sequence  : {[r.get('tool') for r in res['transcript'] if r.get('tool')]}")
        print(f"  latency        : {res['latency_s']} s")
        if res.get("fail_reason"):
            print(f"  FAIL           : {res['fail_reason']}")
        results.append(res)

    if args.output:
        Path(args.output).write_text(json.dumps(results, ensure_ascii=False, indent=2))
        print(f"\n[write] {args.output}")


if __name__ == "__main__":
    main()
