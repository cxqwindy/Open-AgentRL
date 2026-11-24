#!/usr/bin/env python3
"""Standalone multi-turn inference for the NMR MATCHING tool agent (no VeRL trainer).

Loads a HuggingFace checkpoint, builds the same system+user prompt shape used by RL rollout,
and dispatches tool calls through NmrMatchingTool. Proves the full chain:
    model -> <tool_call> -> parse -> runtime observation -> <answer>.

Input: a JSON file OR the alpaca-style nmr_matching_test_1000.json (use --from_test to pull N cases).
Each case needs candidate SMILES + experimental c/h peaks.

Usage:
  # single case from a JSON file:
  python recipe/nmr_matching_agent/standalone_infer_matching.py \
      --model_path <hf_ckpt> --input case.json --output result.json

  # first N cases straight from the test_1000 alpaca file:
  python recipe/nmr_matching_agent/standalone_infer_matching.py \
      --model_path <hf_ckpt> \
      --from_test /fs_mol/cxq/Xmaster_AgentRL/repair_sft/LLaMA-Factory/data/nmr_matching_test_1000.json \
      --n 2 --output result.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recipe.nmr_matching_agent.evidence_reconstruction import (  # noqa: E402
    EvidenceReconstructionError,
    reconstruct_model_final_turn,
)
from recipe.nmr_matching_agent.prompt import build_system_prompt  # noqa: E402
from recipe.nmr_matching_agent.task_input import normalize_task_input  # noqa: E402

ACTION_RE = re.compile(r"<tool_call>\s*({[\s\S]*?})\s*</tool_call>", re.IGNORECASE)
ANSWER_RE = re.compile(r"<answer>\s*([\s\S]*?)\s*</answer>", re.IGNORECASE)

# Matching tools exposed consistently with tool_config_matching.yaml + prompt.TOOL_DOC_FULL.
TOOL_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "count_atoms": {
        "type": "function",
        "function": {
            "name": "count_atoms",
            "description": (
                "Count carbon or hydrogen atoms in the candidate molecule (exact, via RDKit) and "
                "compare against the experimental NMR signal count."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "element": {"type": "string", "enum": ["C", "H"], "description": "'C' or 'H'."}
                },
                "required": ["element"],
            },
        },
    },
    "spectral_match": {
        "type": "function",
        "function": {
            "name": "spectral_match",
            "description": (
                "Forward-predict the candidate's NMR spectrum and score its similarity (0..1) "
                "against the experimental spectrum, with per-nucleus (13C/1H) sub-scores."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sigma_c": {"type": "number", "description": "Optional 13C match width in ppm (default 5.0)."},
                    "sigma_h": {"type": "number", "description": "Optional 1H match width in ppm (default 0.3)."},
                },
                "required": [],
            },
        },
    },
    "check_substructure": {
        "type": "function",
        "function": {
            "name": "check_substructure",
            "description": (
                "Check whether a functional group / substructure is present in the candidate "
                "(exact RDKit SMARTS match). Pass a group name (e.g. 'ester','methoxy','tms') or a "
                "raw SMARTS string."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "A group name or a raw SMARTS pattern."}
                },
                "required": ["pattern"],
            },
        },
    },
    "functional_group_scan": {
        "type": "function",
        "function": {
            "name": "functional_group_scan",
            "description": "Report all known functional groups present in the candidate at once.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    "peak_assignment": {
        "type": "function",
        "function": {
            "name": "peak_assignment",
            "description": (
                "Globally assign predicted and experimental peaks one-to-one with the Hungarian "
                "algorithm. Returns bidirectional unmatched peaks, residual statistics, large "
                "local contradictions, signal counts, and H group/integration vectors. Fixed "
                "policies are C match 5 ppm / large 8 ppm and H match 0.3 ppm / large 0.35 ppm."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "nucleus": {"type": "string", "enum": ["C", "H"],
                                "description": "'C' for 13C peaks (default), 'H' for 1H peaks."},
                },
                "required": [],
            },
        },
    },
}


# ---------------------------------------------------------------------------
# input parsing (candidate + peaks)
# ---------------------------------------------------------------------------
_C_NMR_RE = re.compile(r"13C NMR[^:]*:\s*([^\n]+)", re.IGNORECASE)
_H_NMR_RE = re.compile(r"1H NMR[^:]*:\s*([^\n]+)", re.IGNORECASE)
_CAND_RE = re.compile(r"Candidate SMILES:\s*(\S+)", re.IGNORECASE)
_H_PEAK_RE = re.compile(r"([0-9]+\.?[0-9]*)\s*\(([^)]*)\)")


def parse_alpaca_input(input_text: str) -> Dict[str, Any]:
    """Parse the alpaca `input` string (Candidate SMILES + 13C/1H NMR lines) into a case dict."""
    cand_m = _CAND_RE.search(input_text)
    candidate = cand_m.group(1).strip() if cand_m else ""

    c_peaks: List[Dict[str, Any]] = []
    cm = _C_NMR_RE.search(input_text)
    if cm and "N/A" not in cm.group(1):
        for tok in cm.group(1).split(","):
            tok = tok.strip()
            try:
                c_peaks.append({"delta (ppm)": float(tok)})
            except ValueError:
                continue

    h_peaks: List[Dict[str, Any]] = []
    hm = _H_NMR_RE.search(input_text)
    if hm and "N/A" not in hm.group(1):
        for delta_s, inner in _H_PEAK_RE.findall(hm.group(1)):
            try:
                delta = float(delta_s)
            except ValueError:
                continue
            nH = None
            m_nh = re.search(r"(\d+)\s*H", inner)
            if m_nh:
                nH = int(m_nh.group(1))
            h_peaks.append({"delta": delta, "centroid": delta, "nH": nH})
    return {"candidate_smiles": candidate, "c_nmr_peaks": c_peaks, "h_nmr_peaks": h_peaks}


def build_user_case(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Return the canonical physical task. Evaluation metadata is not copied."""
    return normalize_task_input(raw)


def parse_action(text: str) -> Dict[str, Any]:
    answer = ANSWER_RE.search(text or "")
    if answer:
        return {"type": "answer", "answer": answer.group(1).strip()}
    tool = ACTION_RE.search(text or "")
    if not tool:
        return {"type": "none"}
    try:
        obj = json.loads(tool.group(1))
    except Exception as exc:
        return {"type": "invalid_tool_json", "error": str(exc), "raw": tool.group(1)}
    return {
        "type": "tool",
        "name": obj.get("name"),
        "arguments": obj.get("arguments") if isinstance(obj.get("arguments"), dict) else {},
    }


DEFAULT_CACHE = str(Path(__file__).resolve().parent / "prediction_cache.json")


def load_tools(
    prediction_cache_path: str = DEFAULT_CACHE,
    allow_remote_predict: bool = True,
    h_tolerance: int = 2,
) -> Dict[str, Any]:
    """Instantiate all configured matching tools sharing one session manager.

    prediction_cache_path defaults to the recipe's full cache so spectral_match resolves from cache.
    allow_remote_predict defaults True here (unlike RL rollout) so an eval on an unseen candidate can
    still fall back to the remote predictor instead of returning ok=False.
    """
    from verl.tools.nmr_matching_tools import NmrMatchingSessionManager, NmrMatchingTool
    from verl.tools.schemas import OpenAIFunctionToolSchema

    manager = NmrMatchingSessionManager(
        prediction_cache_path=prediction_cache_path,
        allow_remote_predict=allow_remote_predict,
        h_tolerance=h_tolerance,
    )
    tools = {}
    for name, schema_dict in TOOL_SCHEMAS.items():
        schema = OpenAIFunctionToolSchema.model_validate(schema_dict)
        tools[name] = NmrMatchingTool({"session_manager": manager}, schema)
    return tools


def make_messages(case: Dict[str, Any]) -> List[Dict[str, str]]:
    task = normalize_task_input(case)
    return [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": json.dumps(task, ensure_ascii=False)},
    ]


def load_model_and_tokenizer(model_path: str):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    # Always resolve the architecture from config.json.  In particular, Qwen3
    # checkpoints are not architecture-compatible with Qwen2ForCausalLM
    # (Qwen3 has q/k norms and no q/k/v projection biases).
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    )

    configured_arch = (getattr(config, "architectures", None) or [type(model).__name__])[0]
    loaded_arch = type(model).__name__
    if getattr(config, "model_type", None) == "qwen3" and loaded_arch != "Qwen3ForCausalLM":
        raise RuntimeError(
            f"Qwen3 checkpoint resolved to incompatible model class {loaded_arch}; "
            "expected Qwen3ForCausalLM"
        )
    print(
        f"[model] config={type(config).__name__} model_type={getattr(config, 'model_type', None)} "
        f"configured_arch={configured_arch} loaded_arch={loaded_arch}"
    )
    model.config.use_cache = True
    model.eval()
    return tok, model


def generate_once(tok, model, messages, max_new_tokens=1024, use_tools=True) -> str:
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList

    class _Stop(StoppingCriteria):
        def __init__(self, prompt_len):
            self.prompt_len = prompt_len

        def __call__(self, input_ids, scores, **kw):
            tail = max(self.prompt_len, int(input_ids.shape[1]) - 64)
            text = tok.decode(input_ids[0, tail:], skip_special_tokens=False)
            return "</tool_call>" in text or "</answer>" in text or "<|im_end|>" in text

    # Match training-time prompt: LLaMA-Factory qwen3 template appends the tool schema block to
    # the system message via QWEN_TOOL_PROMPT. HF chat_template only does this when tools= is passed.
    if use_tools:
        tools_list = [v for v in TOOL_SCHEMAS.values()]
        prompt = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, tools=tools_list
        )
    else:
        # For key_evidence generation: no tools, just plain conversation
        prompt = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    prompt_len = int(inputs["input_ids"].shape[1])
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            repetition_penalty=1.05,
            pad_token_id=tok.eos_token_id,
            stopping_criteria=StoppingCriteriaList([_Stop(prompt_len)]),
        )
    return tok.decode(out[0, prompt_len:], skip_special_tokens=False)


async def run_agent(
    tok,
    model,
    case: Dict[str, Any],
    max_turns: int = 6,
    tools: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    tools = tools if tools is not None else load_tools()
    messages = make_messages(case)
    session_id = f"standalone_{int(time.time()*1000)}"
    transcript: List[Dict[str, Any]] = []
    final_answer = ""
    fail_reason: Optional[str] = None
    evidence_rebuilt = False
    evidence_rebuild_diagnostics: List[str] = []

    for tool in tools.values():
        await tool.create(instance_id=session_id, create_kwargs={"raw_prompt": messages, "episode_id": session_id})
    try:
        for turn in range(1, max_turns + 1):
            assistant_text = generate_once(tok, model, messages)
            action = parse_action(assistant_text)
            row = {"turn": turn, "assistant": assistant_text, "action": action}
            messages.append({"role": "assistant", "content": assistant_text})
            if action["type"] == "answer":
                transcript.append(row)  # retain the model's raw final turn for audit/debugging
                raw_answer = action.get("answer", "")
                if not raw_answer:
                    fail_reason = "empty_answer"
                    break
                try:
                    final_answer, evidence_rebuild_diagnostics = reconstruct_model_final_turn(
                        raw_answer, transcript
                    )
                    evidence_rebuilt = True
                except EvidenceReconstructionError as exc:
                    fail_reason = f"evidence_rebuild:{exc.code}"
                break
            if action["type"] != "tool":
                fail_reason = "unparseable_action" if action["type"] == "none" else action["type"]
                transcript.append(row)
                break
            name = action.get("name")
            if name not in tools:
                fail_reason = f"unknown_tool:{name}"
                transcript.append(row)
                break
            response, reward, metrics = await tools[name].execute(session_id, action.get("arguments") or {})
            row["tool_observation"] = response.text
            row["tool_metrics"] = metrics
            transcript.append(row)
            messages.append({"role": "tool", "name": name, "content": response.text or ""})
    finally:
        for tool in tools.values():
            try:
                await tool.release(session_id)
            except Exception:
                pass

    if not final_answer and fail_reason is None:
        fail_reason = "max_turns"

    return {
        "case_candidate": case.get("candidate_smiles"),
        "answer": final_answer,
        "fail_reason": fail_reason,
        "tool_sequence": [t["action"].get("name", t["action"].get("type")) for t in transcript],
        "transcript": transcript,
        "evidence_rebuilt": evidence_rebuilt,
        "evidence_rebuild_diagnostics": evidence_rebuild_diagnostics,
    }


def _template_tool_metrics(tool_name: str, observation: Dict[str, Any]) -> Dict[str, Any]:
    """Mirror the model-path metrics shape using values already present in the observation."""
    metrics: Dict[str, Any] = {"ok": "error" not in observation}
    if tool_name == "count_atoms":
        metrics["extra_cand_count"] = observation.get("candidate_count")
        if observation.get("element") == "C":
            metrics["extra_exp_signals"] = observation.get("experimental_distinct_signals")
        else:
            metrics["extra_exp_total"] = observation.get("experimental_total_nH")
            for key in ("exact_match", "difference", "tolerance"):
                if key in observation:
                    metrics[f"extra_{key}"] = observation.get(key)
    elif tool_name == "spectral_match":
        metrics.update({
            "extra_similarity": observation.get("similarity"),
            "extra_c_sim": observation.get("c_sim"),
            "extra_h_sim": observation.get("h_sim"),
            "extra_pred_ok": observation.get("ok"),
        })
    elif tool_name == "functional_group_scan":
        metrics["extra_n_groups"] = observation.get("n_groups")
    return metrics


def build_template_transcript(router_result, final_assistant_turn: str) -> List[Dict[str, Any]]:
    """Expand real router tool results into the same turn schema as model-path transcripts.

    The assistant text is deterministic narration; every tool_observation is the actual result
    already produced by RuleBasedRouter.run_all_tools. No model is loaded or called.
    """
    results = router_result.tool_results
    steps = [
        ("count_atoms", {"element": "C"}, results.count_c),
        ("count_atoms", {"element": "H"}, results.count_h),
        ("spectral_match", {}, results.spectral),
        ("functional_group_scan", {}, results.fg_scan),
    ]
    include_assignment = not (
        router_result.decision.assignment_vetoed
        and router_result.decision.assignment_shadow_mode
    )
    if include_assignment and getattr(results, "peak_c", None):
        steps.append(("peak_assignment", {"nucleus": "C"}, results.peak_c))
    if include_assignment and getattr(results, "peak_h", None):
        steps.append(("peak_assignment", {"nucleus": "H"}, results.peak_h))
    transcript: List[Dict[str, Any]] = []
    language = router_result.decision.template_language or "en"
    for turn, (tool_name, arguments, observation) in enumerate(steps, 1):
        call_label = (
            f"count_atoms({arguments['element']})" if tool_name == "count_atoms" else tool_name
        )
        if language == "zh":
            think = (
                f"调用 {call_label} 获取确定性路由证据。" if turn == 1
                else f"已记录上一工具的真实返回。下一步调用 {call_label} 获取更多证据。"
            )
        elif language == "ja":
            think = (
                f"{call_label} を呼び出し、決定論的なルーター証拠を取得する。" if turn == 1
                else f"前のツール結果を記録した。次に {call_label} を呼び出して証拠を追加する。"
            )
        else:
            think = (
                f"Call {call_label} to collect deterministic router evidence." if turn == 1
                else f"The previous tool result is recorded. Next: call {call_label} for more evidence."
            )
        call = {"name": tool_name, "arguments": arguments}
        assistant = (
            f"<think>{think}</think>\n"
            f"<tool_call>{json.dumps(call, ensure_ascii=False)}</tool_call>"
        )
        transcript.append({
            "turn": turn,
            "assistant": assistant,
            "action": {"type": "tool", "name": tool_name, "arguments": arguments},
            "tool_observation": json.dumps(observation, ensure_ascii=False, allow_nan=False),
            "tool_metrics": _template_tool_metrics(tool_name, observation),
        })

    answer_match = ANSWER_RE.search(final_assistant_turn)
    if not answer_match:
        raise ValueError("template final assistant turn has no <answer> block")
    transcript.append({
        "turn": len(steps) + 1,
        "assistant": final_assistant_turn,
        "action": {"type": "answer", "answer": answer_match.group(1).strip()},
    })
    return transcript


def generate_key_evidence_only(case: Dict[str, Any], router_result) -> str:
    """Generate key_evidence by extracting numerical values from think_block.

    Extracts similarity, C count, H count from the deterministic think_block and
    formats them into a standardized key_evidence string.
    """
    think_block = router_result.think_block
    conf = router_result.confidence

    # Extract numerical evidence from think_block using regex
    sim_match = re.search(r'similarity=([\d.]+)', think_block)
    c_match = re.search(r'candidate_count=(\d+).*?experimental_signals=(\d+)', think_block)
    h_match = re.search(r'candidate_count=(\d+).*?experimental_total_nH=(\d+)', think_block.split('count_atoms(H)')[1] if 'count_atoms(H)' in think_block else '')

    # Build evidence from extracted values
    evidence_parts = []

    if sim_match:
        sim_val = sim_match.group(1)
        evidence_parts.append(f"spectral_match similarity={sim_val}")

    if c_match:
        c_cand, c_exp = c_match.group(1), c_match.group(2)
        evidence_parts.append(f"C count consistent ({c_cand} atoms / {c_exp} signals)")

    if h_match:
        h_cand, h_exp = h_match.group(1), h_match.group(2)
        evidence_parts.append(f"H count consistent ({h_cand}/{h_exp} protons)")

    # Fallback: use tool results directly if regex fails
    if not evidence_parts:
        tr = router_result.tool_results
        if tr.spectral.get('ok'):
            evidence_parts.append(f"spectral_match similarity={tr.spectral['similarity']:.3f}")
        if tr.count_c.get('match'):
            evidence_parts.append(f"C count consistent ({tr.count_c['candidate_count']} atoms / {tr.count_c['experimental_distinct_signals']} signals)")
        if tr.count_h.get('match') is not False:
            evidence_parts.append(f"H count consistent ({tr.count_h['candidate_count']}/{tr.count_h.get('experimental_total_nH', '?')} protons)")

    evidence = "; ".join(evidence_parts) + "."

    print(f"[DEBUG] Extracted key_evidence: {evidence}")

    # Build complete answer
    full_answer = (
        f"<think>{think_block}</think>\n"
        f'<answer>{{"final_answer": "Yes", "confidence": {conf:.2f}, '
        f'"key_evidence": "{evidence}"}}</answer>'
    )
    print(f"[DEBUG] Full answer built, length={len(full_answer)}")
    return full_answer


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--input", help="single JSON case file (candidate_smiles + c/h peaks)")
    ap.add_argument("--from_test", help="alpaca nmr_matching_test_1000.json to pull cases from")
    ap.add_argument("--n", type=int, default=2, help="how many cases to run when using --from_test")
    ap.add_argument("--output", default="matching_agent_result.json")
    ap.add_argument("--max_turns", type=int, default=6)
    ap.add_argument("--use_router", action="store_true",
                    help="Enable rule-based router: bypass LLM for strong-positive cases")
    ap.add_argument("--router_config", default=None,
                    help="Path to router_config.json (default: router_config.json in recipe dir)")
    ap.add_argument("--router_sim_threshold", type=float, default=None,
                    help="Override both sim thresholds (back-compat shortcut)")
    ap.add_argument("--router_sim_threshold_both", type=float, default=None,
                    help="Override sim_threshold_both (C&H both match)")
    ap.add_argument("--router_sim_threshold_strict", type=float, default=None,
                    help="Override sim_threshold_strict (only one of C/H is available)")
    ap.add_argument("--router_template_variant_seed", type=int, default=None,
                    help="Override deterministic SHA-256 template wording seed")
    ap.add_argument("--router_skip_model_evidence", action="store_true",
                    help="Template path: skip model call for key_evidence generation (use hardcoded evidence)")
    return ap.parse_args()


_STRUCTURED_TASK_KEYS = (
    "candidate_smiles",
    "candidate",
    "c_nmr_peaks",
    "h_nmr_peaks",
    "experimental_13c_peaks",
    "experimental_1h_peaks",
)
_SAMPLE_ID_KEYS = ("case_id", "source_id", "idx", "id")


def _has_structured_task(rec: Dict[str, Any]) -> bool:
    return any(rec.get(key) for key in _STRUCTURED_TASK_KEYS)


def _source_sample_id(rec: Dict[str, Any]) -> Any:
    for key in _SAMPLE_ID_KEYS:
        if key in rec and rec[key] not in (None, ""):
            return rec[key]
    return None


def prepare_eval_record(rec: Dict[str, Any], pair_index: int) -> Dict[str, Any]:
    """Split one dataset row into a model task and an external evaluation record.

    ``pair_index`` is the harness identity. It stays unique when the source row
    has no id or repeats an id, and it is not copied into the task JSON.
    """
    if not isinstance(rec, dict):
        raise TypeError("evaluation record must be an object")
    if isinstance(rec.get("input"), str) and not _has_structured_task(rec):
        physical = parse_alpaca_input(rec["input"])
    else:
        physical = rec
    meta: Dict[str, Any] = {
        "sample_id": _source_sample_id(rec),
        "_label": rec.get("_label", rec.get("label")),
        "_negative_type": rec.get("_negative_type", rec.get("negative_type")),
    }
    if "_gt_smiles" in rec or "gt_smiles" in rec:
        meta["_gt_smiles"] = rec["_gt_smiles"] if "_gt_smiles" in rec else rec.get("gt_smiles")
    if "_candidate_smiles" in rec:
        meta["_candidate_smiles"] = rec.get("_candidate_smiles")
    if "output" in rec:
        meta["output"] = rec.get("output")
    if "teacher_pred" in rec:
        meta["teacher_pred"] = rec.get("teacher_pred")
    return {
        "pair_id": pair_index,
        "inference_input": normalize_task_input(physical),
        "evaluation_meta": meta,
    }


def load_from_test(path: str, n: Optional[int] = None) -> List[Dict[str, Any]]:
    with open(path) as handle:
        data = json.load(handle)
    if n is not None:
        data = data[:n]
    return [prepare_eval_record(rec, index) for index, rec in enumerate(data)]


def attach_evaluation_meta(result: Dict[str, Any], item: Dict[str, Any]) -> Dict[str, Any]:
    """Copy gold from the harness record after inference returns.

    The source ``output`` text is intentionally left off the prediction row:
    summary scripts treat a top-level ``output`` value as a predicted answer.
    """
    merged = dict(result)
    meta = item["evaluation_meta"]
    merged["_label"] = meta.get("_label")
    merged["_negative_type"] = meta.get("_negative_type")
    merged["_sample_id"] = meta.get("sample_id")
    merged["_pair_id"] = item["pair_id"]
    if "_gt_smiles" in meta:
        merged["_gt_smiles"] = meta.get("_gt_smiles")
    if "_candidate_smiles" in meta:
        merged["_candidate_smiles"] = meta.get("_candidate_smiles")
    return merged


def reassemble_eval_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Put sharded prediction rows back into dataset order."""
    return sorted(results, key=lambda row: row["_pair_id"])


def run_evaluation(
    items: List[Dict[str, Any]],
    *,
    use_router: bool,
    router_cfg: Optional[Dict[str, Any]],
    router_cache: Optional[Dict[str, Any]],
    max_turns: int,
    ensure_model,
    skip_model_evidence: bool,
) -> List[Dict[str, Any]]:
    """Run the CLI branch loop. Model and router receive ``inference_input`` only."""
    Router = None
    if use_router:
        from recipe.nmr_matching_agent.rule_based_router import RuleBasedRouter as Router

    results: List[Dict[str, Any]] = []
    n_template = 0
    n_model = 0
    for item in items:
        case = item["inference_input"]
        if use_router:
            router = Router(case, prediction_cache=router_cache, config=router_cfg)
            rr = router.analyze()
            if rr.route == "template":
                n_template += 1
                # Both legacy flag modes remain zero-model. Always use the selected
                # multilingual router variant so think/key_evidence stay in one language.
                if skip_model_evidence:
                    print("[DEBUG] Template path: using deterministic router evidence")
                else:
                    print("[DEBUG] Template path: using selected deterministic router variant")
                full_answer = rr.full_assistant_turn
                template_transcript = build_template_transcript(rr, full_answer)
                res: Dict[str, Any] = {
                    "case_candidate": case.get("candidate_smiles"),
                    "answer": full_answer,
                    "fail_reason": None,
                    "tool_sequence": [
                        turn["action"].get("name", turn["action"].get("type"))
                        for turn in template_transcript
                    ],
                    "transcript": template_transcript,
                    "evidence_rebuilt": True,
                    "evidence_rebuild_diagnostics": [],
                    "_transcript_source": "deterministic_router",
                    "_route": "template",
                    "_template_id": rr.decision.template_id,
                    "_template_variant_id": rr.decision.template_variant_id,
                    "_template_language": rr.decision.template_language,
                    "_template_variant_seed": router.template_variant_seed,
                    "_template_variant_digest": rr.decision.template_variant_digest,
                    "_c_available": rr.decision.c_available,
                    "_h_available": rr.decision.h_available,
                    "_c_match": rr.decision.c_match,
                    "_h_match": rr.decision.h_match,
                    "_assignment_vetoed": rr.decision.assignment_vetoed,
                    "_assignment_veto_reasons": rr.decision.assignment_veto_reasons,
                    "_assignment_shadow_mode": rr.decision.assignment_shadow_mode,
                    "_confidence": rr.decision.confidence,
                    "_skip_model": skip_model_evidence,
                }
            else:
                n_model += 1
                tok, model = ensure_model()
                res = asyncio.run(run_agent(tok, model, case, max_turns=max_turns))
                res["_route"] = "model"
                res["_transcript_source"] = "model"
                res["_assignment_vetoed"] = rr.decision.assignment_vetoed
                res["_assignment_veto_reasons"] = rr.decision.assignment_veto_reasons
                res["_assignment_shadow_mode"] = rr.decision.assignment_shadow_mode
        else:
            tok, model = ensure_model()
            res = asyncio.run(run_agent(tok, model, case, max_turns=max_turns))
            res["_route"] = "model"
            res["_transcript_source"] = "model"

        res = attach_evaluation_meta(res, item)
        results.append(res)
        answer_preview = res["answer"][:120] if res["answer"] else ""
        print(json.dumps({
            "candidate": (res["case_candidate"] or "")[:60],
            "label": res.get("_label"),
            "answer": answer_preview,
            "route": res.get("_route"),
            "tool_sequence": res["tool_sequence"],
            "fail_reason": res["fail_reason"],
        }, ensure_ascii=False, indent=2))

    if use_router:
        total = len(items)
        print(f"\n[router] template={n_template}/{total} ({100 * n_template // total if total else 0}%)  "
              f"model={n_model}/{total}")
    return results


def main():
    args = parse_args()
    if args.from_test:
        items = load_from_test(args.from_test, args.n)
    elif args.input:
        with open(args.input) as handle:
            items = [prepare_eval_record(json.load(handle), 0)]
    else:
        raise SystemExit("provide --input or --from_test")

    # --- router setup (optional) ---
    router_cache: Optional[Dict[str, Any]] = None
    router_cfg: Optional[Dict[str, Any]] = None
    if args.use_router:
        from recipe.nmr_matching_agent.rule_based_router import load_config, load_prediction_cache
        router_cfg = load_config(args.router_config)
        if args.router_sim_threshold is not None:
            # back-compat: set both thresholds to the same value
            router_cfg["sim_threshold_both"] = args.router_sim_threshold
            router_cfg["sim_threshold_strict"] = args.router_sim_threshold
        if args.router_sim_threshold_both is not None:
            router_cfg["sim_threshold_both"] = args.router_sim_threshold_both
        if args.router_sim_threshold_strict is not None:
            router_cfg["sim_threshold_strict"] = args.router_sim_threshold_strict
        if args.router_template_variant_seed is not None:
            router_cfg["template_variant_seed"] = args.router_template_variant_seed
        router_cache = load_prediction_cache()
        th_both = router_cfg.get("sim_threshold_both", 0.85)
        th_strict = router_cfg.get("sim_threshold_strict", 0.92)
        print(f"[router] enabled  sim_threshold_both={th_both}  sim_threshold_strict={th_strict}  "
              f"h_tol={router_cfg.get('h_tolerance', 2)}  "
              f"template_seed={router_cfg.get('template_variant_seed', 0)}  "
              f"languages={router_cfg.get('template_languages', ['en', 'zh', 'ja'])}")

    # Lazy model loading: only load if at least one case needs the model.
    tok = model = None

    def _ensure_model():
        nonlocal tok, model
        if tok is None:
            print(f"[model] loading {args.model_path} ...")
            tok, model = load_model_and_tokenizer(args.model_path)
        return tok, model

    results = run_evaluation(
        items,
        use_router=args.use_router,
        router_cfg=router_cfg,
        router_cache=router_cache,
        max_turns=args.max_turns,
        ensure_model=_ensure_model,
        skip_model_evidence=args.router_skip_model_evidence,
    )

    Path(args.output).write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"[write] {args.output}")

    # Debug: verify file was written
    if Path(args.output).exists():
        file_size = Path(args.output).stat().st_size
        print(f"[DEBUG] File written successfully, size={file_size} bytes")
        # Print first result's answer length for verification
        if results:
            print(f"[DEBUG] First result answer length: {len(results[0].get('answer', ''))}")
    else:
        print(f"[DEBUG] WARNING: File does not exist after write!")


if __name__ == "__main__":
    main()
