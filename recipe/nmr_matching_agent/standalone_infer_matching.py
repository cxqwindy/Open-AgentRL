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
import copy
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recipe.nmr_matching_agent.evidence_reconstruction import (  # noqa: E402
    EvidenceReconstructionError,
    reconstruct_model_final_turn,
)
from recipe.nmr_matching_agent.prompt import build_system_prompt  # noqa: E402
from recipe.nmr_matching_agent.task_input import (  # noqa: E402
    C_PEAK_FIELDS,
    H_PEAK_FIELDS,
    HARNESS_PAIR_ID_KEY,
    HARNESS_RUN_ID_KEY,
    SCHEMA_VERSION,
    has_experimental_signal,
    normalize_task_input,
    parse_alpaca_input,
    parse_task_text,
    project_peaks,
    task_to_lossless_text,
    value_presence,
)

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


# Alpaca / lossless parsing lives in task_input.parse_task_text. The names
# below stay importable for the historical CLI and parquet builder.


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
    return [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": task_to_lossless_text(case)},
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


_PHYSICAL_KEYS = (
    "candidate_smiles",
    "candidate",
    "c_nmr_peaks",
    "h_nmr_peaks",
    "experimental_13c_peaks",
    "experimental_1h_peaks",
    "nmr_solvent",
    "solvent",
    "nmr_frequency",
    "nmr_frequency_nucleus",
    "nmr_freq",
    "experimental_total_nH",
)
_SAMPLE_ID_KEYS = ("case_id", "source_id", "idx", "id")


def _source_sample_id(rec: Dict[str, Any]) -> Any:
    for key in _SAMPLE_ID_KEYS:
        if key in rec and rec[key] not in (None, ""):
            return rec[key]
    return None


def _present_peaks(raw: Dict[str, Any], primary: str, alias: Optional[str], nucleus: str) -> List[Any]:
    """Return canonical peaks from one source. An empty list means 'not provided'."""
    peaks = raw.get(primary) or None
    if not peaks and alias:
        peaks = raw.get(alias) or None
    if not peaks:
        return []
    if not isinstance(peaks, list):
        raise TypeError(f"{primary} must be a list when provided")
    if not isinstance(peaks[0], dict):
        if nucleus == "C":
            peaks = [{"delta (ppm)": float(value)} for value in peaks]
        else:
            peaks = [{"delta": float(value), "centroid": float(value), "nH": None} for value in peaks]
    fields = C_PEAK_FIELDS if nucleus == "C" else H_PEAK_FIELDS
    return project_peaks(peaks, fields)


def _merge_peak_lists(text_peaks: List[Any], struct_peaks: List[Any], nucleus: str) -> List[Any]:
    """Keep whichever source has peaks. When both do, require agreement and keep the union.

    Within structured fields the historical alias priority still applies before
    this comparison: ``c_nmr_peaks`` wins over ``experimental_13c_peaks``, and
    ``h_nmr_peaks`` wins over ``experimental_1h_peaks``.
    """
    if not text_peaks:
        return struct_peaks
    if not struct_peaks:
        return text_peaks
    if len(text_peaks) != len(struct_peaks):
        raise ValueError(
            f"{nucleus} peak count differs between text input and structured fields "
            f"({len(text_peaks)} vs {len(struct_peaks)})"
        )
    merged: List[Any] = []
    for index, (text_peak, struct_peak) in enumerate(zip(text_peaks, struct_peaks)):
        if not isinstance(text_peak, dict) or not isinstance(struct_peak, dict):
            if text_peak != struct_peak:
                raise ValueError(f"{nucleus} peak {index} differs between text input and structured fields")
            merged.append(copy.deepcopy(text_peak))
            continue
        for key in set(text_peak) & set(struct_peak):
            if text_peak[key] != struct_peak[key]:
                raise ValueError(
                    f"{nucleus} peak {index} field {key!r} differs between text input and structured fields"
                )
        combined = dict(text_peak)
        combined.update(struct_peak)
        merged.append(combined)
    return merged


def _field_was_provided(source: Mapping[str, Any], key: str, explicit_null: bool) -> bool:
    if key not in source:
        return False
    if explicit_null:
        return True
    return source.get(key) is not None and source.get(key) != ""


def _resolved_scalars(structured: Mapping[str, Any]) -> Dict[str, Any]:
    """Apply the historical alias priority before comparing text with structured fields."""
    resolved: Dict[str, Any] = {}
    if "nmr_solvent" in structured or "solvent" in structured:
        resolved["nmr_solvent"] = structured.get("nmr_solvent") or structured.get("solvent")
    if "nmr_frequency" in structured or "nmr_freq" in structured:
        resolved["nmr_frequency"] = structured.get("nmr_frequency") or structured.get("nmr_freq")
    if "nmr_frequency_nucleus" in structured:
        resolved["nmr_frequency_nucleus"] = structured.get("nmr_frequency_nucleus")
    if "experimental_total_nH" in structured:
        resolved["experimental_total_nH"] = structured.get("experimental_total_nH")
    return resolved


def _merge_source_value(
    name: str,
    text: Optional[Dict[str, Any]],
    structured: Mapping[str, Any],
    key: str,
    *,
    text_explicit_null: bool,
) -> Any:
    """Keep a one-sided value. An explicit disagreement, including null versus known, raises."""
    text_has = text is not None and _field_was_provided(text, key, text_explicit_null)
    struct_has = _field_was_provided(structured, key, True)
    if text_has and struct_has and text.get(key) != structured.get(key):
        raise ValueError(f"{name} differs between text input and structured fields")
    if struct_has:
        return structured.get(key)
    if text_has and text is not None:
        return text.get(key)
    return None


def _compose_inference_input(rec: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Build the physical task from text and structured fields.

    Rules:
    - A JSON object string is the lossless task. Legacy Alpaca text is still
      read, including an explicit solvent, raw frequency, J, multiplicity, and
      range. A copied header frequency stays one string with nucleus unknown.
    - Structured aliases keep their old priority. Gold keys such as ``_label``,
      ``_gt_smiles``, and ``_candidate_smiles`` are not candidate or peak sources.
    - A candidate or peak list supplied by only one source is kept. Supplying the
      same candidate alongside text does not drop the text spectrum.
    - If both sources provide a value and they disagree, raise. Null versus a
      known value is a disagreement. The label is not used to choose a side.
    - Both nuclei without a usable peak is an error, not a task that can be sent
      to inference.
    """
    text_raw = rec.get("input")
    diagnostics: Dict[str, Any] = {
        "format": "structured",
        "warnings": [],
        "conflicts": [],
        "headers": [],
    }
    text = None
    if isinstance(text_raw, str) and text_raw.strip():
        text, diagnostics = parse_task_text(text_raw)
    structured = {key: rec[key] for key in _PHYSICAL_KEYS if key in rec}
    text_explicit_null = diagnostics.get("format") == "lossless_json"
    if text is None:
        task = normalize_task_input(structured)
        if not has_experimental_signal(task):
            raise ValueError("task has no experimental NMR peaks")
        return task, diagnostics

    text_candidate = text.get("candidate_smiles") or None
    structured_candidate = structured.get("candidate_smiles") or structured.get("candidate") or None
    if text_candidate and structured_candidate and text_candidate != structured_candidate:
        raise ValueError("candidate SMILES differs between text input and structured fields")
    candidate = structured_candidate or text_candidate
    if not candidate:
        raise ValueError("input must contain candidate_smiles")

    c_peaks = _merge_peak_lists(
        _present_peaks(text, "c_nmr_peaks", None, "C"),
        _present_peaks(structured, "c_nmr_peaks", "experimental_13c_peaks", "C"),
        "13C",
    )
    h_peaks = _merge_peak_lists(
        _present_peaks(text, "h_nmr_peaks", None, "H"),
        _present_peaks(structured, "h_nmr_peaks", "experimental_1h_peaks", "H"),
        "1H",
    )
    merged: Dict[str, Any] = {
        "candidate_smiles": candidate,
        "c_nmr_peaks": c_peaks,
        "h_nmr_peaks": h_peaks,
    }
    resolved = _resolved_scalars(structured)
    for key in ("nmr_solvent", "nmr_frequency", "nmr_frequency_nucleus", "experimental_total_nH"):
        text_has_key = bool(text_explicit_null and text is not None and key in text)
        value = _merge_source_value(
            key, text, resolved, key, text_explicit_null=text_has_key
        )
        if value is not None or text_has_key or key in resolved:
            merged[key] = value
    task = normalize_task_input(merged)
    if not has_experimental_signal(task):
        raise ValueError("task has no experimental NMR peaks")
    return task, diagnostics


def _pair_index_for(rec: Dict[str, Any], local_index: int) -> int:
    if HARNESS_PAIR_ID_KEY not in rec:
        return local_index
    value = rec[HARNESS_PAIR_ID_KEY]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"invalid {HARNESS_PAIR_ID_KEY}: {value!r}")
    return value


def _harness_run_id(rec: Dict[str, Any]) -> Optional[str]:
    if HARNESS_RUN_ID_KEY not in rec:
        return None
    value = rec[HARNESS_RUN_ID_KEY]
    if not isinstance(value, str) or not value:
        raise ValueError(f"invalid {HARNESS_RUN_ID_KEY}")
    return value


def prepare_eval_record(rec: Dict[str, Any], pair_index: int) -> Dict[str, Any]:
    """Split one dataset row into a model task and an external evaluation record.

    ``pair_index`` is the harness identity for this run. It is not a source
    ``case_id``/``idx`` and it is not copied into the task JSON.
    """
    if not isinstance(rec, dict):
        raise TypeError("evaluation record must be an object")
    if isinstance(pair_index, bool) or not isinstance(pair_index, int) or pair_index < 0:
        raise ValueError(f"invalid harness pair index: {pair_index!r}")
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
    run_id = _harness_run_id(rec)
    if run_id is not None:
        meta["run_id"] = run_id
    task, diagnostics = _compose_inference_input(rec)
    hash_material = {
        "input": rec.get("input") if isinstance(rec.get("input"), str) else None,
        "structured": {key: rec[key] for key in _PHYSICAL_KEYS if key in rec},
    }
    encoded = json.dumps(hash_material, ensure_ascii=False, sort_keys=True, allow_nan=False, default=str)
    return {
        "pair_id": pair_index,
        "run_id": run_id,
        "inference_input": task,
        "input_sidecar": {
            "schema_version": SCHEMA_VERSION,
            "input_sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
            "format": diagnostics.get("format"),
            "warnings": diagnostics.get("warnings") or [],
            "conflicts": diagnostics.get("conflicts") or [],
            "headers": diagnostics.get("headers") or [],
            "field_presence": {
                key: value_presence(rec, key)
                for key in (
                    "c_nmr_peaks",
                    "h_nmr_peaks",
                    "nmr_frequency",
                    "nmr_frequency_nucleus",
                    "experimental_total_nH",
                )
            },
            "runtime_projection": (
                "NmrMatchingRuntime loads peak lists with `or []`. A null list "
                "therefore becomes an empty list inside the tool. field_presence "
                "keeps the original missing/null/empty distinction and is not a "
                "substitute for the exact spectrum fingerprint."
            ),
        },
        "evaluation_meta": meta,
    }


def load_from_test(path: str, n: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load a full file or one shard.

    A shard written by the production splitter carries ``_harness_pair_id``.
    That global index is reused. A file without it uses the local row number,
    which is the global index only when the file is the full input.
    """
    with open(path) as handle:
        data = json.load(handle)
    if n is not None:
        data = data[:n]
    return [prepare_eval_record(rec, _pair_index_for(rec, index)) for index, rec in enumerate(data)]


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
    if item.get("run_id"):
        merged[HARNESS_RUN_ID_KEY] = item["run_id"]
    if "_gt_smiles" in meta:
        merged["_gt_smiles"] = meta.get("_gt_smiles")
    if "_candidate_smiles" in meta:
        merged["_candidate_smiles"] = meta.get("_candidate_smiles")
    return merged


def reassemble_eval_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Put prediction rows back into harness order. Duplicate ids are rejected."""
    ids = [row["_pair_id"] for row in results]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate harness pair id")
    return sorted(results, key=lambda row: row["_pair_id"])


def merge_eval_results(
    parts: List[List[Dict[str, Any]]],
    manifest: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Merge production shard outputs.

    Stamped rows are checked against the split manifest, including duplicate
    ids, missing rows, and rows from another run. Failed rows are kept.
    Rows without harness ids can be read only when no manifest is supplied.
    They stay in the given file order, that order is not a verified global
    reconstruction, and no ids are invented. A manifest plus unstamped rows
    is rejected instead of being treated as a successful reorder.
    """
    rows: List[Dict[str, Any]] = []
    for part in parts:
        if not isinstance(part, list):
            raise ValueError("each shard result must be a JSON list")
        rows.extend(part)

    def _stamped(row: Dict[str, Any]) -> bool:
        return isinstance(row, dict) and "_pair_id" in row and HARNESS_RUN_ID_KEY in row

    if rows and not all(isinstance(row, dict) for row in rows):
        raise ValueError("each result row must be an object")
    flags = [_stamped(row) for row in rows]
    if not any(flags):
        if manifest is not None:
            raise ValueError(
                "shard manifest was provided but result rows have no harness pair ids; "
                "reorder is not verified and global ids were not invented"
            )
        return {"rows": rows, "mode": "legacy_input_order", "reorder_verified": False}
    if not all(flags):
        raise ValueError("refusing to mix harness-stamped results with unstamped results")
    if manifest is None:
        raise ValueError("stamped results require the shard manifest to check for missing samples")

    run_ids = {row[HARNESS_RUN_ID_KEY] for row in rows}
    if len(run_ids) != 1:
        raise ValueError("refusing to merge results from different evaluation runs")
    run_id = next(iter(run_ids))
    if manifest.get("harness_run_id") != run_id:
        raise ValueError("result run id does not match the shard manifest")

    ids = [row["_pair_id"] for row in rows]
    if any(isinstance(value, bool) or not isinstance(value, int) for value in ids):
        raise ValueError("harness pair id must be an integer")
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate harness pair id")
    expected = list(manifest.get("pair_ids") or [])
    if len(expected) != len(set(expected)):
        raise ValueError("manifest pair ids are not unique")
    if set(ids) != set(expected):
        missing = sorted(set(expected) - set(ids))
        extra = sorted(set(ids) - set(expected))
        raise ValueError(f"result rows do not match the manifest (missing={missing}, extra={extra})")
    return {
        "rows": sorted(rows, key=lambda row: row["_pair_id"]),
        "mode": "harness_pair_id",
        "reorder_verified": True,
        "harness_run_id": run_id,
    }


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
