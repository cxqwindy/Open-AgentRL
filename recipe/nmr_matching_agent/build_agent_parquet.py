#!/usr/bin/env python3
"""Build verl RL parquet for the MULTI-TURN NMR matching tool agent (M4).

Difference from recipe/nmr_matching/build_matching_parquet.py:
  - Prompt is [system, user] where user content is the JSON case dict expected
    by NmrMatchingTool.create (candidate_smiles + peaks), NOT the alpaca text.
  - System prompt is exactly what standalone_infer_matching.py sends
    (`prompt.build_system_prompt()`), so RL rollout matches the M3 SFT training
    distribution turn-for-turn.
  - reward_model.ground_truth carries {label, spectral_sim, candidate_smiles,
    gt_smiles, negative_type} exactly like the single-turn parquet, so
    reward_matching_agent.compute_score has everything it needs.

Usage (recommended: reuse the balanced RL train set, just re-shape prompts):
    python build_agent_parquet.py \
        --from_parquet ../nmr_matching/data/rl_train_expanded.parquet \
        --out data/rl_train_expanded_agent.parquet

    python build_agent_parquet.py \
        --from_alpaca /fs_mol/cxq/Xmaster_AgentRL/repair_sft/LLaMA-Factory/data/nmr_matching_test_1000.json \
        --out data/rl_val_1000_agent.parquet
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from prompt import build_system_prompt  # noqa: E402
from standalone_infer_matching import _pair_index_for, parse_alpaca_input, prepare_eval_record  # noqa: E402
from task_input import HARNESS_RUN_ID_KEY, task_to_lossless_text  # noqa: E402

DATA_SOURCE = "nmr_matching_v4_agent"

_INSTRUCTION_RE = re.compile(
    r"Given NMR spectral data and a candidate molecular structure.*?structured JSON object\.\s*\n\n",
    re.DOTALL,
)


def _alpaca_input_to_case(input_text: str) -> dict:
    """Strip the leading instruction and reuse standalone_infer's parser."""
    stripped = _INSTRUCTION_RE.sub("", input_text, count=1)
    return parse_alpaca_input(stripped)


def _as_list(x):
    if x is None:
        return None
    try:
        return list(x)
    except TypeError:
        return None


def _as_dict(x):
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        return json.loads(x)
    try:
        return dict(x)
    except Exception:
        return None


_PAIR_ID_KEYS = ("pair_id", "_harness_pair_id")
_RUN_ID_KEYS = ("run_id", "_harness_run_id")


def _pair_id_value(value):
    """Return a harness pair id, or None when the value is not one.

    Bool is rejected. A numpy integer survives a parquet round trip and is
    still an integer identity, not a new id scheme.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if type(value).__module__ == "numpy" and type(value).__name__.startswith("int"):
        return int(value)
    return None


def _consistent_value(extra: dict, keys, parse, noun: str):
    present = [(key, extra[key]) for key in keys if key in extra]
    if not present:
        return None
    parsed = []
    for key, value in present:
        got = parse(value)
        if got is None:
            raise ValueError(f"invalid {key}: {value!r}")
        parsed.append((key, got))
    if len({item[1] for item in parsed}) > 1:
        detail = ", ".join(f"{key}={value!r}" for key, value in parsed)
        raise ValueError(f"conflicting harness {noun}: {detail}")
    return parsed[0][1]


def _parse_pair_id(value):
    got = _pair_id_value(value)
    if got is None or got < 0:
        return None
    return got


def _source_identity(extra: dict):
    """Read harness identity without turning a missing id into 0 or null."""
    pair_id = _consistent_value(
        extra,
        _PAIR_ID_KEYS,
        _parse_pair_id,
        "pair ids",
    )
    run_id = _consistent_value(
        extra,
        _RUN_ID_KEYS,
        lambda value: value if isinstance(value, str) and value else None,
        "run ids",
    )
    if pair_id is None and run_id is None:
        return None, None, "unverified"
    if pair_id is None or run_id is None:
        return pair_id, run_id, "incomplete"
    return pair_id, run_id, "verified"


def _json_object(value, where: str) -> dict:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{where} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{where} must be an object, got {type(value).__name__}")
    return value


def _layer_sidecar(existing, conversion) -> dict:
    """Keep original headers and conflicts. The new parse is a separate layer."""
    existing = _json_object(existing, "input_sidecar")
    if "source" in existing and "conversion" in existing and "headers" not in existing:
        source = existing["source"]
        if not isinstance(source, dict):
            source = _json_object(source, "input_sidecar.source")
    else:
        source = existing
    return {"source": copy.deepcopy(source), "conversion": conversion}


def _row_from_parquet_row(row) -> dict:
    """Convert a single-turn matching parquet row into an agent-shape row.

    An existing harness pair id and run id are kept. A local ``index`` is
    metadata, not a global id. The integer passed to ``prepare_eval_record``
    is only the call argument and is not written back as a verified identity.
    """
    prompt = _as_list(row["prompt"])
    if not prompt:
        raise ValueError(f"empty prompt in row: {row}")
    last = prompt[-1]
    user_content = last["content"] if isinstance(last, dict) else _as_dict(last)["content"]

    stripped = _INSTRUCTION_RE.sub("", user_content, count=1)
    extra = _as_dict(row["extra_info"]) or {}
    if not isinstance(extra, dict):
        raise ValueError("extra_info must be an object when provided")
    pair_id, run_id, identity_status = _source_identity(extra)
    # prepare_eval_record requires an integer. 0 here is not a global id.
    call_record = {"input": stripped}
    if run_id is not None:
        call_record[HARNESS_RUN_ID_KEY] = run_id
    prepared = prepare_eval_record(call_record, 0 if pair_id is None else pair_id)
    case = prepared["inference_input"]
    if not case.get("candidate_smiles"):
        raise ValueError("task candidate is missing; reward ground_truth is not a candidate source")
    rm = _as_dict(row["reward_model"]) or {}
    gt = _as_dict(rm.get("ground_truth")) or {}
    metadata = copy.deepcopy(extra)
    metadata["identity_status"] = identity_status
    for key in _PAIR_ID_KEYS:
        metadata.pop(key, None)
    for key in _RUN_ID_KEYS:
        metadata.pop(key, None)
    if pair_id is not None:
        metadata["pair_id"] = pair_id
    if run_id is not None:
        metadata["run_id"] = run_id
    if "input_sidecar" in extra and extra.get("input_sidecar") not in (None, ""):
        metadata["input_sidecar"] = _layer_sidecar(extra["input_sidecar"], prepared["input_sidecar"])
    else:
        metadata["input_sidecar"] = prepared["input_sidecar"]

    system_prompt = build_system_prompt()
    return {
        "data_source": DATA_SOURCE,
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task_to_lossless_text(case)},
        ],
        "ability": row.get("ability", "NMR"),
        "reward_model": {"ground_truth": gt},
        "extra_info": metadata,
    }


def _row_from_alpaca(idx: int, rec: dict) -> dict:
    """Use the task schema. ``_gt_smiles`` is reward metadata, not the candidate.

    ``_candidate_smiles`` is checked against the task candidate and never replaces it.
    """
    if not isinstance(rec, dict):
        raise ValueError(f"row {idx}: record is not an object")
    if not rec.get("input") and not rec.get("candidate_smiles") and not rec.get("candidate"):
        raise ValueError(f"row {idx}: task candidate is missing; _gt_smiles is not a candidate source")
    prepared = prepare_eval_record(rec, _pair_index_for(rec, idx))
    case = prepared["inference_input"]
    meta_candidate = rec.get("_candidate_smiles")
    if meta_candidate and meta_candidate != case.get("candidate_smiles"):
        raise ValueError(
            f"row {idx}: _candidate_smiles {meta_candidate!r} conflicts with task candidate "
            f"{case.get('candidate_smiles')!r}"
        )
    gt = {
        "label": rec.get("_label", 0),
        "negative_type": rec.get("_negative_type", "unknown"),
        "candidate_smiles": rec.get("_candidate_smiles"),
        "gt_smiles": rec.get("_gt_smiles"),
        "spectral_sim": None,
    }
    return {
        "data_source": DATA_SOURCE,
        "prompt": [
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": task_to_lossless_text(case)},
        ],
        "ability": "NMR",
        "reward_model": {"ground_truth": gt},
        "extra_info": {
            "index": idx,
            "pair_id": prepared["pair_id"],
            "run_id": prepared["run_id"],
            "label": gt["label"],
            "negative_type": gt["negative_type"],
            "input_sidecar": prepared["input_sidecar"],
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from_parquet", help="reshape an existing single-turn matching parquet")
    ap.add_argument("--from_alpaca", help="build from LLaMA-Factory Alpaca JSON (nmr_matching_test_1000.json)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = []
    if args.from_parquet:
        df = pd.read_parquet(args.from_parquet)
        for _, row in df.iterrows():
            rows.append(_row_from_parquet_row(row))
    elif args.from_alpaca:
        data = json.load(open(args.from_alpaca))
        for i, rec in enumerate(data):
            rows.append(_row_from_alpaca(i, rec))
    else:
        raise SystemExit("must pass --from_parquet or --from_alpaca")

    out = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out)
    print(f"[write] {args.out}  rows={len(out)}")
    print(f"        cols: {list(out.columns)}")
    if len(out) > 0:
        r0 = out.iloc[0]
        print(f"        prompt[0].role = {r0['prompt'][0]['role']}  (len={len(r0['prompt'][0]['content'])})")
        print(f"        prompt[1].role = {r0['prompt'][1]['role']}  (content head: {r0['prompt'][1]['content'][:120]}...)")


if __name__ == "__main__":
    main()
