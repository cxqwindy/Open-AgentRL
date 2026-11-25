"""Entry regressions for the 02B-1 repair.

Expected values are written in the tests or come from the unchanged
``format_alpaca`` text. They are not produced by calling the parser twice.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from recipe.nmr_matching_agent import api_service
from recipe.nmr_matching_agent import build_agent_parquet
from recipe.nmr_matching_agent import call_v5_vllm
from recipe.nmr_matching_agent import standalone_infer_matching as infer
from recipe.nmr_matching_agent.nmr_matching_runtime import NmrMatchingRuntime
from recipe.nmr_matching_agent.rule_based_router import RuleBasedRouter
from recipe.nmr_matching_agent.task_input import (
    TASK_FIELDS,
    TaskInputError,
    describe_integrals,
    parse_h_peak_list,
)
from verl.tools.nmr_matching_tools import _build_record_from_user_json

V64_TRAIN = Path(
    "/fs_mol/cxq/Xmaster_AgentRL/repair_sft/LLaMA-Factory/data/nmr_matching_tool_sft_v6_4.json"
)
V64_VAL = Path(
    "/fs_mol/cxq/Xmaster_AgentRL/repair_sft/LLaMA-Factory/data/nmr_matching_tool_sft_v6_4_val.json"
)
FORMATTER = Path("/fs_mol/cxq/SpecMaster_jp/scripts/build_test_2000.py")

CONFLICT_TEXT = (
    "Candidate SMILES: CCO\n"
    "13C NMR (100 MHz, CDCl3): 57.8\n"
    "1H NMR (400 MHz, CDCl3): 1.20 (s, 3H)\n"
)


def _formatter():
    spec = importlib.util.spec_from_file_location("build_test_2000_unchanged", FORMATTER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.format_alpaca


def _peak(text: str) -> dict:
    peaks = parse_h_peak_list(text)
    assert len(peaks) == 1
    return peaks[0]


@pytest.mark.parametrize("nH", [0, 1, 3, 9, 10, 12, 18, 20, 100])
def test_r1_whole_nh_token_is_not_a_j_prefix(nH: int):
    peak = _peak(f"1.20 (s, J=7.1, {nH}H)")
    assert peak["category"] == "s"
    assert peak["j_values"] == "7.1"
    assert peak["nH"] == nH
    assert peak["delta"] == 1.2
    assert peak["centroid"] == 1.2
    assert "rangeMin" not in peak


@pytest.mark.parametrize(
    "text",
    [
        "1.20(s,J=7.1,12H)",
        "1.20 (s, J=7.1, 12H)",
        "1.20 (s, J = 7.1, 12H)",
        "1.20 (s, J=7.1,  12H)",
    ],
)
def test_r1_spacing_keeps_category_j_and_nh(text: str):
    peak = _peak(text)
    assert peak["category"] == "s"
    assert peak["j_values"] == "7.1"
    assert peak["nH"] == 12


def test_r1_single_and_multiple_j_with_and_without_optional_parts():
    multi = _peak("2.00 (dd, J=7.1, 2.0, 12H)")
    assert multi["category"] == "dd"
    assert multi["j_values"] == "7.1, 2.0"
    assert multi["nH"] == 12

    spaced = _peak("1.20 (app dd, J=7.1, 18H)")
    assert spaced["category"] == "app dd"
    assert spaced["j_values"] == "7.1"
    assert spaced["nH"] == 18

    no_category = _peak("1.20 (J=7.1, 10H)")
    assert "category" not in no_category
    assert no_category["j_values"] == "7.1"
    assert no_category["nH"] == 10

    no_j = _peak("1.20 (s, 9H)")
    assert no_j["category"] == "s"
    assert "j_values" not in no_j
    assert no_j["nH"] == 9

    no_nh = _peak("1.20 (s, J=7.1)")
    assert no_nh["category"] == "s"
    assert no_nh["j_values"] == "7.1"
    assert "nH" not in no_nh

    empty = _peak("9.85 ()")
    assert "category" not in empty
    assert "j_values" not in empty
    assert "nH" not in empty
    assert empty["delta"] == 9.85

    bare = _peak("1.20")
    assert set(bare) == {"delta", "centroid"}
    assert bare["delta"] == 1.2


def test_r1_bad_tail_is_an_error_and_not_a_category():
    with pytest.raises(TaskInputError, match="leftover"):
        parse_h_peak_list("1.20 (s, J=7.1, 12H, leftover)")
    with pytest.raises(TaskInputError, match="12H extra"):
        parse_h_peak_list("1.20 (s, J=7.1, 12H extra)")


def _written_peak(source: dict) -> dict | None:
    delta = source.get("centroid") if source.get("centroid") is not None else source.get("delta")
    if delta is None:
        return None
    category = source.get("category") or ""
    coupling = source.get("j_values") or ""
    expected = {"nH": source.get("nH", 1), "delta": float(f"{float(delta):.2f}")}
    if category:
        expected["category"] = category
    if coupling and coupling != "_":
        expected["j_values"] = coupling.replace("_", ", ").strip(", ")
    return expected


def _v64_nh_mismatches(path: Path) -> list[dict]:
    format_alpaca = _formatter()
    rows = json.loads(path.read_text(encoding="utf-8"))
    mismatches = []
    for index, row in enumerate(rows, 1):
        human = next(turn["value"] for turn in row["conversations"] if turn.get("from") == "human")
        source = json.loads(human)
        payload = dict(source)
        payload["label"] = 0
        text = format_alpaca(payload)["input"]
        body = next(line.split(":", 1)[1].strip() for line in text.splitlines() if line.startswith("1H NMR"))
        try:
            parsed = parse_h_peak_list(body)
        except TaskInputError as exc:
            mismatches.append({"row": index, "error": str(exc)})
            continue
        expected = [item for item in (_written_peak(peak) for peak in source["h_nmr_peaks"]) if item]
        if len(parsed) != len(expected):
            mismatches.append({"row": index, "parsed_peaks": len(parsed), "written_peaks": len(expected)})
            continue
        for peak_index, (got, want) in enumerate(zip(parsed, expected), 1):
            for key in ("nH", "category", "j_values"):
                if got.get(key) != want.get(key):
                    mismatches.append({
                        "row": index,
                        "peak": peak_index,
                        "field": key,
                        "written": want.get(key),
                        "parsed": got.get(key),
                    })
                    break
    return mismatches


def test_r1_registered_v64_text_has_no_nh_misread():
    train = _v64_nh_mismatches(V64_TRAIN)
    val = _v64_nh_mismatches(V64_VAL)
    assert train == []
    assert val == []


def _router_config() -> dict:
    return {
        "sim_threshold_both": 0.85,
        "sim_threshold_strict": 0.92,
        "h_tolerance": 2,
        "template_languages": ["en"],
        "template_variant_seed": 0,
    }


def test_r2_entries_reject_typed_bad_input_before_a_yes_no(monkeypatch):
    nested = {
        "idx": "nested-1",
        "candidate_smiles": {"gt_smiles": "CCO", "label": 1},
        "c_nmr_peaks": [{"delta (ppm)": 57.8}],
        "h_nmr_peaks": [{
            "delta": 1.2,
            "centroid": 1.2,
            "nH": 3,
            "category": {"gt_smiles": "CCO", "label": 1},
        }],
    }
    with pytest.raises(TaskInputError, match="nested-1") as messages_error:
        infer.make_messages(nested)
    assert "candidate_smiles" in str(messages_error.value)
    assert "dict" in str(messages_error.value)

    category_only = {
        "idx": "category-1",
        "candidate_smiles": "CCO",
        "c_nmr_peaks": [{"delta (ppm)": 57.8}],
        "h_nmr_peaks": [{
            "delta": 1.2,
            "centroid": 1.2,
            "nH": 3,
            "category": {"gt_smiles": "CCO", "label": 1},
        }],
    }
    with pytest.raises(TaskInputError, match="category-1") as category_error:
        infer.make_messages(category_only)
    assert "h_nmr_peaks[0].category" in str(category_error.value)

    with pytest.raises(ValueError, match="not-smiles-1") as prepared_error:
        infer.prepare_eval_record({
            "idx": "not-smiles-1",
            "candidate_smiles": "not_smiles",
            "c_nmr_peaks": [{"delta (ppm)": 57.8}, {}, {}],
            "h_nmr_peaks": [{"delta": 1.2, "centroid": 1.2, "nH": 3}],
        }, 7)
    assert "not_smiles" in str(prepared_error.value)

    with pytest.raises(ValueError, match="empty-peak-2") as empty_error:
        infer.prepare_eval_record({
            "idx": "empty-peak-2",
            "candidate_smiles": "CCO",
            "c_nmr_peaks": [{"delta (ppm)": 57.8}, {}, {}],
        }, 8)
    assert "c_nmr_peaks[1]" in str(empty_error.value)

    called = {"chat": 0}

    async def forbidden_chat(*_args, **_kwargs):
        called["chat"] += 1
        raise AssertionError("vLLM stub was called for a rejected record")

    monkeypatch.setattr(call_v5_vllm, "chat_turn", forbidden_chat)
    with pytest.raises(TaskInputError, match="vllm-1") as vllm_error:
        asyncio.run(call_v5_vllm.run_case(
            "http://127.0.0.1:9/v1",
            "stub",
            {
                "idx": "vllm-1",
                "candidate_smiles": "CCO",
                "c_nmr_peaks": [{"delta (ppm)": 57.8}, {}, {}],
                "h_nmr_peaks": [],
            },
            {},
        ))
    assert "c_nmr_peaks[1]" in str(vllm_error.value)
    assert called["chat"] == 0

    record, error = _build_record_from_user_json({
        "idx": "tool-1",
        "candidate_smiles": "CCO",
        "c_nmr_peaks": [{"delta (ppm)": 57.8}],
        "h_nmr_peaks": [{"category": {"gt_smiles": "CCO", "label": 1}, "delta": 1.2}],
    })
    assert record == {}
    assert error is not None
    assert "tool-1" in error
    assert "h_nmr_peaks[0].category" in error
    assert "final_answer" not in error

    with pytest.raises(TaskInputError, match="router-1") as router_error:
        RuleBasedRouter(
            {
                "idx": "router-1",
                "candidate_smiles": "CCO",
                "c_nmr_peaks": [{"delta (ppm)": 57.8}, {}],
            },
            prediction_cache={},
            config=_router_config(),
            allow_remote_predict=False,
        )
    assert "c_nmr_peaks[1]" in str(router_error.value)

    monkeypatch.setenv("NMR_MATCHING_SKIP_STARTUP_LOAD", "1")
    monkeypatch.setattr(api_service.state, "infer", lambda _payload: (_ for _ in ()).throw(
        AssertionError("API infer ran after a bad request")
    ))
    with TestClient(api_service.app) as client:
        response = client.post("/match", json={
            "candidate_smiles": "CCO",
            "c_nmr_peaks": [{"delta (ppm)": 57.8}, {}, {}],
        })
    assert response.status_code == 422
    assert "c_nmr_peaks[1]" in response.text
    assert "final_answer" not in response.json()

    with TestClient(api_service.app) as client:
        smiles = client.post("/match", json={
            "candidate_smiles": "not_smiles",
            "c_nmr_peaks": [{"delta (ppm)": 57.8}],
        })
    assert smiles.status_code == 422
    assert "not_smiles" in smiles.text
    assert "final_answer" not in smiles.json()


def test_r2_keeps_legal_physical_values_and_converts_scalar_peaks():
    legal = {
        "idx": "legal-1",
        "candidate_smiles": "CCO",
        "c_nmr_peaks": [{"delta (ppm)": -1.5}, {"delta (ppm)": 57.8}, {"delta (ppm)": 57.8}],
        "h_nmr_peaks": [
            {"delta": -0.2, "centroid": -0.2, "nH": 0, "category": "No", "j_values": "7.1, 2.0"},
            {"delta": 1.2, "centroid": 1.2, "nH": None, "category": "s"},
            {"rangeMin": 7.1, "rangeMax": 7.4, "category": "m"},
        ],
        "nmr_solvent": "CDCl3",
        "nmr_frequency": "400 MHz",
        "experimental_total_nH": 0,
    }
    messages = infer.make_messages(legal)
    user = json.loads(messages[1]["content"])
    assert user["c_nmr_peaks"] == [
        {"delta (ppm)": -1.5},
        {"delta (ppm)": 57.8},
        {"delta (ppm)": 57.8},
    ]
    assert user["h_nmr_peaks"][0]["nH"] == 0
    assert user["h_nmr_peaks"][0]["category"] == "No"
    assert user["h_nmr_peaks"][0]["j_values"] == "7.1, 2.0"
    assert user["h_nmr_peaks"][1]["nH"] is None
    assert "nH" not in user["h_nmr_peaks"][2]
    assert user["h_nmr_peaks"][2]["rangeMin"] == 7.1
    assert "delta" not in user["h_nmr_peaks"][2]
    assert user["experimental_total_nH"] == 0
    assert set(user) <= set(TASK_FIELDS)

    runtime = NmrMatchingRuntime({
        "candidate_smiles": "CCO",
        "c_nmr_peaks": [57.8, 18.2],
        "h_nmr_peaks": [1.18],
    })
    assert runtime.c_peaks == [{"delta (ppm)": 57.8}, {"delta (ppm)": 18.2}]
    assert runtime.h_peaks == [{"delta": 1.18, "centroid": 1.18, "nH": None}]
    observed = json.loads(asyncio.run(runtime.run_tool_call_async(
        {"function": {"name": "count_atoms", "arguments": {"element": "C"}}}
    ))["content"])
    assert observed["experimental_distinct_signals"] == 2
    assert observed["candidate_count"] == 2
    assert observed["match"] is True

    with pytest.raises(TaskInputError, match="experimental_total_nH"):
        NmrMatchingRuntime({
            "idx": "negative-total",
            "candidate_smiles": "CCO",
            "c_nmr_peaks": [{"delta (ppm)": 57.8}],
            "experimental_total_nH": -3,
        })
    with pytest.raises(TaskInputError, match="h_nmr_peaks\\[0\\].nH"):
        NmrMatchingRuntime({
            "candidate_smiles": "CCO",
            "h_nmr_peaks": [{"delta": 1.2, "nH": -1}],
        })
    described = describe_integrals([{"delta": 1.2, "nH": -4}], -5)
    assert described["experimental_total_nH"] is None
    assert described["explicit_invalid"] is True
    assert described["invalid_nH_count"] == 1
    assert described["completeness"] == "invalid"

    with pytest.raises(TaskInputError, match="h_nmr_peaks\\[0\\]"):
        infer.make_messages({
            "candidate_smiles": "CCO",
            "h_nmr_peaks": [{"delta": None, "rangeMin": 1.0}],
        })
    with pytest.raises(TaskInputError, match="nmr_solvent"):
        infer.make_messages({
            "candidate_smiles": "CCO",
            "c_nmr_peaks": [{"delta (ppm)": 57.8}],
            "nmr_solvent": {"name": "CDCl3"},
        })
    with pytest.raises(TaskInputError, match="must be a list"):
        asyncio.run(call_v5_vllm.run_case(
            "http://127.0.0.1:9/v1",
            "stub",
            {"idx": "scalar-rejected-as-dict", "candidate_smiles": "CCO", "c_nmr_peaks": {"delta (ppm)": 57.8}},
            {},
        ))


def _sidecar(value):
    if isinstance(value, str):
        return json.loads(value)
    return value


def _frequencies(sidecar: dict) -> list[str]:
    return [header["frequency_raw"] for header in sidecar["headers"]]


def test_r4_parquet_and_alpaca_roundtrip_keep_sidecar_out_of_the_model_task(tmp_path: Path):
    alpaca = {
        "idx": "freq-1",
        "_harness_pair_id": 41,
        "_harness_run_id": "run-r4",
        "_label": 0,
        "_gt_smiles": "CCC",
        "_candidate_smiles": "CCO",
        "input": CONFLICT_TEXT,
    }
    alpaca_row = build_agent_parquet._row_from_alpaca(3, alpaca)
    alpaca_side = _sidecar(alpaca_row["extra_info"]["input_sidecar"])
    assert _frequencies(alpaca_side) == ["100 MHz", "400 MHz"]
    assert any(item["field"] == "nmr_frequency" for item in alpaca_side["conflicts"])
    assert alpaca_row["extra_info"]["pair_id"] == 41
    assert alpaca_row["extra_info"]["run_id"] == "run-r4"
    user = json.loads(alpaca_row["prompt"][1]["content"])
    assert user["nmr_frequency"] is None
    assert user["nmr_solvent"] == "CDCl3"
    assert set(user) <= set(TASK_FIELDS)
    assert "100 MHz" not in alpaca_row["prompt"][1]["content"]
    assert "400 MHz" not in alpaca_row["prompt"][1]["content"]
    assert "CCC" not in alpaca_row["prompt"][1]["content"]
    assert alpaca_row["reward_model"]["ground_truth"]["gt_smiles"] == "CCC"

    source = pd.DataFrame([{
        "prompt": [{"role": "user", "content": CONFLICT_TEXT}],
        "extra_info": {"index": 5},
        "reward_model": {"ground_truth": {"candidate_smiles": "NOT-THE-TASK", "gt_smiles": "CCC"}},
        "ability": "NMR",
    }])
    source_path = tmp_path / "source.parquet"
    source.to_parquet(source_path)
    loaded = pd.read_parquet(source_path).iloc[0]
    parquet_row = build_agent_parquet._row_from_parquet_row(loaded)
    parquet_side = _sidecar(parquet_row["extra_info"]["input_sidecar"])
    assert _frequencies(parquet_side) == ["100 MHz", "400 MHz"]
    assert parquet_side["conflicts"]
    parquet_user = json.loads(parquet_row["prompt"][1]["content"])
    assert set(parquet_user) <= set(TASK_FIELDS)
    assert "input_sidecar" not in parquet_row["prompt"][1]["content"]
    assert "gt_smiles" not in parquet_row["prompt"][1]["content"]

    out = pd.DataFrame([alpaca_row, parquet_row])
    out_path = tmp_path / "agent.parquet"
    out.to_parquet(out_path)
    read_back = pd.read_parquet(out_path)
    for row in read_back.to_dict(orient="records"):
        side = _sidecar(row["extra_info"]["input_sidecar"])
        assert _frequencies(side) == ["100 MHz", "400 MHz"]
        assert any(item["code"] == "header_values_differ" for item in side["conflicts"])
        prompt = row["prompt"]
        content = prompt[1]["content"]
        assert "100 MHz" not in content
        assert "400 MHz" not in content
        assert set(json.loads(content)) <= set(TASK_FIELDS)

    json_path = tmp_path / "alpaca_copy.json"
    json_path.write_text(json.dumps([alpaca]), encoding="utf-8")
    copied = json.loads(json_path.read_text(encoding="utf-8"))[0]
    copied_row = build_agent_parquet._row_from_alpaca(0, copied)
    copied_side = _sidecar(copied_row["extra_info"]["input_sidecar"])
    assert _frequencies(copied_side) == ["100 MHz", "400 MHz"]


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({}, "missing"),
        ({"c_nmr_peaks": None}, "null"),
        ({"c_nmr_peaks": []}, "empty_list"),
    ],
)
def test_r4_json_presence_distinguishes_missing_null_and_empty(payload: dict, expected: str):
    task = {
        "candidate_smiles": "CCO",
        "h_nmr_peaks": [{"delta": 1.2, "centroid": 1.2, "nH": 3, "category": "s"}],
    }
    task.update(payload)
    prepared = infer.prepare_eval_record({"idx": f"presence-{expected}", "input": json.dumps(task)}, 2)
    presence = prepared["input_sidecar"]["field_presence"]
    assert presence["json_string"]["c_nmr_peaks"] == expected
    assert presence["outer_structured"]["c_nmr_peaks"] == "missing"
    assert presence["alpaca_text"] is None
    user = json.loads(infer.make_messages(prepared["inference_input"])[1]["content"])
    assert "field_presence" not in user
    assert set(user) <= set(TASK_FIELDS)
    assert user["candidate_smiles"] == "CCO"
    assert user["h_nmr_peaks"][0]["nH"] == 3
