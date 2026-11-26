"""CLI and entry regressions for the second 02B-1 repair.

Identity expectations are the ids written into the temporary parquet. Peak
expectations are the field name and the sample id, not a comparison against
the same normalizer.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
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
from recipe.nmr_matching_agent.task_input import TASK_FIELDS, TaskInputError
from verl.tools.nmr_matching_tools import _build_record_from_user_json

REPO = Path(__file__).resolve().parents[2]
PYTHON = sys.executable
TEXT = (
    "Candidate SMILES: CCO\n"
    "13C NMR (100 MHz, CDCl3): 57.8\n"
    "1H NMR (400 MHz, CDCl3): 1.2 (t, J=7.1, 3H)\n"
)
FALSY = [{}, False, 0, ""]
CONTAINERS = [
    ("c_nmr_peaks", "experimental_13c_peaks"),
    ("h_nmr_peaks", "experimental_1h_peaks"),
]


def _objects(value):
    if isinstance(value, str):
        stripped = value.lstrip()
        if stripped[:1] in "{[":
            try:
                return _objects(json.loads(value))
            except json.JSONDecodeError:
                return value
        return value
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes, dict)):
        return _objects(value.tolist())
    if isinstance(value, dict):
        return {str(key): _objects(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_objects(item) for item in value]
    if type(value).__module__ == "numpy" and type(value).__name__.startswith("int"):
        return int(value)
    return value


def _rows(path: Path) -> list[dict]:
    frame = pd.read_parquet(path)
    rows = []
    for raw in frame.to_dict(orient="records"):
        raw["extra_info"] = _objects(raw["extra_info"])
        raw["reward_model"] = _objects(raw["reward_model"])
        prompt = raw["prompt"]
        if isinstance(prompt, str):
            prompt = json.loads(prompt)
        raw["prompt"] = prompt
        rows.append(raw)
    return rows


def _cli(source: Path, dest: Path) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [
            PYTHON, "-B",
            "recipe/nmr_matching_agent/build_agent_parquet.py",
            "--from_parquet", str(source),
            "--out", str(dest),
        ],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _source_row(pair_id, run_id, label, *, gt="CCC"):
    return {
        "prompt": [{"role": "user", "content": TEXT}],
        "extra_info": {"pair_id": pair_id, "run_id": run_id},
        "reward_model": {"ground_truth": {"label": label, "gt_smiles": gt}},
        "ability": "NMR",
    }


def _frequencies(sidecar: dict) -> list[str]:
    headers = sidecar.get("headers")
    if headers is None and isinstance(sidecar.get("source"), dict):
        headers = sidecar["source"].get("headers")
    if headers is None:
        return []
    return [header["frequency_raw"] for header in list(headers)]


def _conflicts(sidecar: dict) -> list:
    if isinstance(sidecar.get("source"), dict) and "conflicts" in sidecar["source"]:
        return sidecar["source"]["conflicts"]
    return sidecar.get("conflicts") or []


def test_f1_cli_keeps_identity_labels_and_source_sidecar(tmp_path: Path):
    source = tmp_path / "source.parquet"
    converted = tmp_path / "converted.parquet"
    shuffled = tmp_path / "shuffled.parquet"
    reconverted = tmp_path / "reconverted.parquet"
    pd.DataFrame([
        _source_row(41, "review-identity-run", 1),
        _source_row(42, "review-identity-run", 0),
    ]).to_parquet(source)
    first = _cli(source, converted)
    assert first.returncode == 0, first.stderr
    rows = _rows(converted)
    assert [(row["extra_info"]["pair_id"], row["extra_info"]["run_id"]) for row in rows] == [
        (41, "review-identity-run"),
        (42, "review-identity-run"),
    ]
    assert [row["extra_info"]["identity_status"] for row in rows] == ["verified", "verified"]
    assert [int(row["reward_model"]["ground_truth"]["label"]) for row in rows] == [1, 0]
    for row in rows:
        user = json.loads(row["prompt"][1]["content"])
        assert set(user) <= set(TASK_FIELDS)
        content = row["prompt"][1]["content"]
        assert "review-identity-run" not in content
        assert "pair_id" not in content
        assert "gt_smiles" not in content
        assert "CCC" not in content
        assert _frequencies(row["extra_info"]["input_sidecar"]) == ["100 MHz", "400 MHz"]
        assert any(item["field"] == "nmr_frequency" for item in _conflicts(row["extra_info"]["input_sidecar"]))

    backwards = list(reversed(_rows(converted)))
    pd.DataFrame(backwards).to_parquet(shuffled)
    second = _cli(shuffled, reconverted)
    assert second.returncode == 0, second.stderr
    again = _rows(reconverted)
    by_id = {int(row["extra_info"]["pair_id"]): row for row in again}
    assert set(by_id) == {41, 42}
    assert int(by_id[41]["reward_model"]["ground_truth"]["label"]) == 1
    assert int(by_id[42]["reward_model"]["ground_truth"]["label"]) == 0
    assert by_id[41]["extra_info"]["run_id"] == "review-identity-run"
    for row in again:
        sidecar = row["extra_info"]["input_sidecar"]
        assert _frequencies(sidecar) == ["100 MHz", "400 MHz"]
        assert any(item["code"] == "header_values_differ" for item in _conflicts(sidecar))
        assert "conversion" in sidecar
        assert sidecar["source"]["field_presence"]
        user = json.loads(row["prompt"][1]["content"])
        assert set(user) <= set(TASK_FIELDS)
        assert "100 MHz" not in row["prompt"][1]["content"]
        assert "400 MHz" not in row["prompt"][1]["content"]


def test_f1_old_parquet_does_not_invent_a_global_zero(tmp_path: Path):
    source = tmp_path / "old.parquet"
    dest = tmp_path / "old_agent.parquet"
    pd.DataFrame([
        {
            "prompt": [{"role": "user", "content": TEXT}],
            "extra_info": {"index": 0, "kept": "first"},
            "reward_model": {"ground_truth": {"label": 1}},
            "ability": "NMR",
        },
        {
            "prompt": [{"role": "user", "content": TEXT}],
            "extra_info": {"index": 7, "kept": "second"},
            "reward_model": {"ground_truth": {"label": 0}},
            "ability": "NMR",
        },
    ]).to_parquet(source)
    result = _cli(source, dest)
    assert result.returncode == 0, result.stderr
    rows = _rows(dest)
    assert [int(row["extra_info"]["index"]) for row in rows] == [0, 7]
    assert [row["extra_info"]["kept"] for row in rows] == ["first", "second"]
    assert [row["extra_info"]["identity_status"] for row in rows] == ["unverified", "unverified"]
    assert all("pair_id" not in row["extra_info"] for row in rows)
    assert all("run_id" not in row["extra_info"] for row in rows)
    assert [int(row["reward_model"]["ground_truth"]["label"]) for row in rows] == [1, 0]


def test_f1_both_entries_keep_harness_identity_and_reject_conflicts(tmp_path: Path):
    alpaca = {
        "input": TEXT,
        "_harness_pair_id": 41,
        "_harness_run_id": "run-41",
        "_label": 0,
        "_gt_smiles": "CCC",
        "_candidate_smiles": "CCO",
    }
    alpaca_row = build_agent_parquet._row_from_alpaca(3, alpaca)
    assert alpaca_row["extra_info"]["pair_id"] == 41
    assert alpaca_row["extra_info"]["run_id"] == "run-41"
    assert alpaca_row["extra_info"]["index"] == 3
    source = tmp_path / "alpaca_agent.parquet"
    dest = tmp_path / "from_alpaca_agent.parquet"
    pd.DataFrame([alpaca_row]).to_parquet(source)
    result = _cli(source, dest)
    assert result.returncode == 0, result.stderr
    row = _rows(dest)[0]
    assert int(row["extra_info"]["pair_id"]) == 41
    assert row["extra_info"]["run_id"] == "run-41"
    assert int(row["extra_info"]["index"]) == 3
    assert row["extra_info"]["identity_status"] == "verified"
    assert _frequencies(row["extra_info"]["input_sidecar"]) == ["100 MHz", "400 MHz"]
    user = json.loads(row["prompt"][1]["content"])
    assert "pair_id" not in user
    assert set(user) <= set(TASK_FIELDS)

    conflict = tmp_path / "conflict.parquet"
    rejected = tmp_path / "conflict_out.parquet"
    pd.DataFrame([{
        "prompt": [{"role": "user", "content": TEXT}],
        "extra_info": {"pair_id": 41, "_harness_pair_id": 42, "run_id": "review-identity-run"},
        "reward_model": {"ground_truth": {"label": 1}},
        "ability": "NMR",
    }]).to_parquet(conflict)
    failed = _cli(conflict, rejected)
    assert failed.returncode != 0
    assert "conflicting harness pair ids" in failed.stderr
    assert not rejected.exists()

    illegal = tmp_path / "illegal.parquet"
    pd.DataFrame([{
        "prompt": [{"role": "user", "content": TEXT}],
        "extra_info": {"pair_id": True, "run_id": ""},
        "reward_model": {"ground_truth": {"label": 1}},
        "ability": "NMR",
    }]).to_parquet(illegal)
    illegal_out = tmp_path / "illegal_out.parquet"
    illegal_run = _cli(illegal, illegal_out)
    assert illegal_run.returncode != 0
    assert "invalid pair_id" in illegal_run.stderr
    assert not illegal_out.exists()

    negative = tmp_path / "negative.parquet"
    negative_out = tmp_path / "negative_out.parquet"
    pd.DataFrame([{
        "prompt": [{"role": "user", "content": TEXT}],
        "extra_info": {"pair_id": -1, "run_id": "review-identity-run"},
        "reward_model": {"ground_truth": {"label": 1}},
        "ability": "NMR",
    }]).to_parquet(negative)
    negative_run = _cli(negative, negative_out)
    assert negative_run.returncode != 0
    assert "invalid pair_id" in negative_run.stderr
    assert not negative_out.exists()

    blank_run = tmp_path / "blank_run.parquet"
    blank_out = tmp_path / "blank_out.parquet"
    pd.DataFrame([{
        "prompt": [{"role": "user", "content": TEXT}],
        "extra_info": {"pair_id": 41, "run_id": ""},
        "reward_model": {"ground_truth": {"label": 1}},
        "ability": "NMR",
    }]).to_parquet(blank_run)
    blank = _cli(blank_run, blank_out)
    assert blank.returncode != 0
    assert "invalid run_id" in blank.stderr
    assert not blank_out.exists()

    partial = tmp_path / "partial.parquet"
    partial_out = tmp_path / "partial_out.parquet"
    explicit_zero = tmp_path / "explicit_zero.parquet"
    explicit_zero_out = tmp_path / "explicit_zero_out.parquet"
    pd.DataFrame([
        {
            "prompt": [{"role": "user", "content": TEXT}],
            "extra_info": {"pair_id": 41, "index": 0},
            "reward_model": {"ground_truth": {"label": 1}},
            "ability": "NMR",
        },
    ]).to_parquet(partial)
    partial_run = _cli(partial, partial_out)
    assert partial_run.returncode == 0, partial_run.stderr
    partial_row = _rows(partial_out)[0]
    assert int(partial_row["extra_info"]["pair_id"]) == 41
    assert "run_id" not in partial_row["extra_info"]
    assert partial_row["extra_info"]["identity_status"] == "incomplete"
    assert int(partial_row["extra_info"]["index"]) == 0
    pd.DataFrame([{
        "prompt": [{"role": "user", "content": TEXT}],
        "extra_info": {"pair_id": 0, "run_id": "review-identity-run"},
        "reward_model": {"ground_truth": {"label": 0}},
        "ability": "NMR",
    }]).to_parquet(explicit_zero)
    zero_run = _cli(explicit_zero, explicit_zero_out)
    assert zero_run.returncode == 0, zero_run.stderr
    zero_row = _rows(explicit_zero_out)[0]
    assert int(zero_row["extra_info"]["pair_id"]) == 0
    assert zero_row["extra_info"]["run_id"] == "review-identity-run"
    assert zero_row["extra_info"]["identity_status"] == "verified"
    assert int(zero_row["reward_model"]["ground_truth"]["label"]) == 0
    with pytest.raises(ValueError, match="invalid run_id"):
        build_agent_parquet._row_from_parquet_row({
            "prompt": [{"role": "user", "content": TEXT}],
            "extra_info": {"pair_id": 41, "run_id": None},
            "reward_model": {"ground_truth": {"label": 1}},
            "ability": "NMR",
        })
    with pytest.raises(ValueError, match="conflicting harness run ids"):
        build_agent_parquet._row_from_parquet_row({
            "prompt": [{"role": "user", "content": TEXT}],
            "extra_info": {
                "pair_id": 41,
                "run_id": "review-identity-run",
                "_harness_run_id": "other-run",
            },
            "reward_model": {"ground_truth": {"label": 1}},
            "ability": "NMR",
        })


def _router_config():
    return {
        "sim_threshold_both": 0.85,
        "sim_threshold_strict": 0.92,
        "h_tolerance": 2,
        "template_languages": ["en"],
        "template_variant_seed": 0,
    }


def _bad_record(field: str, value):
    record = {
        "idx": "bad-container",
        "candidate_smiles": "CCO",
        "c_nmr_peaks": [{"delta (ppm)": 57.8}],
        "h_nmr_peaks": [{"delta": 1.2, "nH": 3}],
    }
    record[field] = value
    return record


@pytest.mark.parametrize("field", ["c_nmr_peaks", "h_nmr_peaks"])
@pytest.mark.parametrize("value", FALSY)
def test_f2_primary_falsy_container_is_rejected_before_a_decision(field, value):
    record = _bad_record(field, value)
    with pytest.raises(TaskInputError, match="bad-container") as messages:
        infer.make_messages(record)
    assert field in str(messages.value)
    with pytest.raises(TaskInputError, match="bad-container") as prepared:
        infer.prepare_eval_record(record, 4)
    assert field in str(prepared.value)
    with pytest.raises(TaskInputError, match="bad-container") as runtime:
        NmrMatchingRuntime(record)
    assert field in str(runtime.value)
    with pytest.raises(TaskInputError, match="bad-container") as router:
        RuleBasedRouter(
            record,
            prediction_cache={},
            config=_router_config(),
            allow_remote_predict=False,
        )
    assert field in str(router.value)
    tool_record, error = _build_record_from_user_json(record)
    assert tool_record == {}
    assert error is not None
    assert "bad-container" in error
    assert field in error
    assert "final_answer" not in error


@pytest.mark.parametrize(("primary", "alias"), CONTAINERS)
@pytest.mark.parametrize("value", FALSY)
def test_f2_alias_falsy_container_is_rejected(primary, alias, value):
    record = {
        "idx": "bad-container",
        "candidate_smiles": "CCO",
        "c_nmr_peaks": [{"delta (ppm)": 57.8}],
        "h_nmr_peaks": [{"delta": 1.2, "nH": 3}],
    }
    record[primary] = []
    record[alias] = value
    with pytest.raises(TaskInputError, match="bad-container") as caught:
        infer.make_messages(record)
    assert alias in str(caught.value)
    with pytest.raises(TaskInputError, match=alias):
        infer.prepare_eval_record(record, 1)
    with pytest.raises(TaskInputError, match=alias):
        NmrMatchingRuntime(record)


def test_f2_legal_absence_null_empty_and_scalar_lists_stay_distinct():
    missing = {
        "idx": "legal-missing",
        "candidate_smiles": "CCO",
        "h_nmr_peaks": [{"delta": 1.2, "centroid": 1.2, "nH": None}],
    }
    null = {**missing, "c_nmr_peaks": None, "idx": "legal-null"}
    empty = {**missing, "c_nmr_peaks": [], "idx": "legal-empty"}
    for record, expected in ((missing, "missing"), (null, "null"), (empty, "empty_list")):
        prepared = infer.prepare_eval_record(record, 2)
        assert prepared["input_sidecar"]["field_presence"]["outer_structured"]["c_nmr_peaks"] == expected
        user = json.loads(infer.make_messages(record)[1]["content"])
        assert user["c_nmr_peaks"] == []
        assert user["h_nmr_peaks"][0]["nH"] is None
        assert set(user) <= set(TASK_FIELDS)

    scalar = {
        "idx": "legal-scalar",
        "candidate_smiles": "CCO",
        "c_nmr_peaks": [-1.5, 57.8, 57.8],
        "h_nmr_peaks": [1.2],
        "experimental_total_nH": 0,
    }
    user = json.loads(infer.make_messages(scalar)[1]["content"])
    assert user["c_nmr_peaks"] == [
        {"delta (ppm)": -1.5},
        {"delta (ppm)": 57.8},
        {"delta (ppm)": 57.8},
    ]
    assert user["h_nmr_peaks"] == [{"delta": 1.2, "centroid": 1.2, "nH": None}]
    assert user["experimental_total_nH"] == 0
    alias = {
        "idx": "legal-alias",
        "candidate_smiles": "CCO",
        "c_nmr_peaks": [],
        "experimental_13c_peaks": [57.8],
        "h_nmr_peaks": [{"delta": 0.0, "nH": 0, "category": "s"}],
    }
    aliased = json.loads(infer.make_messages(alias)[1]["content"])
    assert aliased["c_nmr_peaks"] == [{"delta (ppm)": 57.8}]
    assert aliased["h_nmr_peaks"][0]["nH"] == 0


@pytest.mark.parametrize("field", ["c_nmr_peaks", "h_nmr_peaks"])
@pytest.mark.parametrize("value", FALSY)
def test_f2_api_and_vllm_reject_the_same_containers(field, value, monkeypatch):
    called = {"chat": 0, "infer": 0}

    async def forbidden_chat(*_args, **_kwargs):
        called["chat"] += 1
        raise AssertionError("vLLM was called")

    monkeypatch.setattr(call_v5_vllm, "chat_turn", forbidden_chat)
    record = _bad_record(field, value)
    with pytest.raises(TaskInputError, match="bad-container") as vllm_error:
        asyncio.run(call_v5_vllm.run_case("http://127.0.0.1:9/v1", "stub", record, {}))
    assert field in str(vllm_error.value)
    assert called["chat"] == 0

    monkeypatch.setenv("NMR_MATCHING_SKIP_STARTUP_LOAD", "1")

    def forbidden_infer(_payload):
        called["infer"] += 1
        raise AssertionError("API infer was called")

    monkeypatch.setattr(api_service.state, "infer", forbidden_infer)
    payload = {
        "candidate_smiles": "CCO",
        "c_nmr_peaks": [{"delta (ppm)": 57.8}],
        "h_nmr_peaks": [{"delta": 1.2, "nH": 3}],
    }
    payload[field] = value
    with TestClient(api_service.app) as client:
        response = client.post("/match", json=payload)
    assert response.status_code == 422
    assert field in response.text
    assert "final_answer" not in response.text
    assert called["infer"] == 0
