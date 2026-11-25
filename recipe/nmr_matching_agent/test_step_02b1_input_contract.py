"""Entry-path tests for the step-02B-1 physical task contract.

Each test calls a real reader, CLI builder, API stub, vLLM stub, router, runtime,
RL tool, or parquet converter. Helpers are not treated as the assertion boundary.
"""

from __future__ import annotations

import asyncio
import copy
import json
from types import SimpleNamespace

import pytest

from recipe.nmr_matching_agent import api_service
from recipe.nmr_matching_agent import build_agent_parquet
from recipe.nmr_matching_agent import call_v5_vllm
from recipe.nmr_matching_agent import standalone_infer_matching as infer
from recipe.nmr_matching_agent.nmr_matching_runtime import NmrMatchingRuntime
from recipe.nmr_matching_agent.rule_based_router import RuleBasedRouter
from recipe.nmr_matching_agent.task_input import (
    normalize_task_input,
    parse_task_text,
    task_to_lossless_text,
)
from verl.tools.nmr_matching_tools import _build_record_from_user_json

PHYSICAL = {
    "candidate_smiles": "CCO",
    "c_nmr_peaks": [
        {"delta (ppm)": 57.8123},
        {"delta (ppm)": 18.2},
        {"delta (ppm)": 57.8123},
    ],
    "h_nmr_peaks": [
        {
            "category": "t",
            "centroid": 1.184,
            "delta": 1.184,
            "j_values": "7.1",
            "nH": 3,
            "rangeMax": 1.2,
            "rangeMin": 1.16,
        },
        {
            "category": "q",
            "centroid": 3.7,
            "delta": 3.7,
            "j_values": "7.1, 2.0",
            "nH": 0,
        },
    ],
    "nmr_solvent": "CDCl3",
    "nmr_frequency": "400 MHz",
    "nmr_frequency_nucleus": "unknown",
    "experimental_total_nH": 6,
}

OLD_TEXT = (
    "Candidate SMILES: CCO\n"
    "13C NMR (400 MHz, CDCl3): 57.8123, 18.2, 57.8123\n"
    "1H NMR (400 MHz, CDCl3): 1.16-1.20 (t, J=7.1, 3H), 3.7 (q, J=7.1, 2.0, 0H)\n"
)

FORBIDDEN = ("_label", "label", "gt_smiles", "_gt_smiles", "idx", "case_id", "negative_type")


def _no_gold(payload) -> None:
    encoded = json.dumps(payload, ensure_ascii=False)
    for key in FORBIDDEN:
        assert f'"{key}"' not in encoded


def _router_config():
    return {
        "sim_threshold_both": 0.85,
        "sim_threshold_strict": 0.92,
        "h_tolerance": 2,
        "template_variant_seed": 0,
        "template_languages": ["en"],
    }


def test_lossless_json_roundtrip_keeps_precision_order_and_zero_integral():
    original = copy.deepcopy(PHYSICAL)
    text = task_to_lossless_text(original)
    assert original == PHYSICAL
    parsed, diagnostics = parse_task_text(text)
    assert diagnostics["format"] == "lossless_json"
    assert parsed == normalize_task_input(PHYSICAL)
    assert parsed["c_nmr_peaks"][0]["delta (ppm)"] == 57.8123
    assert parsed["c_nmr_peaks"][2]["delta (ppm)"] == 57.8123
    assert parsed["h_nmr_peaks"][0]["j_values"] == "7.1"
    assert parsed["h_nmr_peaks"][0]["rangeMin"] == 1.16
    assert parsed["h_nmr_peaks"][1]["nH"] == 0
    assert parsed["nmr_frequency_nucleus"] == "unknown"
    assert parsed["experimental_total_nH"] == 6
    again = normalize_task_input(parsed)
    assert again == parsed


def test_old_text_keeps_explicit_information_and_does_not_restore_a_rounded_source():
    task, diagnostics = parse_task_text(OLD_TEXT)
    assert diagnostics["format"] == "alpaca_text"
    assert diagnostics["nucleus_assignment"] == "unknown"
    assert any(item["code"] == "header_value_copied" for item in diagnostics["warnings"])
    assert task["c_nmr_peaks"] == [
        {"delta (ppm)": 57.8123},
        {"delta (ppm)": 18.2},
        {"delta (ppm)": 57.8123},
    ]
    assert task["h_nmr_peaks"][0]["rangeMin"] == 1.16
    assert task["h_nmr_peaks"][0]["rangeMax"] == 1.20
    assert "delta" not in task["h_nmr_peaks"][0]
    assert "centroid" not in task["h_nmr_peaks"][0]
    assert task["h_nmr_peaks"][0]["j_values"] == "7.1"
    assert task["h_nmr_peaks"][0]["nH"] == 3
    assert task["h_nmr_peaks"][1]["nH"] == 0
    assert task["h_nmr_peaks"][1]["j_values"] == "7.1, 2.0"
    assert task["nmr_solvent"] == "CDCl3"
    assert task["nmr_frequency"] == "400 MHz"
    rounded = (
        "Candidate SMILES: CCO\n"
        "13C NMR (400 MHz, CDCl3): 57.8\n"
        "1H NMR (400 MHz, CDCl3): 1.18 (t, 3H)\n"
    )
    rounded_task, _diagnostics = parse_task_text(rounded)
    assert rounded_task["c_nmr_peaks"] == [{"delta (ppm)": 57.8}]
    assert rounded_task["h_nmr_peaks"][0]["delta"] == 1.18
    assert "rangeMin" not in rounded_task["h_nmr_peaks"][0]


def test_same_physical_input_agrees_across_entries():
    structured = normalize_task_input(PHYSICAL)
    from_json, _diagnostics = parse_task_text(task_to_lossless_text(PHYSICAL))
    args = SimpleNamespace(
        smiles="CCO",
        c_peaks="57.8123,18.2,57.8123",
        h_peaks="1.184(t,J=7.1,3H),3.7(q,J=7.1, 2.0,0H)",
        solvent="CDCl3",
        frequency="400 MHz",
        experimental_total_nH=6,
    )
    cli = call_v5_vllm.user_record_from_case(call_v5_vllm.build_case_from_flags(args))
    request = api_service.MatchRequest(
        candidate_smiles="CCO",
        c_nmr_peaks=copy.deepcopy(PHYSICAL["c_nmr_peaks"]),
        h_nmr_peaks=copy.deepcopy(PHYSICAL["h_nmr_peaks"]),
        nmr_solvent="CDCl3",
        nmr_frequency="400 MHz",
        nmr_frequency_nucleus="unknown",
        experimental_total_nH=6,
    )
    api_case = normalize_task_input(request.model_dump())
    vllm_case = call_v5_vllm.user_record_from_case(PHYSICAL)
    tool_case, error = _build_record_from_user_json(copy.deepcopy(PHYSICAL))
    assert error is None
    row = build_agent_parquet._row_from_parquet_row({
        "prompt": [{"role": "user", "content": task_to_lossless_text(PHYSICAL)}],
        "reward_model": {"ground_truth": {"candidate_smiles": "NOT-THE-TASK", "label": 0}},
        "ability": "NMR",
        "extra_info": {},
    })
    parquet_case = json.loads(row["prompt"][1]["content"])
    assert row["prompt"][1]["content"] == task_to_lossless_text(parquet_case)
    assert infer.make_messages(PHYSICAL)[1]["content"] == task_to_lossless_text(PHYSICAL)
    comparable = []
    for case in (structured, from_json, cli, api_case, vllm_case, tool_case, parquet_case):
        _no_gold(case)
        assert case["candidate_smiles"] == "CCO"
        assert case["c_nmr_peaks"] == structured["c_nmr_peaks"]
        assert case["nmr_solvent"] == "CDCl3"
        assert case["nmr_frequency"] == "400 MHz"
        assert case["nmr_frequency_nucleus"] == "unknown"
        assert case["experimental_total_nH"] == 6
        comparable.append(case["h_nmr_peaks"])
    assert comparable[0] == comparable[1] == comparable[3] == comparable[4] == comparable[5] == comparable[6]
    assert cli["h_nmr_peaks"][0]["j_values"] == "7.1"
    assert cli["h_nmr_peaks"][0]["nH"] == 3
    assert cli["h_nmr_peaks"][1]["nH"] == 0
    assert "rangeMin" not in cli["h_nmr_peaks"][0]
    assert parquet_case["candidate_smiles"] == "CCO"
    assert row["reward_model"]["ground_truth"]["candidate_smiles"] == "NOT-THE-TASK"


def test_alpaca_headers_do_not_invent_two_measured_frequencies():
    conflicted = (
        "Candidate SMILES: CCO\n"
        "13C NMR (100 MHz, CDCl3): 57.8\n"
        "1H NMR (400 MHz, DMSO): 1.18 (t, 3H)\n"
    )
    task, diagnostics = parse_task_text(conflicted)
    assert "nmr_frequency" not in task
    assert "nmr_solvent" not in task
    assert diagnostics["conflicts"]
    prepared = infer.prepare_eval_record({"input": conflicted, "_label": 1, "_gt_smiles": "CCC"}, 0)
    assert prepared["inference_input"]["nmr_frequency"] is None
    assert prepared["inference_input"]["nmr_frequency_nucleus"] == "unknown"
    assert prepared["inference_input"]["nmr_solvent"] is None
    assert prepared["input_sidecar"]["conflicts"]
    _no_gold(prepared["inference_input"])
    assert "_gt_smiles" not in prepared["inference_input"]


def test_missing_null_zero_and_partial_integrals_do_not_become_a_router_no():
    missing = NmrMatchingRuntime({
        "candidate_smiles": "CCO",
        "c_nmr_peaks": [{"delta (ppm)": 57.8}],
        "h_nmr_peaks": [{"delta": 1.2, "centroid": 1.2}, {"delta": 3.7, "centroid": 3.7, "nH": 2}],
    })
    missing_obs = json.loads(asyncio.run(missing.run_tool_call_async(
        {"function": {"name": "count_atoms", "arguments": {"element": "H"}}}
    ))["content"])
    assert missing_obs["experimental_total_nH"] is None
    assert missing_obs["match"] is None
    assert missing_obs["known_nH_sum"] == 2
    assert missing_obs["peak_table_complete"] is False

    explicit_zero = NmrMatchingRuntime({
        "candidate_smiles": "CCO",
        "c_nmr_peaks": [{"delta (ppm)": 57.8}],
        "h_nmr_peaks": [{"delta": 1.2, "nH": None}],
        "experimental_total_nH": 0,
    })
    zero_obs = json.loads(asyncio.run(explicit_zero.run_tool_call_async(
        {"function": {"name": "count_atoms", "arguments": {"element": "H"}}}
    ))["content"])
    assert zero_obs["experimental_total_nH"] == 0
    assert zero_obs["match"] is False

    conflict = NmrMatchingRuntime({
        "candidate_smiles": "CCO",
        "c_nmr_peaks": [{"delta (ppm)": 57.8}],
        "h_nmr_peaks": [{"delta": 1.2, "nH": 2}, {"delta": 3.7, "nH": 3}, {"delta": 2.5, "nH": 1}],
        "experimental_total_nH": 4,
    })
    conflict_obs = json.loads(asyncio.run(conflict.run_tool_call_async(
        {"function": {"name": "count_atoms", "arguments": {"element": "H"}}}
    ))["content"])
    assert conflict_obs["experimental_total_nH"] is None
    assert conflict_obs["match"] is None
    assert conflict_obs["integral_conflict"] is True
    assert conflict_obs["peak_table_nH_sum"] == 6

    router = RuleBasedRouter(
        {
            "candidate_smiles": "CCO",
            "c_nmr_peaks": [{"delta (ppm)": 57.8}],
            "h_nmr_peaks": [{"delta": 1.2, "nH": 2}, {"delta": 3.7, "nH": None}],
        },
        prediction_cache={},
        config=_router_config(),
        allow_remote_predict=False,
    )
    result = router.analyze()
    assert result.decision.h_available is False
    count_mismatch = (result.decision.c_available and not result.decision.c_match) or (
        result.decision.h_available and not result.decision.h_match
    )
    assert count_mismatch is False
    assert router.h_tol == 2


def test_gold_and_reward_candidate_do_not_enter_the_task():
    raw = copy.deepcopy(PHYSICAL)
    raw["h_nmr_peaks"][0]["label"] = 1
    raw["h_nmr_peaks"][0]["gt_smiles"] = "CCC"
    raw["_label"] = 0
    raw["idx"] = "case-1"
    raw["_gt_smiles"] = "CCC"
    before = copy.deepcopy(raw)
    prepared = infer.prepare_eval_record({"input": task_to_lossless_text(PHYSICAL), **raw}, 4)
    assert raw == before
    _no_gold(prepared["inference_input"])
    assert prepared["inference_input"]["candidate_smiles"] == "CCO"
    assert prepared["evaluation_meta"]["_gt_smiles"] == "CCC"
    messages = infer.make_messages(prepared["inference_input"])
    user = json.loads(messages[1]["content"])
    _no_gold(user)
    router = RuleBasedRouter(raw, prediction_cache={}, config=_router_config(), allow_remote_predict=False)
    _no_gold(router.record)
    tool_record, error = _build_record_from_user_json(raw)
    assert error is None
    _no_gold(tool_record)
    assert "idx" not in tool_record

    alpaca = {
        "input": (
            "Candidate SMILES: CCO\n"
            "13C NMR (400 MHz, CDCl3): 57.8\n"
            "1H NMR (400 MHz, CDCl3): 1.18 (t, 3H)\n"
        ),
        "_gt_smiles": "CCC",
        "_label": 0,
    }
    with pytest.raises(ValueError, match="candidate"):
        build_agent_parquet._row_from_alpaca(3, {"_gt_smiles": "CCC", "_label": 1})
    kept = build_agent_parquet._row_from_alpaca(3, {**alpaca, "_candidate_smiles": "CCO"})
    assert json.loads(kept["prompt"][1]["content"])["candidate_smiles"] == "CCO"
    assert kept["reward_model"]["ground_truth"]["gt_smiles"] == "CCC"
    with pytest.raises(ValueError, match="conflicts"):
        build_agent_parquet._row_from_alpaca(3, {**alpaca, "_candidate_smiles": "CCN"})


def test_bad_values_empty_spectrum_and_mixed_conflicts_are_rejected():
    with pytest.raises(Exception):
        normalize_task_input({
            "candidate_smiles": "CCO",
            "c_nmr_peaks": [{"delta (ppm)": float("nan")}],
            "h_nmr_peaks": [],
        })
    with pytest.raises(Exception):
        parse_task_text('{"candidate_smiles":"CCO","c_nmr_peaks":[{"delta (ppm)": Infinity}]}')
    with pytest.raises(ValueError, match="no experimental NMR peaks"):
        infer.prepare_eval_record({
            "candidate_smiles": "CCO",
            "c_nmr_peaks": [],
            "h_nmr_peaks": [],
        }, 0)
    with pytest.raises(ValueError, match="differs"):
        infer.prepare_eval_record({
            "input": (
                "Candidate SMILES: CCO\n"
                "13C NMR (400 MHz, CDCl3): 57.8\n"
                "1H NMR (400 MHz, CDCl3): 1.18 (t, 3H)\n"
            ),
            "nmr_solvent": "DMSO",
        }, 1)
    with pytest.raises(ValueError, match="differs"):
        infer.prepare_eval_record({
            "input": (
                "Candidate SMILES: CCO\n"
                "13C NMR (400 MHz, CDCl3): 57.8\n"
                "1H NMR (400 MHz, CDCl3): 1.18 (t, 3H)\n"
            ),
            "h_nmr_peaks": [{"delta": 9.9, "centroid": 9.9, "nH": 1, "category": "s"}],
        }, 2)
    with pytest.raises(Exception, match="unsupported"):
        call_v5_vllm.parse_h_peaks_arg("9.85(1H),not-a-peak")
    parsed = call_v5_vllm.parse_h_peaks_arg("9.85(),1.2")
    assert "nH" not in parsed[0]
    assert "nH" not in parsed[1]


def test_vllm_and_api_stubs_keep_the_physical_record(monkeypatch):
    captured = {}

    async def fake_chat(_client, _url, _model, messages):
        captured["messages"] = messages
        answer = '{"final_answer":"Yes","confidence":0.5,"key_evidence":"stub"}'
        return f"<answer>{answer}</answer>", {"role": "assistant", "content": answer}

    monkeypatch.setattr(call_v5_vllm, "chat_turn", fake_chat)
    result = asyncio.run(call_v5_vllm.run_case("http://127.0.0.1:9/v1", "stub", PHYSICAL, {}))
    assert result["final"]["final_answer"] == "Yes"
    user = json.loads(captured["messages"][1]["content"])
    assert user["experimental_total_nH"] == 6
    assert user["nmr_frequency_nucleus"] == "unknown"
    _no_gold(user)

    class ApiRouter:
        def __init__(self, case, **_kwargs):
            captured["api_case"] = copy.deepcopy(case)

        def analyze(self):
            return SimpleNamespace(route="model")

    async def fake_agent(_tok, _model, case, max_turns=6, tools=None):
        captured["agent_case"] = copy.deepcopy(case)
        return {"answer": '<answer>{"final_answer":"No","confidence":0.2,"key_evidence":"stub"}</answer>'}

    monkeypatch.setattr(api_service, "RuleBasedRouter", ApiRouter)
    monkeypatch.setattr(api_service, "run_agent", fake_agent)
    state = api_service.ServiceState()
    state.ready = True
    state.router_cache = {}
    state.router_config = _router_config()
    state.tokenizer = object()
    state.model = object()
    state.tools = {}
    response = state.infer(api_service.MatchRequest(
        candidate_smiles="CCO",
        c_nmr_peaks=[{"delta (ppm)": 57.8, "label": 1}],
        h_nmr_peaks=[{"delta": 1.2, "centroid": 1.2, "nH": None, "gt_smiles": "CCC"}],
        experimental_total_nH=0,
        nmr_frequency="101 MHz",
        nmr_frequency_nucleus="1H",
    ))
    assert response.final_answer == "No"
    assert captured["api_case"]["experimental_total_nH"] == 0
    assert captured["api_case"]["nmr_frequency_nucleus"] == "1H"
    assert captured["api_case"]["h_nmr_peaks"][0]["nH"] is None
    _no_gold(captured["api_case"])
    _no_gold(captured["agent_case"])
    assert captured["api_case"] == captured["agent_case"]
