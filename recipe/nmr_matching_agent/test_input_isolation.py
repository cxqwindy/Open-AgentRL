"""CPU regression tests for evaluation-metadata isolation.

The assertions inspect messages and records captured at generate_once, tool
create, the CLI branch loop, the router, and the API. They do not check that a
helper was merely called.
"""

from __future__ import annotations

import asyncio
import copy
import json
import socket
from types import SimpleNamespace

import pytest

from recipe.nmr_matching_agent import api_service
from recipe.nmr_matching_agent import standalone_infer_matching as infer
from recipe.nmr_matching_agent.rule_based_router import RuleBasedRouter, build_prefilled_messages
from recipe.nmr_matching_agent.task_input import normalize_task_input

FORBIDDEN_KEYS = {
    "idx",
    "case_id",
    "source_id",
    "id",
    "label",
    "_label",
    "negative_type",
    "_negative_type",
    "gt_smiles",
    "_gt_smiles",
    "output",
    "teacher_pred",
    "_candidate_smiles",
    "include_model_reasoning_in_key_evidence",
    "instruction",
    "pair_id",
    "_pair_id",
    "_sample_id",
}

FIXED_GENERATION = '<answer>{"final_answer":"No","confidence":0.5,"key_evidence":"fixed"}</answer>'


@pytest.fixture(autouse=True)
def block_model_and_network(monkeypatch):
    def blocked(*_args, **_kwargs):
        raise RuntimeError("model load blocked")

    monkeypatch.setattr(infer, "load_model_and_tokenizer", blocked)
    monkeypatch.setattr(infer, "load_tools", blocked)

    def blocked_connection(*_args, **_kwargs):
        raise RuntimeError("network blocked")

    monkeypatch.setattr(socket, "create_connection", blocked_connection)


def _keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _keys(item)


def assert_task_clean(payload) -> dict:
    parsed = json.loads(payload) if isinstance(payload, str) else payload
    leaked = set(_keys(parsed)) & FORBIDDEN_KEYS
    assert not leaked, leaked
    return parsed


def user_content(messages) -> str:
    users = [message["content"] for message in messages if message["role"] == "user"]
    assert len(users) == 1
    return users[0]


def physical_case() -> dict:
    return {
        "candidate_smiles": "CCO",
        "c_nmr_peaks": [{"delta (ppm)": 57.8}, {"delta (ppm)": 18.2}],
        "h_nmr_peaks": [{
            "category": "t",
            "centroid": 1.18,
            "delta": 1.18,
            "j_values": [7.1],
            "nH": 3,
            "rangeMax": 1.2,
            "rangeMin": 1.16,
        }],
        "nmr_solvent": "CDCl3",
        "nmr_frequency": "400 MHz",
        "experimental_total_nH": None,
    }


def with_metadata(case: dict, **extra) -> dict:
    dirty = copy.deepcopy(case)
    dirty.update(extra)
    dirty["c_nmr_peaks"] = [
        {**peak, "label": extra.get("_label", 1), "gt_smiles": extra.get("_gt_smiles", "CCO")}
        for peak in dirty["c_nmr_peaks"]
    ]
    dirty["h_nmr_peaks"] = [
        {**peak, "_negative_type": extra.get("_negative_type", "positive"), "teacher_pred": "Yes"}
        for peak in dirty["h_nmr_peaks"]
    ]
    return dirty


class CaptureTool:
    def __init__(self):
        self.raw_prompts = []

    async def create(self, instance_id=None, create_kwargs=None, **_kwargs):
        self.raw_prompts.append(copy.deepcopy((create_kwargs or {}).get("raw_prompt")))
        return None

    async def release(self, instance_id):
        return None

    async def execute(self, instance_id, arguments):
        return SimpleNamespace(text="{}"), 0.0, {}


def install_capture(monkeypatch):
    captured = []

    def generate(_tok, _model, messages, max_new_tokens=1024, use_tools=True):
        captured.append(copy.deepcopy(messages))
        return FIXED_GENERATION

    monkeypatch.setattr(infer, "generate_once", generate)
    return captured


def test_t1_metadata_does_not_change_generate_messages(monkeypatch):
    captured = install_capture(monkeypatch)
    tool = CaptureTool()
    base = physical_case()
    variants = [
        base,
        with_metadata(base, _label=1, _negative_type="positive", _gt_smiles="CCO", idx="case-a"),
        with_metadata(base, _label=0, _negative_type="neg/shift", _gt_smiles="CC", idx="case-b", output="No"),
        with_metadata(base, label=1, negative_type="positive", gt_smiles="CCC", case_id="other", teacher_pred="Yes"),
    ]
    for case in variants:
        before = copy.deepcopy(case)
        asyncio.run(infer.run_agent(None, None, case, tools={"spectral_match": tool}))
        assert case == before

    prompts = [user_content(messages) for messages in captured]
    assert len(prompts) == len(variants)
    assert len(set(prompts)) == 1
    task = assert_task_clean(prompts[0])
    assert task["candidate_smiles"] == "CCO"
    assert task["c_nmr_peaks"] == base["c_nmr_peaks"]
    assert task["h_nmr_peaks"] == base["h_nmr_peaks"]
    assert all(message["content"] == captured[0][0]["content"] for message in (item[0] for item in captured))


def _alpaca(smiles: str, carbon: str) -> str:
    return (
        f"Candidate SMILES: {smiles}\n"
        f"13C NMR (400 MHz, CDCl3): {carbon}\n"
        "1H NMR (400 MHz, CDCl3): 3.65 (q, 2H), 1.18 (t, 3H)\n"
    )


def _template_router(case, **_kwargs):
    decision = SimpleNamespace(
        template_id="T5",
        template_variant_id="T5-en",
        template_language="en",
        template_variant_digest="0123456789abcdef",
        c_available=True,
        h_available=False,
        c_match=True,
        h_match=False,
        assignment_vetoed=False,
        assignment_veto_reasons=[],
        assignment_shadow_mode=True,
        confidence=0.9,
    )
    observations = SimpleNamespace(
        count_c={"element": "C", "candidate_count": 2, "experimental_distinct_signals": 1, "match": True},
        count_h={"element": "H", "candidate_count": 6, "experimental_total_nH": None, "match": None},
        spectral={"similarity": 0.95, "ok": True, "c_sim": 0.95, "h_sim": 0.0},
        fg_scan={"groups_present": [], "n_groups": 0},
        peak_c=None,
        peak_h=None,
    )
    final = (
        '<think>Template think.</think>\n'
        '<answer>{"final_answer":"Yes","confidence":0.90,"key_evidence":"tool template"}</answer>'
    )
    return SimpleNamespace(
        route="template",
        full_assistant_turn=final,
        decision=decision,
        tool_results=observations,
        template_variant_seed=0,
    )


class RecordingRouter:
    seen = []

    def __init__(self, case, prediction_cache=None, config=None, allow_remote_predict=True):
        self.case = copy.deepcopy(case)
        self.template_variant_seed = (config or {}).get("template_variant_seed", 0)
        RecordingRouter.seen.append(self.case)
        self._route = "template" if (config or {}).get("_test_route") == "template" else "model"

    def analyze(self):
        if self._route == "template":
            result = _template_router(self.case)
            result.template_variant_seed = self.template_variant_seed
            return result
        return SimpleNamespace(
            route="model",
            decision=SimpleNamespace(
                assignment_vetoed=False,
                assignment_veto_reasons=[],
                assignment_shadow_mode=True,
            ),
        )


def _write_from_test(tmp_path):
    records = [
        {
            "input": _alpaca("CCO", "57.8, 18.2"),
            "output": "Yes",
            "_label": 1,
            "_negative_type": "positive",
            "_gt_smiles": "CCO",
            "_candidate_smiles": "CCO",
            "idx": "dup",
        },
        {
            "input": _alpaca("CC", "30.0"),
            "output": "No",
            "_label": 0,
            "_negative_type": "neg/decoy",
            "_gt_smiles": "CCC",
            "_candidate_smiles": "CC",
            "idx": "dup",
        },
        {
            "candidate_smiles": "CCO",
            "c_nmr_peaks": [{"delta (ppm)": 57.8, "label": "Yes", "gt_smiles": "CCO"}],
            "h_nmr_peaks": [{"delta": 1.18, "centroid": 1.18, "nH": None, "category": "No", "_label": 0}],
            "nmr_solvent": "CDCl3",
            "_label": 0,
            "_negative_type": "neg/shift",
            "_gt_smiles": "CC",
            "output": "No",
            "teacher_pred": "Yes",
        },
    ]
    path = tmp_path / "from_test.json"
    path.write_text(json.dumps(records), encoding="utf-8")
    return path, records


def test_t2_from_test_model_router_and_template(tmp_path, monkeypatch):
    path, records = _write_from_test(tmp_path)
    items = infer.load_from_test(str(path), None)
    assert items[0]["inference_input"]["candidate_smiles"] == "CCO"
    # 02B-1 reads the explicit Alpaca header. The same string on both headers
    # stays one raw frequency and does not become a 13C and a 1H measurement.
    assert items[0]["inference_input"]["nmr_solvent"] == "CDCl3"
    assert items[0]["inference_input"]["nmr_frequency"] == "400 MHz"
    assert items[0]["inference_input"]["nmr_frequency_nucleus"] == "unknown"
    assert items[0]["inference_input"]["h_nmr_peaks"][0]["nH"] == 2
    assert items[0]["pair_id"] != items[1]["pair_id"]
    assert items[0]["evaluation_meta"]["sample_id"] == items[1]["evaluation_meta"]["sample_id"] == "dup"
    assert_task_clean(items[2]["inference_input"])
    assert items[2]["inference_input"]["h_nmr_peaks"][0]["category"] == "No"
    assert items[2]["inference_input"]["h_nmr_peaks"][0]["nH"] is None

    seen_cases = []

    async def fake_agent(_tok, _model, case, max_turns=6, tools=None):
        seen_cases.append(copy.deepcopy(case))
        return {
            "answer": "FIXED",
            "fail_reason": None,
            "tool_sequence": [],
            "transcript": [],
            "case_candidate": case.get("candidate_smiles"),
        }

    monkeypatch.setattr(infer, "run_agent", fake_agent)
    monkeypatch.setattr(
        "recipe.nmr_matching_agent.rule_based_router.RuleBasedRouter",
        RecordingRouter,
    )

    def no_model():
        raise AssertionError("pure template path requested a model")

    RecordingRouter.seen = []
    template_cfg = {"_test_route": "template", "template_variant_seed": 0}
    templated = infer.run_evaluation(
        items,
        use_router=True,
        router_cfg=template_cfg,
        router_cache={},
        max_turns=6,
        ensure_model=no_model,
        skip_model_evidence=False,
    )
    assert seen_cases == []
    assert [row["_route"] for row in templated] == ["template", "template", "template"]
    assert [row["_label"] for row in templated] == [1, 0, 0]
    assert [row["_negative_type"] for row in templated] == ["positive", "neg/decoy", "neg/shift"]
    assert templated[0]["_gt_smiles"] == "CCO"
    assert templated[1]["_sample_id"] == "dup" and templated[2]["_sample_id"] is None
    assert all("output" not in row for row in templated)
    assert all(row["answer"] != "Yes" and row["_label"] != "FIXED" for row in templated)
    for seen, item in zip(RecordingRouter.seen, items):
        assert seen == item["inference_input"]
        assert_task_clean(seen)

    seen_cases.clear()
    RecordingRouter.seen = []
    routed = infer.run_evaluation(
        items,
        use_router=True,
        router_cfg={"_test_route": "model"},
        router_cache={},
        max_turns=6,
        ensure_model=lambda: (None, None),
        skip_model_evidence=False,
    )
    assert seen_cases == [item["inference_input"] for item in items]
    assert [row["_route"] for row in routed] == ["model", "model", "model"]
    assert [row["_label"] for row in routed] == [1, 0, 0]
    assert routed[0]["_pair_id"] == 0 and routed[1]["_pair_id"] == 1
    for case in seen_cases:
        assert_task_clean(case)

    plain = infer.run_evaluation(
        items,
        use_router=False,
        router_cfg=None,
        router_cache=None,
        max_turns=6,
        ensure_model=lambda: (None, None),
        skip_model_evidence=False,
    )
    assert [row["_label"] for row in plain] == [1, 0, 0]
    assert [row["_negative_type"] for row in plain] == [record.get("_negative_type") for record in records]
    assert plain[0]["answer"] == "FIXED" and plain[0]["_label"] == 1


def test_t3_direct_run_agent_strips_metadata(monkeypatch):
    captured = install_capture(monkeypatch)
    tool = CaptureTool()
    dirty = with_metadata(
        physical_case(),
        _label=0,
        _negative_type="neg/shift",
        _gt_smiles="CC",
        idx="direct",
        output="No",
        teacher_pred="Yes",
    )
    original = copy.deepcopy(dirty)
    asyncio.run(infer.run_agent(None, None, dirty, tools={"spectral_match": tool}))
    assert dirty == original
    assert tool.raw_prompts and captured
    prompt_user = user_content(tool.raw_prompts[0])
    generated_user = user_content(captured[0])
    assert prompt_user == generated_user
    task = assert_task_clean(prompt_user)
    assert task["h_nmr_peaks"][0]["nH"] == 3
    assert task["h_nmr_peaks"][0]["j_values"] == [7.1]
    assert "label" not in task["c_nmr_peaks"][0]


def test_t4_preserves_physical_fields_aliases_and_null_integrals():
    raw = {
        "candidate": "CCC",
        "candidate_smiles": "",
        "experimental_13c_peaks": [57.8, 18],
        "c_nmr_peaks": [],
        "experimental_1h_peaks": [1.18],
        "solvent": "CDCl3",
        "nmr_freq": "400 MHz",
        "experimental_total_nH": 0,
        "_label": 1,
        "gt_smiles": "CCO",
        "idx": 7,
    }
    snapshot = copy.deepcopy(raw)
    normalized = normalize_task_input(raw)
    assert raw == snapshot
    assert normalized["candidate_smiles"] == "CCC"
    assert normalized["c_nmr_peaks"] == [{"delta (ppm)": 57.8}, {"delta (ppm)": 18.0}]
    assert normalized["h_nmr_peaks"] == [{"delta": 1.18, "centroid": 1.18, "nH": None}]
    assert normalized["nmr_solvent"] == "CDCl3"
    assert normalized["nmr_frequency"] == "400 MHz"
    assert normalized["experimental_total_nH"] == 0
    assert normalize_task_input(normalized) == normalized
    assert infer.build_user_case(raw) == normalized
    assert "idx" not in normalized

    priority = {
        "candidate_smiles": "CCO",
        "candidate": "CCC",
        "c_nmr_peaks": [{"delta (ppm)": 10.0, "label": 1, "gt_smiles": "CC"}],
        "experimental_13c_peaks": [99.0],
        "h_nmr_peaks": [{
            "category": "No",
            "centroid": 2.5,
            "delta": 2.5,
            "j_values": [3.0, 8.0],
            "nH": None,
            "rangeMax": 2.6,
            "rangeMin": 2.4,
            "_gt_smiles": "CCO",
        }],
        "experimental_1h_peaks": [0.1],
        "nmr_solvent": "DMSO",
        "solvent": "CDCl3",
        "nmr_frequency": "500 MHz",
        "nmr_freq": "400 MHz",
        "experimental_total_nH": 6,
    }
    priority_snapshot = copy.deepcopy(priority)
    kept = normalize_task_input(priority)
    kept["c_nmr_peaks"][0]["delta (ppm)"] = 123
    assert priority == priority_snapshot
    kept = normalize_task_input(priority)
    assert kept["candidate_smiles"] == "CCO"
    assert kept["c_nmr_peaks"] == [{"delta (ppm)": 10.0}]
    assert kept["h_nmr_peaks"] == [{
        "category": "No",
        "centroid": 2.5,
        "delta": 2.5,
        "j_values": [3.0, 8.0],
        "nH": None,
        "rangeMax": 2.6,
        "rangeMin": 2.4,
    }]
    assert kept["experimental_total_nH"] == 6
    assert kept["nmr_solvent"] == "DMSO"
    assert kept["nmr_frequency"] == "500 MHz"
    assert normalize_task_input(kept) == kept
    message = json.loads(infer.make_messages(priority)[1]["content"])
    assert message == kept

    missing_integral = {
        "candidate_smiles": "C",
        "c_nmr_peaks": [{"delta (ppm)": 1.0}],
        "h_nmr_peaks": [{"delta": 1.0, "centroid": 1.0, "category": "s"}],
        "experimental_total_nH": 6,
    }
    projected = normalize_task_input(missing_integral)
    assert "nH" not in projected["h_nmr_peaks"][0]
    assert projected["experimental_total_nH"] == 6


def test_direct_router_and_prefilled_messages_ignore_metadata():
    clean = physical_case()
    dirty = with_metadata(clean, _label=1, _negative_type="positive", _gt_smiles="CCO", case_id="r1")
    dirty_snapshot = copy.deepcopy(dirty)
    clean_router = RuleBasedRouter(clean, prediction_cache={}, allow_remote_predict=False)
    dirty_router = RuleBasedRouter(dirty, prediction_cache={}, allow_remote_predict=False)
    assert dirty == dirty_snapshot
    assert clean_router.record == dirty_router.record
    assert_task_clean(clean_router.record)
    clean_variant = clean_router._select_template_variant("T1")
    dirty_variant = dirty_router._select_template_variant("T1")
    assert clean_variant[0].variant_id == dirty_variant[0].variant_id
    assert clean_variant[1] == dirty_variant[1]
    clean_result = clean_router.analyze()
    dirty_result = dirty_router.analyze()
    assert clean_result.route == dirty_result.route
    assert clean_result.decision.template_id == dirty_result.decision.template_id
    assert clean_result.decision.template_variant_digest == dirty_result.decision.template_variant_digest

    observations = SimpleNamespace(
        count_c={"element": "C"},
        count_h={"element": "H"},
        spectral={"ok": False},
        fg_scan={"n_groups": 0},
        peak_c=None,
        peak_h=None,
    )
    routed = SimpleNamespace(
        tool_results=observations,
        decision=SimpleNamespace(assignment_vetoed=False, assignment_shadow_mode=True),
    )
    messages = build_prefilled_messages(dirty, routed, "system")
    assert json.loads(user_content(messages)) == clean_router.record
    assert_task_clean(user_content(messages))


def _api_request(**overrides):
    payload = {
        "candidate_smiles": "CC",
        "c_nmr_peaks": [{"delta (ppm)": 10.0, "_label": 1, "gt_smiles": "C"}],
        "h_nmr_peaks": [{"delta": 1.2, "centroid": 1.2, "nH": None, "category": "No", "label": "Yes"}],
        "include_model_reasoning_in_key_evidence": False,
    }
    payload.update(overrides)
    return api_service.MatchRequest(**payload)


def _ready_state():
    state = api_service.ServiceState()
    state.ready = True
    state.router_cache = {}
    state.router_config = {}
    state.tokenizer = object()
    state.model = object()
    state.tools = {}
    state.max_turns = 6
    return state


def test_t5_api_three_paths_keep_response_shape(monkeypatch):
    captured = {}

    class ApiRouter:
        def __init__(self, case, **_kwargs):
            captured["router_case"] = copy.deepcopy(case)

        def analyze(self):
            return captured["routed"]

    monkeypatch.setattr(api_service, "RuleBasedRouter", ApiRouter)

    final = (
        '<think>Template think.</think>'
        '<answer>{"final_answer":"Yes","confidence":0.9,"key_evidence":"tool template"}</answer>'
    )
    decision = SimpleNamespace(
        assignment_vetoed=False,
        assignment_shadow_mode=True,
        template_language="en",
    )
    observations = SimpleNamespace(
        count_c={"element": "C", "candidate_count": 2, "experimental_distinct_signals": 1, "match": True},
        count_h={"element": "H", "candidate_count": 6, "experimental_total_nH": 6, "match": True},
        spectral={"similarity": 0.95, "ok": True},
        fg_scan={"groups_present": ["methyl"], "n_groups": 1},
        peak_c=None,
        peak_h=None,
    )
    captured["routed"] = SimpleNamespace(
        route="template",
        full_assistant_turn=final,
        decision=decision,
        tool_results=observations,
    )

    def fail_agent(*_args, **_kwargs):
        raise AssertionError("template path called run_agent")

    monkeypatch.setattr(api_service, "run_agent", fail_agent)
    state = _ready_state()
    quiet = state.infer(_api_request(include_model_reasoning_in_key_evidence=False))
    assert quiet.final_answer == "Yes"
    assert quiet.key_evidence == "tool template"
    assert [turn.tool_call.name for turn in quiet.trajectory if turn.tool_call] == [
        "count_atoms", "count_atoms", "spectral_match", "functional_group_scan",
    ]
    assert quiet.trajectory[-1].final_answer == "Yes"
    task = assert_task_clean(captured["router_case"])
    assert task["h_nmr_peaks"][0]["nH"] is None
    assert task["h_nmr_peaks"][0]["category"] == "No"
    assert "include_model_reasoning_in_key_evidence" not in captured["router_case"]

    def fake_generate(_tok, _model, messages, max_new_tokens=1024, use_tools=True):
        captured["messages"] = copy.deepcopy(messages)
        captured["use_tools"] = use_tools
        return (
            '<think>Reasoning from the supplied observations. Therefore final_answer = Yes.</think>'
            '<answer>{"final_answer":"Yes"}</answer>'
        )

    monkeypatch.setattr(api_service, "generate_once", fake_generate)
    explained = state.infer(_api_request(include_model_reasoning_in_key_evidence=True))
    assert explained.final_answer == "Yes"
    assert explained.key_evidence == "Reasoning from the supplied observations. tool template"
    assert captured["use_tools"] is False
    explained_task = assert_task_clean(user_content(captured["messages"]))
    assert explained_task == task
    assert any(turn.tool_call and turn.tool_call.name == "spectral_match" for turn in explained.trajectory)

    captured["routed"] = SimpleNamespace(route="model")
    calls = {"agent": 0}

    async def agent(_tok, _model, case, max_turns=6, tools=None):
        calls["agent"] += 1
        captured["agent_case"] = copy.deepcopy(case)
        return {
            "answer": (
                '<think>spectral_match returned similarity=0.4089. Therefore final_answer = No.</think>\n'
                '<answer>{"final_answer":"No","confidence":1.0,'
                '"key_evidence":"spectral_match similarity=0.4089"}</answer>'
            ),
            "fail_reason": None,
            "transcript": [
                {
                    "turn": 1,
                    "assistant": '<think>Call the tool.</think><tool_call>{"name":"spectral_match","arguments":{}}</tool_call>',
                    "action": {"type": "tool", "name": "spectral_match", "arguments": {}},
                    "tool_observation": json.dumps({"similarity": 0.4089, "ok": True}),
                },
                {
                    "turn": 2,
                    "assistant": (
                        '<think>Raw model final think.</think>'
                        '<answer>{"final_answer":"No","confidence":1.0,"key_evidence":"hallucinated secret"}</answer>'
                    ),
                    "action": {"type": "answer", "answer": "raw"},
                },
            ],
        }

    monkeypatch.setattr(api_service, "run_agent", agent)
    modeled = state.infer(_api_request(include_model_reasoning_in_key_evidence=True))
    assert calls == {"agent": 1}
    assert modeled.final_answer == "No"
    assert modeled.key_evidence == "spectral_match similarity=0.4089"
    assert modeled.trajectory[0].tool_call.name == "spectral_match"
    assert_task_clean(captured["agent_case"])
    assert captured["agent_case"] == task


def test_t6_reordered_results_keep_their_own_gold(tmp_path, monkeypatch):
    records = [
        {
            "case_id": "a",
            "candidate_smiles": "CCO",
            "c_nmr_peaks": [{"delta (ppm)": 57.8}],
            "h_nmr_peaks": [],
            "_label": 1,
            "_negative_type": "positive",
            "_gt_smiles": "CCO",
            "output": "Yes",
        },
        {
            "case_id": "b",
            "candidate_smiles": "CCO",
            "c_nmr_peaks": [{"delta (ppm)": 18.2}],
            "h_nmr_peaks": [],
            "_label": 0,
            "_negative_type": "neg/shift",
            "_gt_smiles": "CC",
            "output": "No",
        },
        {
            "candidate_smiles": "CCC",
            "c_nmr_peaks": [{"delta (ppm)": 11.0}],
            "h_nmr_peaks": [],
            "_label": 0,
            "_negative_type": "neg/random",
            "_gt_smiles": "CCO",
            "output": "No",
        },
        {
            "idx": "dup",
            "candidate_smiles": "CC",
            "c_nmr_peaks": [{"delta (ppm)": 5.0}],
            "h_nmr_peaks": [],
            "_label": 1,
            "_negative_type": "positive",
            "_gt_smiles": "CC",
            "output": "Yes",
        },
        {
            "idx": "dup",
            "candidate_smiles": "CC",
            "c_nmr_peaks": [{"delta (ppm)": 30.0}],
            "h_nmr_peaks": [],
            "_label": 0,
            "_negative_type": "neg/decoy",
            "_gt_smiles": "CCC",
            "output": "No",
        },
    ]
    path = tmp_path / "pairs.json"
    path.write_text(json.dumps(records), encoding="utf-8")
    items = infer.load_from_test(str(path), None)
    assert [item["evaluation_meta"]["sample_id"] for item in items] == ["a", "b", None, "dup", "dup"]
    assert items[0]["inference_input"]["candidate_smiles"] == items[1]["inference_input"]["candidate_smiles"]
    assert items[0]["inference_input"] != items[1]["inference_input"]
    for item in items:
        assert "pair_id" not in item["inference_input"]
        assert_task_clean(item["inference_input"])

    async def fake_agent(_tok, _model, case, max_turns=6, tools=None):
        shift = case["c_nmr_peaks"][0]["delta (ppm)"]
        return {
            "answer": f"shift={shift}",
            "fail_reason": None,
            "tool_sequence": [],
            "transcript": [],
            "case_candidate": case["candidate_smiles"],
            "_label": "from-model",
            "_negative_type": "from-model",
        }

    monkeypatch.setattr(infer, "run_agent", fake_agent)
    shuffled = list(reversed(items))
    executed = infer.run_evaluation(
        shuffled,
        use_router=False,
        router_cfg=None,
        router_cache=None,
        max_turns=6,
        ensure_model=lambda: (None, None),
        skip_model_evidence=False,
    )
    assert [row["_pair_id"] for row in executed] == [4, 3, 2, 1, 0]
    merged = infer.reassemble_eval_results(executed)
    assert [row["_pair_id"] for row in merged] == [0, 1, 2, 3, 4]
    assert [row["_label"] for row in merged] == [1, 0, 0, 1, 0]
    assert [row["_negative_type"] for row in merged] == [
        "positive", "neg/shift", "neg/random", "positive", "neg/decoy",
    ]
    assert [row["_gt_smiles"] for row in merged] == ["CCO", "CC", "CCO", "CC", "CCC"]
    assert [row["answer"] for row in merged] == ["shift=57.8", "shift=18.2", "shift=11.0", "shift=5.0", "shift=30.0"]
    assert merged[0]["case_candidate"] == merged[1]["case_candidate"] == "CCO"
    assert merged[0]["_label"] != merged[1]["_label"]
    assert merged[2]["_sample_id"] is None
    assert merged[3]["_sample_id"] == merged[4]["_sample_id"] == "dup"
    assert merged[3]["answer"] != merged[4]["answer"]
    assert all("output" not in row for row in merged)
    assert all(row["_label"] != "from-model" for row in merged)
