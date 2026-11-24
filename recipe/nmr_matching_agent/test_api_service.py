from __future__ import annotations

import json

from fastapi.testclient import TestClient

from recipe.nmr_matching_agent import api_service


def result_fixture() -> dict:
    grounded = (
        '<think>spectral_match returned similarity=0.4089. Therefore final_answer = No.</think>\n'
        '<answer>{"final_answer":"No","confidence":1.0,'
        '"key_evidence":"spectral_match similarity=0.4089"}</answer>'
    )
    return {
        "answer": grounded,
        "fail_reason": None,
        "_route": "model",
        "_template_id": "must_not_leak",
        "transcript": [
            {
                "turn": 1,
                "assistant": '<think>Call the tool.</think><tool_call>{"name":"spectral_match","arguments":{}}</tool_call>',
                "action": {"type": "tool", "name": "spectral_match", "arguments": {}},
                "tool_observation": json.dumps({"similarity": 0.4089, "ok": True}),
            },
            {
                "turn": 2,
                "assistant": '<think>Raw model final think.</think><answer>{"final_answer":"No","confidence":1.0,"key_evidence":"hallucinated secret"}</answer>',
                "action": {"type": "answer", "answer": "raw model answer must not leak"},
            },
        ],
    }


def test_public_response_is_minimal_and_keeps_tool_trace() -> None:
    response = api_service.public_response(result_fixture()).model_dump()
    assert response == {
        "final_answer": "No",
        "final_think": "spectral_match returned similarity=0.4089. Therefore final_answer = No.",
        "key_evidence": "spectral_match similarity=0.4089",
        "trajectory": [
            {
                "turn": 1,
                "think": "Call the tool.",
                "tool_call": {"name": "spectral_match", "arguments": {}},
                "tool_output": {"similarity": 0.4089, "ok": True},
                "final_answer": None,
            },
            {
                "turn": 2,
                "think": "Raw model final think.",
                "tool_call": None,
                "tool_output": None,
                "final_answer": "No",
            },
        ],
    }
    encoded = json.dumps(response)
    for forbidden in ("confidence", "_route", "template", "hallucinated secret", "raw model answer"):
        assert forbidden not in encoded


def test_health_and_match_routes_without_loading_model(monkeypatch) -> None:
    monkeypatch.setenv("NMR_MATCHING_SKIP_STARTUP_LOAD", "1")
    expected = api_service.public_response(result_fixture())
    monkeypatch.setattr(api_service.state, "infer", lambda payload: expected)
    with TestClient(api_service.app) as client:
        assert client.get("/health").json() == {"ready": False}
        response = client.post("/match", json={
            "candidate_smiles": "CC",
            "c_nmr_peaks": [{"delta (ppm)": 10.0}],
        })
        assert response.status_code == 200
        assert response.json()["final_answer"] == "No"
        assert set(response.json()) == {"final_answer", "final_think", "key_evidence", "trajectory"}


def test_request_requires_at_least_one_spectrum(monkeypatch) -> None:
    monkeypatch.setenv("NMR_MATCHING_SKIP_STARTUP_LOAD", "1")
    with TestClient(api_service.app) as client:
        response = client.post("/match", json={"candidate_smiles": "CC"})
        assert response.status_code == 422


def test_optional_model_reasoning_does_not_dilute_no_evidence() -> None:
    tagged = result_fixture()["answer"]
    augmented = api_service._prepend_model_reasoning(tagged, "Model reasoning from final think.")
    obj = api_service._answer_object(augmented)
    assert obj["key_evidence"] == "spectral_match similarity=0.4089"
    assert obj["final_answer"] == "No" and obj["confidence"] == 1.0


def test_prefilled_reasoning_uses_router_messages_without_tools(monkeypatch) -> None:
    routed = type("Routed", (), {"tool_results": type("Tools", (), {
        "count_c": {"element": "C", "candidate_count": 2, "experimental_distinct_signals": 2, "match": True},
        "count_h": {"element": "H", "candidate_count": 6, "experimental_total_nH": 5, "match": True},
        "spectral": {"similarity": 0.8263, "ok": True},
        "fg_scan": {"groups_present": ["hydroxyl"], "n_groups": 1},
    })()})()
    captured = {}
    def fake_generate(tok, model, messages, use_tools):
        captured["messages"] = messages; captured["use_tools"] = use_tools
        return '<think>Reasoning from the supplied observations. Therefore final_answer = Yes.</think><answer>{"final_answer":"Yes"}</answer>'
    monkeypatch.setattr(api_service, "generate_once", fake_generate)
    reasoning = api_service._prefilled_model_reasoning(object(), object(), {"candidate_smiles": "CCO"}, routed)
    assert reasoning == "Reasoning from the supplied observations."
    assert captured["use_tools"] is False
    assert sum(m["role"] == "tool" for m in captured["messages"]) == 4


def test_template_true_uses_prefilled_once_not_agent(monkeypatch) -> None:
    final = '<think>Template think.</think><answer>{"final_answer":"Yes","confidence":0.9,"key_evidence":"tool template"}</answer>'
    routed = type("Routed", (), {"route": "template", "full_assistant_turn": final})()
    monkeypatch.setattr(api_service, "RuleBasedRouter", lambda *a, **k: type("R", (), {"analyze": lambda self: routed})())
    monkeypatch.setattr(api_service, "build_template_transcript", lambda *a: [{"turn": 1, "assistant": final, "action": {"type": "answer", "answer": "{}"}}])
    calls = {"prefilled": 0, "agent": 0}
    def prefilled(*a): calls["prefilled"] += 1; return "Model reasoning."
    async def agent(*a, **k): calls["agent"] += 1; return result_fixture()
    monkeypatch.setattr(api_service, "_prefilled_model_reasoning", prefilled)
    monkeypatch.setattr(api_service, "run_agent", agent)
    s=api_service.ServiceState();s.ready=True;s.router_cache={};s.router_config={};s.tokenizer=object();s.model=object();s.tools={}
    out=s.infer(api_service.MatchRequest(candidate_smiles="CC",c_nmr_peaks=[{"delta (ppm)":1}],include_model_reasoning_in_key_evidence=True))
    assert calls=={"prefilled":1,"agent":0}
    assert out.key_evidence=="Model reasoning. tool template"


def test_model_path_still_runs_autonomous_agent(monkeypatch) -> None:
    routed = type("Routed", (), {"route": "model"})()
    monkeypatch.setattr(api_service, "RuleBasedRouter", lambda *a, **k: type("R", (), {"analyze": lambda self: routed})())
    calls={"agent":0,"prefilled":0}
    async def agent(*a, **k): calls["agent"]+=1; return result_fixture()
    monkeypatch.setattr(api_service, "run_agent", agent)
    monkeypatch.setattr(api_service, "_prefilled_model_reasoning", lambda *a: calls.__setitem__("prefilled",calls["prefilled"]+1))
    s=api_service.ServiceState();s.ready=True;s.router_cache={};s.router_config={};s.tokenizer=object();s.model=object();s.tools={}
    out=s.infer(api_service.MatchRequest(candidate_smiles="CC",c_nmr_peaks=[{"delta (ppm)":1}],include_model_reasoning_in_key_evidence=True))
    assert calls=={"agent":1,"prefilled":0}
    assert out.final_answer=="No"
