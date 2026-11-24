from __future__ import annotations

import json

from recipe.nmr_matching_agent.rule_based_router import (
    RouteDecision,
    RuleBasedRouter,
    ToolResults,
)
from recipe.nmr_matching_agent.standalone_infer_matching import build_template_transcript


def fixture_results() -> tuple[ToolResults, RouteDecision]:
    results = ToolResults(
        count_c={
            "element": "C", "candidate_count": 25,
            "experimental_distinct_signals": 22, "match": True,
            "note": "C observation",
        },
        count_h={
            "element": "H", "candidate_count": 24,
            "experimental_total_nH": 24, "match": True,
            "note": "H observation",
        },
        spectral={
            "similarity": 0.9655, "c_sim": 0.9531, "h_sim": 0.9872,
            "n_pred_c": 22, "n_exp_c": 22, "n_pred_h": 13, "n_exp_h": 13,
            "c_data_absent": False, "ok": True, "note": "spectral observation",
        },
        fg_scan={
            "groups_present": ["hydroxyl", "aromatic_ring"],
            "n_groups": 2, "note": "group observation",
        },
    )
    decision = RouteDecision(
        route="template", n_positive=3, confidence=0.968,
        template_id="T3", template_variant_id="T3-en", template_language="en",
        template_variant_digest="0123456789abcdef",
        c_available=True, h_available=True,
        c_match=True, h_match=True, sim=0.9655,
    )
    return results, decision


def test_router_template_preserves_exact_observation_values() -> None:
    results, decision = fixture_results()
    router = object.__new__(RuleBasedRouter)
    think = router.fill_template("T3-en", results)
    key = router._build_key_evidence(results, decision)
    assert "similarity=0.9655" in think
    assert "c_sim=0.9531" in think and "h_sim=0.9872" in think
    assert "similarity=0.966" not in think
    assert key.startswith("spectral_match similarity=0.9655")


def test_template_transcript_matches_model_turn_schema() -> None:
    results, decision = fixture_results()
    final = (
        '<think>spectral_match returned similarity=0.9655. Therefore final_answer = Yes.</think>\n'
        '<answer>{"final_answer":"Yes","confidence":0.97,'
        '"key_evidence":"spectral_match similarity=0.9655."}</answer>'
    )
    router_result = type("RouterResult", (), {
        "tool_results": results,
        "decision": decision,
    })()
    transcript = build_template_transcript(router_result, final)

    assert len(transcript) == 5
    assert [turn["action"].get("name", turn["action"]["type"]) for turn in transcript] == [
        "count_atoms", "count_atoms", "spectral_match", "functional_group_scan", "answer"
    ]
    for index, turn in enumerate(transcript[:4], 1):
        assert turn["turn"] == index
        assert turn["action"]["type"] == "tool"
        assert isinstance(json.loads(turn["tool_observation"]), dict)
        assert turn["tool_metrics"]["ok"] is True
        assert "<think>" in turn["assistant"] and "<tool_call>" in turn["assistant"]
    assert json.loads(transcript[0]["tool_observation"]) == results.count_c
    assert json.loads(transcript[1]["tool_observation"]) == results.count_h
    assert json.loads(transcript[2]["tool_observation"]) == results.spectral
    assert json.loads(transcript[3]["tool_observation"]) == results.fg_scan
    assert transcript[-1] == {
        "turn": 5,
        "assistant": final,
        "action": {
            "type": "answer",
            "answer": '{"final_answer":"Yes","confidence":0.97,'
                      '"key_evidence":"spectral_match similarity=0.9655."}',
        },
    }


def test_template_transcript_narration_follows_selected_language() -> None:
    results, decision = fixture_results()
    decision.template_language = "zh"
    final = (
        '<think>现有判据均通过。因此 final_answer = Yes。</think>\n'
        '<answer>{"final_answer":"Yes","confidence":0.97,'
        '"key_evidence":"spectral_match similarity=0.9655。"}</answer>'
    )
    router_result = type("RouterResult", (), {
        "tool_results": results,
        "decision": decision,
    })()
    transcript = build_template_transcript(router_result, final)
    assert "调用 count_atoms(C) 获取确定性路由证据" in transcript[0]["assistant"]
    assert "已记录上一工具的真实返回" in transcript[1]["assistant"]
    assert transcript[-1]["assistant"] == final


def test_shadow_veto_keeps_assignment_out_of_public_yes_transcript() -> None:
    results, decision = fixture_results()
    results.peak_c = {"contract_version": 2, "ok": True, "n_large_residual": 1}
    results.peak_h = {"contract_version": 2, "ok": True, "n_large_residual": 0}
    decision.assignment_vetoed = True
    decision.assignment_shadow_mode = True
    router_result = type("RouterResult", (), {"tool_results": results, "decision": decision})()
    final = '<think>Base criteria pass.</think><answer>{"final_answer":"Yes"}</answer>'
    transcript = build_template_transcript(router_result, final)
    assert all((t.get("action") or {}).get("name") != "peak_assignment" for t in transcript)
