from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from recipe.nmr_matching_agent.evidence_reconstruction import (
    EvidenceReconstructionError,
    reconstruct_model_final_turn,
)
from recipe.nmr_matching_agent import standalone_infer_matching as infer


def tool_row(turn: int, name: str, observation: dict) -> dict:
    return {
        "turn": turn,
        "assistant": "<think>model text must not be reused</think>",
        "action": {"type": "tool", "name": name, "arguments": {}},
        "tool_observation": json.dumps(observation),
    }


def answer_object(tagged: str) -> dict:
    body = infer.ANSWER_RE.search(tagged)
    assert body
    return json.loads(body.group(1))


def test_reconstructs_from_current_spectral_observation() -> None:
    transcript = [
        tool_row(1, "count_atoms", {
            "element": "H", "candidate_count": 30,
            "experimental_total_nH": 30, "match": True,
        }),
        tool_row(2, "spectral_match", {
            "similarity": 0.4089, "c_sim": 0.4298, "h_sim": 0.3671,
            "n_pred_c": 21, "n_exp_c": 10, "n_pred_h": 10, "n_exp_h": 5,
            "c_data_absent": False, "ok": True,
        }),
    ]
    raw = json.dumps({
        "final_answer": "No", "confidence": 1.0,
        "key_evidence": "hallucinated 18-proton tert-butyl argument",
        "extra": {"preserve": True},
    })
    tagged, diagnostics = reconstruct_model_final_turn(raw, transcript)
    parsed = answer_object(tagged)

    assert diagnostics == []
    assert parsed["final_answer"] == "No"
    assert parsed["confidence"] == 1.0
    assert parsed["extra"] == {"preserve": True}
    assert parsed["key_evidence"] == (
        "spectral_match shows low chemical-shift agreement: similarity=0.4089; "
        "predicted/experimental peak counts are 13C 21/10 and 1H 10/5; "
        "the weakest component is 1H (h_sim=0.3671 versus c_sim=0.4298)."
    )
    assert "hallucinated" not in tagged and "18-proton" not in tagged
    assert "candidate_count=30" not in tagged
    assert "The decisive negative evidence is:" in tagged


def test_no_with_equal_peak_counts_highlights_lowest_component() -> None:
    transcript = [
        tool_row(1, "count_atoms", {
            "element": "H", "candidate_count": 7,
            "experimental_total_nH": 7, "match": True,
            "exact_match": True, "difference": 0, "tolerance": 2,
        }),
        tool_row(2, "spectral_match", {
            "similarity": 0.5692, "c_sim": 0.7086, "h_sim": 0.3834,
            "n_pred_c": 4, "n_exp_c": 4, "n_pred_h": 3, "n_exp_h": 3,
            "c_data_absent": False, "ok": True,
        }),
    ]
    tagged, diagnostics = reconstruct_model_final_turn(
        '{"final_answer":"No","confidence":0.9,"key_evidence":"tool dump"}',
        transcript,
    )
    parsed = answer_object(tagged)
    expected = (
        "Peak counts agree, but spectral_match shows low chemical-shift agreement: "
        "similarity=0.5692; the weakest component is 1H "
        "(h_sim=0.3834 versus c_sim=0.7086)."
    )
    assert diagnostics == []
    assert parsed["key_evidence"] == expected
    assert f"The decisive negative evidence is: {expected}" in tagged
    assert "candidate_count=7" not in tagged


def test_count_contradiction_has_priority_even_for_yes() -> None:
    transcript = [
        tool_row(1, "count_atoms", {
            "element": "C", "candidate_count": 8,
            "experimental_distinct_signals": 10, "match": False,
        }),
        tool_row(2, "spectral_match", {
            "similarity": 0.95, "c_sim": 0.96, "h_sim": 0.94,
            "n_pred_c": 8, "n_exp_c": 10, "n_pred_h": 5, "n_exp_h": 5,
            "c_data_absent": False, "ok": True,
        }),
    ]
    tagged, _ = reconstruct_model_final_turn(
        '{"final_answer":"Yes","confidence":0.7,"key_evidence":"wrong"}', transcript
    )
    parsed = answer_object(tagged)
    assert parsed["final_answer"] == "Yes" and parsed["confidence"] == 0.7
    assert "candidate_count=8" in parsed["key_evidence"]
    assert "experimental_signals=10" in parsed["key_evidence"]
    assert "match=false" in parsed["key_evidence"]


def test_peak_assignment_contradiction_has_priority() -> None:
    transcript = [
        tool_row(1, "spectral_match", {
            "similarity": 0.7, "c_sim": 0.8, "h_sim": 0.6,
            "n_pred_c": 4, "n_exp_c": 5, "n_pred_h": 3, "n_exp_h": 3,
            "c_data_absent": False, "ok": True,
        }),
        tool_row(2, "peak_assignment", {
            "nucleus": "C", "tol_ppm": 5.0, "n_exp": 5,
            "n_matched": 3, "n_unmatched": 2, "coverage": 0.6,
            "assignments": [], "unmatched_exp_ppm": [12.3, 180.1], "ok": True,
        }),
    ]
    tagged, _ = reconstruct_model_final_turn(
        '{"final_answer":"No","confidence":0.8,"key_evidence":"wrong"}', transcript
    )
    key = answer_object(tagged)["key_evidence"]
    assert "n_unmatched=2" in key and "coverage=0.6" in key
    assert "[12.3, 180.1]" in key


@pytest.mark.parametrize(
    ("answer", "transcript", "code"),
    [
        ("not json", [tool_row(1, "count_atoms", {"element": "C"})],
         "invalid_final_answer_json"),
        ('{"confidence":0.5}', [tool_row(1, "count_atoms", {"element": "C"})],
         "invalid_final_answer_value"),
        ('{"final_answer":"No"}', [], "no_parseable_tool_observation"),
    ],
)
def test_reconstruction_fails_closed(answer: str, transcript: list, code: str) -> None:
    with pytest.raises(EvidenceReconstructionError) as caught:
        reconstruct_model_final_turn(answer, transcript)
    assert caught.value.code == code


def test_run_agent_rebuilds_top_level_answer_and_keeps_raw_transcript(monkeypatch) -> None:
    class FakeTool:
        async def create(self, **kwargs) -> None:
            return None

        async def release(self, instance_id: str) -> None:
            return None

        async def execute(self, instance_id: str, arguments: dict):
            obs = {
                "similarity": 0.4089, "c_sim": 0.4298, "h_sim": 0.3671,
                "n_pred_c": 21, "n_exp_c": 10, "n_pred_h": 10, "n_exp_h": 5,
                "c_data_absent": False, "ok": True,
            }
            return SimpleNamespace(text=json.dumps(obs)), 0.0, {}

    generations = iter([
        '<tool_call>{"name":"spectral_match","arguments":{}}</tool_call>',
        '<answer>{"final_answer":"No","confidence":1.0,'
        '"key_evidence":"hallucinated model evidence"}</answer>',
    ])
    monkeypatch.setattr(infer, "load_tools", lambda: {"spectral_match": FakeTool()})
    monkeypatch.setattr(infer, "generate_once", lambda *args, **kwargs: next(generations))

    result = asyncio.run(infer.run_agent(None, None, {
        "candidate_smiles": "C", "c_nmr_peaks": [], "h_nmr_peaks": [],
    }))
    assert result["fail_reason"] is None
    assert result["evidence_rebuilt"] is True
    assert result["evidence_rebuild_diagnostics"] == []
    assert result["tool_sequence"] == ["spectral_match", "answer"]
    assert "hallucinated model evidence" in result["transcript"][-1]["assistant"]
    assert "hallucinated model evidence" not in result["answer"]
    assert answer_object(result["answer"])["key_evidence"].startswith(
        "spectral_match shows low chemical-shift agreement: similarity=0.4089"
    )


def test_count_contradiction_beats_earlier_assignment_contradiction() -> None:
    transcript = [
        tool_row(1, "peak_assignment", {
            "nucleus": "C", "ok": True, "n_unmatched_exp": 1,
            "coverage": 0.8, "unmatched_exp_ppm": [180.1],
        }),
        tool_row(2, "count_atoms", {
            "element": "C", "candidate_count": 8,
            "experimental_distinct_signals": 10, "match": False,
        }),
    ]
    tagged, _ = reconstruct_model_final_turn('{"final_answer":"No"}', transcript)
    key = answer_object(tagged)["key_evidence"]
    assert key.startswith("Decisive carbon-count contradiction:")
    assert "candidate_count=8" in key and "experimental_signals=10" in key


def test_legacy_assignment_unmatched_list_is_a_direct_contradiction() -> None:
    transcript = [tool_row(1, "peak_assignment", {
        "nucleus": "H", "ok": "true", "coverage": "0.5",
        "unmatched_exp": [1.2, 4.8],
    })]
    tagged, _ = reconstruct_model_final_turn('{"final_answer":"No"}', transcript)
    key = answer_object(tagged)["key_evidence"]
    assert key.startswith("Direct H peak-assignment contradiction:")
    assert "n_unmatched=2" in key and "unmatched_exp_ppm=[1.2, 4.8]" in key


def test_numeric_strings_are_normalized_for_equal_peak_counts() -> None:
    transcript = [tool_row(1, "spectral_match", {
        "similarity": "0.5692", "c_sim": "0.7086", "h_sim": "0.3834",
        "n_pred_c": "4", "n_exp_c": 4, "n_pred_h": "3.0", "n_exp_h": 3,
        "c_data_absent": "false", "ok": "true",
    })]
    tagged, _ = reconstruct_model_final_turn('{"final_answer":"No"}', transcript)
    key = answer_object(tagged)["key_evidence"]
    assert key.startswith("Peak counts agree, but")
    assert '"0.5692"' not in key and "similarity=0.5692" in key
    assert "1H (h_sim=0.3834 versus c_sim=0.7086)" in key


def test_h_only_negative_uses_available_count_wording() -> None:
    transcript = [tool_row(1, "spectral_match", {
        "similarity": 0.4, "c_sim": 0.0, "h_sim": 0.4,
        "n_pred_c": 0, "n_exp_c": 0, "n_pred_h": 3, "n_exp_h": 3,
        "c_data_absent": "true", "ok": True,
    })]
    tagged, _ = reconstruct_model_final_turn('{"final_answer":"No"}', transcript)
    key = answer_object(tagged)["key_evidence"]
    assert key.startswith("The available 1H peak count agrees, but")
    assert "1H h_sim=0.4" in key and "experimental 13C data are absent" in key


@pytest.mark.parametrize("similarity", [None, 0.95, "not-a-number", float("inf")])
def test_no_without_grounded_negative_evidence_fails_closed(similarity) -> None:
    observation = {
        "similarity": similarity, "c_sim": 0.95, "h_sim": 0.95,
        "n_pred_c": 4, "n_exp_c": 4, "n_pred_h": 3, "n_exp_h": 3,
        "c_data_absent": False, "ok": True,
    }
    transcript = [{
        "turn": 1,
        "action": {"type": "tool", "name": "spectral_match", "arguments": {}},
        "tool_observation": observation,
    }]
    with pytest.raises(EvidenceReconstructionError) as caught:
        reconstruct_model_final_turn('{"final_answer":"No"}', transcript)
    assert caught.value.code == "no_grounded_negative_evidence"
