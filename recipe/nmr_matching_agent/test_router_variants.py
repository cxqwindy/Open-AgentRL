from __future__ import annotations

import json

import pytest

from recipe.nmr_matching_agent.cot_to_trajectory_v5 import _validate_think_number_provenance
from recipe.nmr_matching_agent.rule_based_router import RuleBasedRouter, ToolResults


def router(record: dict | None = None, *, seed: int = 0, languages=None) -> RuleBasedRouter:
    obj = object.__new__(RuleBasedRouter)
    obj.record = record or {
        "candidate_smiles": "CCO",
        "c_nmr_peaks": [{"delta (ppm)": 10.0}],
        "h_nmr_peaks": [{"delta": 1.0, "nH": 3}],
        "nmr_solvent": "CDCl3",
        "nmr_frequency": "400 MHz",
    }
    obj.sim_threshold_both = 0.85
    obj.sim_threshold_strict = 0.92
    obj.h_tol = 2
    obj.weights = {"sim": 0.40, "c_sim": 0.30, "h_sim": 0.30}
    obj.template_variant_seed = seed
    obj.template_languages = list(languages or ["en", "zh", "ja"])
    return obj


def results(
    *,
    c_count=10,
    c_exp=10,
    c_tool_match=True,
    h_count=12,
    h_exp=12,
    h_tool_match=True,
    sim=0.95,
    spectral_ok=True,
    c_absent=False,
) -> ToolResults:
    return ToolResults(
        count_c={
            "element": "C", "candidate_count": c_count,
            "experimental_distinct_signals": c_exp, "match": c_tool_match,
        },
        count_h={
            "element": "H", "candidate_count": h_count,
            "experimental_total_nH": h_exp, "match": h_tool_match,
        },
        spectral={
            "similarity": sim, "c_sim": 0.94, "h_sim": 0.96,
            "n_pred_c": c_count, "n_exp_c": c_exp,
            "n_pred_h": 5, "n_exp_h": 5,
            "c_data_absent": c_absent, "ok": spectral_ok,
        },
        fg_scan={"groups_present": ["alcohol"], "n_groups": 1},
    )


@pytest.mark.parametrize(
    ("tool_results", "expected_route", "expected_template", "c_available", "h_available"),
    [
        (results(sim=0.85), "template", "T1", True, True),
        (results(c_exp=0, c_absent=True, sim=0.91), "model", "", False, True),
        (results(c_exp=0, c_absent=True, sim=0.92), "template", "T4", False, True),
        (results(h_exp=None, h_tool_match=None, sim=0.91), "model", "", True, False),
        (results(h_exp=None, h_tool_match=None, sim=0.92), "template", "T5", True, False),
        (results(c_exp=0, c_absent=True, h_exp=None, h_tool_match=None, sim=0.99),
         "model", "", False, False),
        (results(c_count=8, c_exp=10, c_tool_match=False, sim=0.99),
         "model", "", True, True),
        (results(h_count=8, h_exp=12, h_tool_match=False, sim=0.99),
         "model", "", True, True),
        (results(h_count=10, h_exp=12, h_tool_match=False, sim=0.85),
         "template", "T2", True, True),
        (results(c_count=12, c_exp=10, sim=0.85), "template", "T3", True, True),
        (results(sim=0.99, spectral_ok=False), "model", "", True, True),
    ],
)
def test_available_pass_mismatch_routing(
    tool_results: ToolResults,
    expected_route: str,
    expected_template: str,
    c_available: bool,
    h_available: bool,
) -> None:
    decision = router().evaluate_criteria(tool_results)
    assert decision.route == expected_route
    assert decision.template_id == expected_template
    assert decision.c_available is c_available
    assert decision.h_available is h_available
    if expected_route == "template":
        assert decision.template_variant_id.startswith(expected_template + "-")
        assert decision.template_language in {"en", "zh", "ja"}
        assert len(decision.template_variant_digest) == 16
    else:
        assert decision.template_variant_id == ""
        assert decision.template_language == ""


def test_template_variant_selection_is_stable_and_seeded() -> None:
    record = {
        "candidate_smiles": "CCO", "c_nmr_peaks": [{"delta (ppm)": 20.0}],
        "h_nmr_peaks": [{"delta": 1.2, "nH": 3}], "nmr_solvent": "CDCl3",
    }
    first = router(record, seed=7).evaluate_criteria(results(sim=0.85))
    second = router(dict(reversed(list(record.items()))), seed=7).evaluate_criteria(results(sim=0.85))
    assert first.template_variant_id == second.template_variant_id
    assert first.template_variant_digest == second.template_variant_digest

    selected = {
        router(record, seed=seed).evaluate_criteria(results(sim=0.85)).template_variant_id
        for seed in range(20)
    }
    assert len(selected) >= 2
    english = router(record, seed=99, languages=["en"]).evaluate_criteria(results(sim=0.85))
    assert english.template_variant_id == "T1-en"
    assert english.template_language == "en"


@pytest.mark.parametrize(
    ("tool_results", "template_id"),
    [
        (results(sim=0.85), "T1"),
        (results(h_count=10, h_exp=12, h_tool_match=False), "T2"),
        (results(c_count=12, c_exp=10), "T3"),
        (results(c_exp=0, c_absent=True), "T4"),
        (results(h_exp=None, h_tool_match=None), "T5"),
    ],
)
@pytest.mark.parametrize("language", ["en", "zh", "ja"])
def test_every_variant_is_complete_and_tool_grounded(
    tool_results: ToolResults, template_id: str, language: str
) -> None:
    r = router(seed=3, languages=[language])
    decision = r.evaluate_criteria(tool_results)
    assert decision.route == "template" and decision.template_id == template_id
    think = r.fill_template(decision.template_variant_id, tool_results)
    key = r._build_key_evidence(tool_results, decision)
    assert "{" not in think and "}" not in think
    assert "final_answer = Yes" in think
    assert decision.template_variant_id == f"{template_id}-{language}"

    messages = []
    for name, obs in [
        ("count_atoms", tool_results.count_c),
        ("count_atoms", tool_results.count_h),
        ("spectral_match", tool_results.spectral),
        ("functional_group_scan", tool_results.fg_scan),
    ]:
        messages.append({"role": "tool", "name": name, "content": json.dumps(obs)})
    messages.append({"role": "assistant", "content": f"<think>{think} {key}</think>"})
    assert _validate_think_number_provenance(messages) == []


def test_runtime_h_tolerance_exposes_exact_and_difference_fields() -> None:
    from recipe.nmr_matching_agent.nmr_matching_runtime import NmrMatchingRuntime

    tolerant = NmrMatchingRuntime({
        "candidate_smiles": "CCO",
        "c_nmr_peaks": [{"delta (ppm)": 58.0}, {"delta (ppm)": 18.0}],
        "h_nmr_peaks": [{"delta": 3.6, "nH": 2}, {"delta": 1.2, "nH": 3}],
    }, h_tolerance=2)
    obs, _ = tolerant._handle_count_atoms({"element": "H"})
    assert obs == {
        "element": "H", "candidate_count": 6, "experimental_total_nH": 5,
        "match": True, "exact_match": False, "difference": 1, "tolerance": 2,
        "note": "candidate has 6 H atoms; 1H integrations sum to 5 protons. Difference 1 is within H-integration tolerance ±2 (e.g. unreported exchangeable H).",
    }

    strict = NmrMatchingRuntime({
        "candidate_smiles": "CCO",
        "h_nmr_peaks": [{"delta": 1.2, "nH": 3}],
    }, h_tolerance=2)
    obs, _ = strict._handle_count_atoms({"element": "H"})
    assert obs["difference"] == 3 and obs["match"] is False and obs["exact_match"] is False


def test_assignment_veto_shadow_and_enforce() -> None:
    r = router()
    r.assignment_veto = {"enabled": True, "shadow_mode": True,
                         "c_max_abs_ppm": 8.0, "h_max_abs_ppm": 0.35,
                         "fail_closed": True}
    tr = results(sim=0.90)
    tr.peak_c = {"ok": True, "residual_stats": {"max_abs": 9.0}}
    tr.peak_h = {"ok": True, "residual_stats": {"max_abs": 0.1}}
    decision = r.evaluate_criteria(tr)
    decision = r.apply_assignment_veto(tr, decision)
    assert decision.would_template and decision.assignment_vetoed
    assert decision.route == "template" and decision.assignment_shadow_mode
    assert decision.assignment_veto_reasons == ["C:max_abs=9.0>8.0"]

    r.assignment_veto["shadow_mode"] = False
    decision = r.apply_assignment_veto(tr, r.evaluate_criteria(tr))
    assert decision.route == "model" and decision.assignment_vetoed
    assert decision.template_id == ""
