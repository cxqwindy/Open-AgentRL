#!/usr/bin/env python3
"""M2 unit test: spectral_match / check_substructure / functional_group_scan.

Uses the EXISTING prediction cache (recipe/nmr_matching/prediction_cache.json) so it runs with no
network. Validates:
  * spectral_match returns a similarity in [0,1] and is higher for a matched positive than for a
    mismatched candidate on the same spectrum.
  * check_substructure resolves named groups (ester/methoxy/tms) and raw SMARTS correctly.
  * functional_group_scan reports plausible groups.

Run:  python recipe/nmr_matching_agent/test_tools_m2.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recipe.nmr_matching_agent.nmr_matching_runtime import NmrMatchingRuntime  # noqa: E402

CACHE_PATH = REPO_ROOT / "recipe" / "nmr_matching" / "prediction_cache.json"


def _load_cache():
    with open(CACHE_PATH) as f:
        return json.load(f)


def _run(runtime, name, args):
    call = {"function": {"name": name, "arguments": args}}
    res = asyncio.run(runtime.run_tool_call_async(call))
    return json.loads(res["content"]), res["ok"]


def test_check_substructure():
    # candidate with an ester + aromatic ring + ethoxy
    cand = "CCOC(=O)c1ccccc1"
    rt = NmrMatchingRuntime({"candidate_smiles": cand, "c_nmr_peaks": [{"delta (ppm)": 1.0}], "h_nmr_peaks": []})
    obs, ok = _run(rt, "check_substructure", {"pattern": "ester"})
    print("[substruct ester]", obs)
    assert ok and obs["present_in_candidate"] is True, obs

    obs, _ = _run(rt, "check_substructure", {"pattern": "ethoxy"})
    print("[substruct ethoxy]", obs)
    assert obs["present_in_candidate"] is True, obs

    obs, _ = _run(rt, "check_substructure", {"pattern": "tms"})
    print("[substruct tms]", obs)
    assert obs["present_in_candidate"] is False, obs

    # raw SMARTS also works
    obs, _ = _run(rt, "check_substructure", {"pattern": "c1ccccc1"})
    print("[substruct raw aromatic]", obs)
    assert obs["present_in_candidate"] is True, obs


def test_functional_group_scan():
    cand = "CCOC(=O)c1ccccc1"  # ester + aromatic ring
    rt = NmrMatchingRuntime({"candidate_smiles": cand, "c_nmr_peaks": [{"delta (ppm)": 1.0}], "h_nmr_peaks": []})
    obs, ok = _run(rt, "functional_group_scan", {})
    print("[fg scan]", obs)
    assert ok, obs
    assert "aromatic_ring" in obs["groups_present"], obs
    assert "ester" in obs["groups_present"], obs


def test_spectral_match_positive_vs_mismatch():
    cache = _load_cache()
    # Pick a cached candidate that has a predicted spectrum; use ITS OWN predicted peaks as a
    # pseudo-experimental spectrum -> self-match must score high. Then score a DIFFERENT candidate
    # against the same spectrum -> should be lower.
    items = [(k, v) for k, v in cache.items() if v.get("ok") and v.get("c_peaks")]
    assert items, "cache has no usable predictions"
    (smi_a, pred_a) = items[0]
    (smi_b, pred_b) = items[10]

    # experimental spectrum = A's predicted peaks, in the dict form the scorer expects
    exp_c = [{"delta (ppm)": v} for v in pred_a["c_peaks"]]
    exp_h = [{"delta": v, "centroid": v} for v in pred_a["h_peaks"]]

    rt_self = NmrMatchingRuntime(
        {"candidate_smiles": smi_a, "c_nmr_peaks": exp_c, "h_nmr_peaks": exp_h},
        prediction_cache=cache,
    )
    obs_self, ok = _run(rt_self, "spectral_match", {})
    print("[spectral self-match]", {k: obs_self[k] for k in ("similarity", "c_sim", "h_sim", "ok")})
    assert ok and obs_self["ok"], obs_self
    assert 0.0 <= obs_self["similarity"] <= 1.0, obs_self
    assert obs_self["similarity"] > 0.5, f"self-match should be high: {obs_self}"

    rt_mis = NmrMatchingRuntime(
        {"candidate_smiles": smi_b, "c_nmr_peaks": exp_c, "h_nmr_peaks": exp_h},
        prediction_cache=cache,
    )
    obs_mis, _ = _run(rt_mis, "spectral_match", {})
    print("[spectral mismatch]", {k: obs_mis[k] for k in ("similarity", "c_sim", "h_sim", "ok")})
    assert obs_mis["similarity"] <= obs_self["similarity"], (obs_self, obs_mis)


def test_spectral_match_cache_miss():
    # an unseen candidate with no cache entry and no remote -> ok=False, similarity 0
    rt = NmrMatchingRuntime(
        {"candidate_smiles": "C#CC#CC#CC#CC#C", "c_nmr_peaks": [{"delta (ppm)": 70.0}], "h_nmr_peaks": []},
        prediction_cache={}, allow_remote_predict=False,
    )
    obs, ok = _run(rt, "spectral_match", {})
    print("[spectral cache-miss]", obs)
    assert ok and obs["ok"] is False and obs["similarity"] == 0.0, obs


if __name__ == "__main__":
    test_check_substructure()
    test_functional_group_scan()
    test_spectral_match_positive_vs_mismatch()
    test_spectral_match_cache_miss()
    print("\nAll M2 tool tests passed.")


def test_peak_assignment_hungarian_one_to_one_and_rectangular():
    pred = {"CC": {"ok": True, "c_peaks": [10.0, 30.0], "h_peaks": [1.0],
                   "C_groups": [[1], [2]], "H_groups": [[3, 4, 5, 6, 7, 8]]}}
    rt = NmrMatchingRuntime({"candidate_smiles": "CC", "c_nmr_peaks": [
        {"delta (ppm)": 11.0}, {"delta (ppm)": 12.0}, {"delta (ppm)": 29.0}],
        "h_nmr_peaks": [{"delta": 1.1, "nH": 6}]}, prediction_cache=pred)
    obs, _ = rt._handle_peak_assignment({"nucleus": "C"})
    assert obs["contract_version"] == 2 and obs["algorithm"] == "hungarian_max_in_tolerance_then_min_abs"
    used = [x["pred_index"] for x in obs["assignments"] if x["pred_index"] is not None]
    assert len(used) == len(set(used)) == 2
    assert obs["n_exp"] == 3 and obs["n_pred"] == 2 and obs["n_unmatched_exp"] == 1
    assert set(obs["residual_stats"]) == {"n", "mae", "rmse", "max_abs"}


def test_peak_assignment_reports_unmatched_pred_and_h_groups():
    pred = {"CC": {"ok": True, "c_peaks": [10.0, 30.0], "h_peaks": [1.0, 4.0],
                   "C_groups": [[1], [2]], "H_groups": [[3, 4, 5], [6, 7, 8]]}}
    rt = NmrMatchingRuntime({"candidate_smiles": "CC", "c_nmr_peaks": [{"delta (ppm)": 10.1}],
        "h_nmr_peaks": [{"centroid": 1.1, "nH": 6}]}, prediction_cache=pred)
    c, _ = rt._handle_peak_assignment({"nucleus": "C"})
    h, _ = rt._handle_peak_assignment({"nucleus": "H"})
    assert c["n_unmatched_pred"] == 1 and c["unmatched_pred_ppm"] == [30.0]
    assert h["predicted_group_sizes"] == [3, 3] and h["experimental_integrations"] == [6]


def test_peak_assignment_maximizes_in_tolerance_cardinality_before_residual():
    pred = {"CC": {"ok": True, "c_peaks": [6.0, 7.0], "h_peaks": [],
                   "C_groups": [[1], [2]], "H_groups": []}}
    rt = NmrMatchingRuntime({"candidate_smiles": "CC", "c_nmr_peaks": [
        {"delta (ppm)": 0.0}, {"delta (ppm)": 1.0}], "h_nmr_peaks": []}, prediction_cache=pred)
    obs, _ = rt._handle_peak_assignment({"nucleus": "C"})
    assert obs["n_matched"] == 1
    assert any(x["exp_ppm"] == 1.0 and x["pred_ppm"] == 6.0 and x["matched"] for x in obs["assignments"])


def test_peak_assignment_no_predicted_peaks_has_complete_v2_shape():
    pred = {"CC": {"ok": True, "c_peaks": [], "h_peaks": [], "C_groups": [], "H_groups": []}}
    rt = NmrMatchingRuntime({"candidate_smiles": "CC", "c_nmr_peaks": [{"delta (ppm)": 10.0}], "h_nmr_peaks": []}, prediction_cache=pred)
    obs, _ = rt._handle_peak_assignment({"nucleus": "C"})
    required = {"contract_version", "algorithm", "nucleus", "tol_ppm", "n_exp", "n_pred", "n_assigned", "n_matched", "n_unmatched", "n_unmatched_exp", "n_unmatched_pred", "coverage", "exp_coverage", "pred_coverage", "symmetric_coverage", "assignments", "unmatched_exp_ppm", "unmatched_pred_ppm", "n_large_residual", "large_residual_threshold_ppm", "large_residuals", "residual_stats", "ok", "status"}
    assert required <= set(obs)
    assert obs["status"] == "no_predicted_peaks" and obs["n_unmatched_exp"] == 1


def test_peak_assignment_no_experimental_data_has_complete_v2_shape():
    pred = {"CC": {"ok": True, "c_peaks": [10.0], "h_peaks": [], "C_groups": [[1]], "H_groups": []}}
    rt = NmrMatchingRuntime({"candidate_smiles": "CC", "c_nmr_peaks": [], "h_nmr_peaks": []}, prediction_cache=pred)
    obs, _ = rt._handle_peak_assignment({"nucleus": "C"})
    assert obs["status"] == "no_experimental_data" and obs["tol_ppm"] == 5.0
    assert obs["n_assigned"] == 0 and obs["large_residuals"] == []


def test_peak_assignment_prediction_unavailable_has_complete_v2_shape():
    rt = NmrMatchingRuntime({"candidate_smiles": "CC", "c_nmr_peaks": [{"delta (ppm)": 10.0}], "h_nmr_peaks": []}, prediction_cache={}, allow_remote_predict=False)
    obs, _ = rt._handle_peak_assignment({"nucleus": "C"})
    assert obs["status"] == "prediction_unavailable" and obs["tol_ppm"] == 5.0
    assert obs["n_unmatched_exp"] == 1 and len(obs["assignments"]) == 1
    assert obs["large_residuals"] == [] and set(obs["residual_stats"]) == {"n", "mae", "rmse", "max_abs"}
