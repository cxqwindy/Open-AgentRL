#!/usr/bin/env python3
"""M0 unit test: count_atoms tool end-to-end through the runtime + verl wrapper.

Covers three known candidates (no GPU / no network):
  1. insert_ch2 negative  -- candidate has MORE carbons than the 13C signals (a real v4 case).
  2. same_formula isomer  -- same atom count as GT, so count_atoms is NON-discriminative (match=True).
  3. positive             -- candidate == GT, counts consistent.

Also drives the NmrMatchingTool.create/execute/release lifecycle with a synthetic raw_prompt to
prove the wrapper reads candidate+peaks from the user JSON and returns the runtime observation.

Run:  python recipe/nmr_matching_agent/test_tools_m0.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recipe.nmr_matching_agent.nmr_matching_runtime import NmrMatchingRuntime, molecule_formula  # noqa: E402


def _run(runtime: NmrMatchingRuntime, element: str) -> dict:
    call = {"function": {"name": "count_atoms", "arguments": {"element": element}}}
    res = asyncio.run(runtime.run_tool_call_async(call))
    assert res["ok"], res
    return json.loads(res["content"])


def test_insert_ch2_negative():
    # Real v4 case (first cot_traces_v4 line). NOTE: Gemini's CoT claimed "at least 23 carbons",
    # but the true RDKit count is 22 -- a factual error the deterministic tool corrects (this is
    # precisely the value the tool route provides over in-head estimation). Spectrum has 18 signals.
    cand = "CCCOC(=O)C(CC(Nc1cccc(C(=O)OCC)c1)N1CCOC1=O)C(=O)OCC"
    c_peaks = [{"delta (ppm)": v} for v in
               [169.2, 169.0, 166.7, 157.9, 144.6, 131.7, 129.8, 120.2, 117.2, 115.2,
                62.4, 62.12, 61.1, 49.0, 39.3, 32.1, 14.4, 14.1]]  # 18 signals
    rt = NmrMatchingRuntime({"candidate_smiles": cand, "c_nmr_peaks": c_peaks, "h_nmr_peaks": []})
    obs = _run(rt, "C")
    print("[insert_ch2] C:", obs)
    assert obs["candidate_count"] == 22, obs
    assert obs["experimental_distinct_signals"] == 18, obs
    # 22 >= 18 so count_atoms alone does NOT flag it (this neg needs spectral_match / substructure);
    # the point of M0 is that the number is CORRECT, not that it discriminates.
    assert obs["match"] is True, obs


def test_fewer_carbons_contradiction():
    # Synthetic: a tiny candidate against a spectrum with more signals than it has carbons.
    cand = "CCO"  # 2 carbons
    c_peaks = [{"delta (ppm)": v} for v in [70.0, 60.0, 50.0, 40.0]]  # 4 signals > 2 carbons
    rt = NmrMatchingRuntime({"candidate_smiles": cand, "c_nmr_peaks": c_peaks, "h_nmr_peaks": []})
    obs = _run(rt, "C")
    print("[contradiction] C:", obs)
    assert obs["candidate_count"] == 2, obs
    assert obs["match"] is False, obs  # fewer carbons than signals = hard contradiction


def test_hydrogen_count():
    # Ethanol C2H6O: 6 hydrogens. Experimental integrations summing to 6 -> match.
    cand = "CCO"
    h_peaks = [
        {"delta": 3.7, "category": "q", "nH": 2},
        {"delta": 1.2, "category": "t", "nH": 3},
        {"delta": 2.5, "category": "s", "nH": 1},
    ]  # sum nH = 6
    rt = NmrMatchingRuntime({"candidate_smiles": cand, "c_nmr_peaks": [], "h_nmr_peaks": h_peaks})
    obs = _run(rt, "H")
    print("[H count] H:", obs)
    assert obs["candidate_count"] == 6, obs
    assert obs["experimental_total_nH"] == 6, obs
    assert obs["match"] is True, obs


def test_positive_consistent():
    # A real positive from test_1000: candidate == GT.
    cand = "COc1cc(O)c(C)c(C#Cc2cccc(OCc3ccccc3)c2OC)c1C"
    # 22 13C signals in that record; candidate has >=22 carbons -> consistent.
    c_peaks = [{"delta (ppm)": float(i)} for i in range(22)]
    rt = NmrMatchingRuntime({"candidate_smiles": cand, "c_nmr_peaks": c_peaks, "h_nmr_peaks": []})
    obs = _run(rt, "C")
    print("[positive] C:", obs, "formula:", molecule_formula(cand))
    assert obs["candidate_count"] >= obs["experimental_distinct_signals"], obs
    assert obs["match"] is True, obs


def test_wrapper_lifecycle():
    """Drive NmrMatchingTool.create/execute/release with a synthetic raw_prompt (no manager singleton reuse)."""
    from verl.tools.nmr_matching_tools import NmrMatchingSessionManager, NmrMatchingTool
    from verl.tools.schemas import OpenAIFunctionToolSchema

    schema = OpenAIFunctionToolSchema.model_validate({
        "type": "function",
        "function": {
            "name": "count_atoms",
            "description": "count",
            "parameters": {
                "type": "object",
                "properties": {"element": {"type": "string", "enum": ["C", "H"]}},
                "required": ["element"],
            },
        },
    })
    manager = NmrMatchingSessionManager(prediction_cache_path="", allow_remote_predict=False)
    tool = NmrMatchingTool({"session_manager": manager}, schema)

    user_json = json.dumps({
        "candidate_smiles": "CCO",
        "c_nmr_peaks": [{"delta (ppm)": 70.0}, {"delta (ppm)": 60.0}],
        "h_nmr_peaks": [],
    })
    raw_prompt = [{"role": "system", "content": "..."}, {"role": "user", "content": user_json}]

    async def _drive():
        instance_id, resp = await tool.create(create_kwargs={"raw_prompt": raw_prompt, "episode_id": "ep1"})
        assert "ready" in resp.text, resp.text
        tool_resp, reward, metrics = await tool.execute(instance_id, {"element": "C"})
        obs = json.loads(tool_resp.text)
        assert obs["candidate_count"] == 2, obs
        assert metrics["ok"] is True, metrics
        await tool.release(instance_id)
        return obs

    obs = asyncio.run(_drive())
    print("[wrapper] execute obs:", obs)


if __name__ == "__main__":
    test_insert_ch2_negative()
    test_fewer_carbons_contradiction()
    test_hydrogen_count()
    test_positive_consistent()
    test_wrapper_lifecycle()
    print("\nAll M0 count_atoms tests passed.")
