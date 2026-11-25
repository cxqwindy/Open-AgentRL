"""Deterministic "evidence tool" runtime for the NMR spectrum<->SMILES MATCHING agent.

Unlike the generation-task runtime (`nmr/workflow_v2/nmr_agent_runtime_v3_with_formula.py`,
which searches/repairs/ranks a candidate POOL), the matching task gets the candidate SMILES
GIVEN in the user record. Every tool here VERIFIES one property of that fixed candidate against
the fixed experimental spectrum. There is no pool, no formula-exact repair, no rank-and-select.

Tools (dispatch table `self.handlers`):
  * count_atoms          -- RDKit exact C/H count vs experimental signal count       [M0]
  * spectral_match       -- forward-predict candidate spectrum, score vs experimental [M2]
  * check_substructure   -- SMARTS substructure presence in candidate                 [M2]
  * functional_group_scan-- which known functional groups are in the candidate        [M2]
  * peak_assignment      -- one-to-one Hungarian C/H signal assignment               [M2]

Each handler returns (obs_dict, extra_dict); `run_tool_call_async` wraps that into the
{"content": <json str>, "ok": bool, "extra": {...}} shape the verl tool wrapper expects.

The only potentially-slow tool is `spectral_match` (remote NMR forward prediction). It is served
from an in-memory prediction cache keyed by canonical SMILES (see NmrMatchingRuntime.__init__);
misses fall back to the remote predictor. See recipe README + plan.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from scipy.optimize import linear_sum_assignment

from recipe.nmr_matching_agent.task_input import describe_integrals

try:  # silence RDKit sanitize spam
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
except Exception:
    pass

# The deterministic "brains" live in SpecMaster_jp (spectral scorer + forward predictor) and in
# this recipe dir (named SMARTS). Import lazily/robustly so the M0 count_atoms path works even if
# the SpecMaster tree is unavailable.
_SPECMASTER = "/fs_mol/cxq/SpecMaster_jp/perturbation_generator"
if _SPECMASTER not in sys.path:
    sys.path.insert(0, _SPECMASTER)

try:
    from spectral_scorer import spectral_similarity  # noqa: E402
except Exception:  # pragma: no cover
    spectral_similarity = None  # type: ignore

try:
    from nmr_predictor import _canonical as _predictor_canonical  # noqa: E402
except Exception:  # pragma: no cover
    _predictor_canonical = None  # type: ignore

_RECIPE_DIR = Path(__file__).resolve().parent
if str(_RECIPE_DIR) not in sys.path:
    sys.path.insert(0, str(_RECIPE_DIR))
try:
    from named_smarts import NAMED_SMARTS, resolve_pattern  # noqa: E402
except Exception:  # pragma: no cover
    NAMED_SMARTS, resolve_pattern = {}, None  # type: ignore


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _mol_from_smiles(smi: str) -> Optional[Chem.Mol]:
    if not isinstance(smi, str) or not smi.strip():
        return None
    try:
        m = Chem.MolFromSmiles(smi)
    except Exception:
        return None
    return m


def _count_carbons(mol: Chem.Mol) -> int:
    return sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 6)


def _count_hydrogens(mol: Chem.Mol) -> int:
    """Total H count (implicit + explicit) via AddHs."""
    try:
        molh = Chem.AddHs(mol)
    except Exception:
        return -1
    return sum(1 for a in molh.GetAtoms() if a.GetAtomicNum() == 1)


def _experimental_h_total(h_peaks: List[Dict[str, Any]]) -> Optional[int]:
    """Complete peak-table integral, or None when any peak integral is unknown.

    A partial sum is not returned. ``h_tolerance`` is not applied here.
    """
    described = describe_integrals(h_peaks, None)
    if not described["peak_table_complete"]:
        return None
    return described["peak_table_nH_sum"]


# ---------------------------------------------------------------------------
# runtime
# ---------------------------------------------------------------------------
class NmrMatchingRuntime:
    """One instance per matching episode. Holds the fixed candidate + experimental spectrum."""

    def __init__(
        self,
        record: Dict[str, Any],
        *,
        prediction_cache: Optional[Dict[str, Dict[str, Any]]] = None,
        allow_remote_predict: bool = False,
        sigma_c: float = 5.0,
        sigma_h: float = 0.3,
        h_tolerance: int = 2,
        **_ignored: Any,
    ) -> None:
        self.record = record or {}
        self.candidate_smiles: str = str(self.record.get("candidate_smiles", "") or "").strip()
        self.c_peaks: List[Dict[str, Any]] = self.record.get("c_nmr_peaks") or []
        self.h_peaks: List[Dict[str, Any]] = self.record.get("h_nmr_peaks") or []
        self.prediction_cache = prediction_cache or {}
        self.allow_remote_predict = bool(allow_remote_predict)
        self.sigma_c = float(sigma_c)
        self.sigma_h = float(sigma_h)
        self.h_tolerance = int(h_tolerance)

        self._mol = _mol_from_smiles(self.candidate_smiles)
        self._pred: Optional[Dict[str, Any]] = None  # lazily resolved predicted spectrum

        self.handlers = {
            "count_atoms": self._handle_count_atoms,
            "spectral_match": self._handle_spectral_match,
            "check_substructure": self._handle_check_substructure,
            "functional_group_scan": self._handle_functional_group_scan,
            "peak_assignment": self._handle_peak_assignment,
        }

    # --- dispatch -----------------------------------------------------------
    async def run_tool_call_async(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        fn = tool_call.get("function", {}) if isinstance(tool_call, dict) else {}
        name = fn.get("name")
        args = fn.get("arguments") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {}

        handler = self.handlers.get(name)
        if handler is None:
            return {
                "content": json.dumps({"error": f"unknown tool: {name}"}),
                "ok": False,
                "extra": {},
            }
        if self._mol is None:
            return {
                "content": json.dumps({"error": "candidate SMILES is invalid/unparseable"}),
                "ok": False,
                "extra": {},
            }
        try:
            obs, extra = handler(args)
        except Exception as e:  # pragma: no cover - defensive
            return {
                "content": json.dumps({"error": f"{type(e).__name__}: {e}"}),
                "ok": False,
                "extra": {},
            }
        return {"content": json.dumps(obs, ensure_ascii=False), "ok": True, "extra": extra}

    # --- count_atoms --------------------------------------------------------
    def _handle_count_atoms(self, args: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        element = str(args.get("element", "C")).strip().upper()
        if element not in ("C", "H"):
            return {"error": "element must be 'C' or 'H'"}, {}

        if element == "C":
            cand_count = _count_carbons(self._mol)
            exp_signals = len(self.c_peaks)
            # A candidate can have MORE carbons than distinct 13C signals (symmetry /
            # accidental overlap), but it can NEVER have FEWER carbons than resolved
            # signals. candidate_count < experimental_signals is a hard contradiction.
            match = cand_count >= exp_signals
            note = (
                f"candidate has {cand_count} C atoms; 13C spectrum shows {exp_signals} signals. "
                + (
                    "Consistent (candidate C >= signals; extra carbons may be equivalent by symmetry)."
                    if match
                    else "CONTRADICTION: fewer candidate carbons than resolved 13C signals."
                )
            )
            obs = {
                "element": "C",
                "candidate_count": cand_count,
                "experimental_distinct_signals": exp_signals,
                "match": match,
                "note": note,
            }
            return obs, {"cand_count": cand_count, "exp_signals": exp_signals}

        # element == "H"
        # h_tolerance is unchanged. An incomplete peak table is not a total, and
        # an explicit total that disagrees with a complete peak table is not used.
        cand_count = _count_hydrogens(self._mol)
        integrals = describe_integrals(self.h_peaks, self.record.get("experimental_total_nH"))
        exp_total = integrals["experimental_total_nH"]
        if exp_total is None:
            if integrals["integral_conflict"]:
                note = (
                    f"candidate has {cand_count} H atoms; explicit total "
                    f"{integrals['explicit_experimental_total_nH']} conflicts with complete "
                    f"peak-table sum {integrals['peak_table_nH_sum']}. Neither is used as the total."
                )
            elif integrals["completeness"] == "partial":
                note = (
                    f"candidate has {cand_count} H atoms; only "
                    f"{integrals['known_nH_count']} of {integrals['peak_count']} peak integrals "
                    f"are known (known sum {integrals['known_nH_sum']}). "
                    "That partial sum is not the experimental total."
                )
            else:
                note = (
                    f"candidate has {cand_count} H atoms; 1H integrations unavailable in this spectrum, "
                    "cannot compare counts."
                )
            match = None
            exact_match = None
            difference = None
        else:
            difference = abs(cand_count - exp_total)
            exact_match = difference == 0
            match = difference <= self.h_tolerance
            if exact_match:
                verdict = "Counts match exactly."
            elif match:
                verdict = (
                    f"Difference {difference} is within H-integration tolerance "
                    f"±{self.h_tolerance} (e.g. unreported exchangeable H)."
                )
            else:
                verdict = "MISMATCH beyond H-integration tolerance."
            note = (
                f"candidate has {cand_count} H atoms; 1H integrations sum to {exp_total} protons. "
                + verdict
            )
        obs = {
            "element": "H",
            "candidate_count": cand_count,
            "experimental_total_nH": exp_total,
            "known_nH_sum": integrals["known_nH_sum"],
            "peak_table_nH_sum": integrals["peak_table_nH_sum"],
            "peak_table_complete": integrals["peak_table_complete"],
            "integral_conflict": integrals["integral_conflict"],
            "integral_completeness": integrals["completeness"],
            "match": match,
            "exact_match": exact_match,
            "difference": difference,
            "tolerance": self.h_tolerance,
            "note": note,
        }
        return obs, {
            "cand_count": cand_count,
            "exp_total": exp_total,
            "exact_match": exact_match,
            "difference": difference,
            "tolerance": self.h_tolerance,
        }

    # --- spectral_match -----------------------------------------------------
    def _resolve_prediction(self) -> Optional[Dict[str, Any]]:
        """Return the candidate's predicted spectrum {c_peaks, h_peaks, ok}, from cache or remote."""
        if self._pred is not None:
            return self._pred
        # 1) raw key
        pred = self.prediction_cache.get(self.candidate_smiles)
        # 2) canonical key
        if pred is None and _predictor_canonical is not None:
            canon = _predictor_canonical(self.candidate_smiles)
            if canon is not None:
                pred = self.prediction_cache.get(canon)
                if pred is None:
                    # cache may be raw-keyed; scan-match by canonical (rare, small cost)
                    for k, v in self.prediction_cache.items():
                        if _predictor_canonical(k) == canon:
                            pred = v
                            break
        # 3) remote fallback (only if explicitly allowed; rollout should never hit this)
        if pred is None and self.allow_remote_predict:
            try:
                from nmr_predictor import predict_nmr_peaks  # noqa: E402
                pred = predict_nmr_peaks(self.candidate_smiles)
            except Exception:
                pred = None
        self._pred = pred or {"c_peaks": [], "h_peaks": [], "ok": False, "_source": "miss"}
        return self._pred

    def _handle_spectral_match(self, args: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if spectral_similarity is None:
            return {"error": "spectral_scorer unavailable"}, {}
        sigma_c = float(args.get("sigma_c", self.sigma_c))
        sigma_h = float(args.get("sigma_h", self.sigma_h))

        pred = self._resolve_prediction()
        if not pred.get("ok"):
            return (
                {
                    "similarity": 0.0,
                    "ok": False,
                    "note": "no predicted spectrum available for this candidate (cache miss / prediction failed)",
                },
                {"pred_ok": False},
            )

        pred_c = pred.get("c_peaks", [])
        pred_h = pred.get("h_peaks", [])

        sim = float(spectral_similarity(
            pred_c_peaks=pred_c, pred_h_peaks=pred_h,
            exp_c_peaks=self.c_peaks, exp_h_peaks=self.h_peaks,
            sigma_c=sigma_c, sigma_h=sigma_h,
        ))
        # per-nucleus sub-scores (re-run scorer per nucleus for interpretability)
        c_absent = len(self.c_peaks) == 0
        c_sim = float(spectral_similarity(pred_c, [], self.c_peaks, None, sigma_c=sigma_c)) if pred_c and self.c_peaks else 0.0
        h_sim = float(spectral_similarity([], pred_h, None, self.h_peaks, sigma_h=sigma_h)) if pred_h and self.h_peaks else 0.0

        # When no experimental 13C signals exist, c_sim=0.0 is uninformative and
        # similarity already reflects 1H only (spectral_similarity re-normalises).
        c_absent_note = (
            " WARNING: n_exp_c=0 — no experimental 13C signals available; c_sim=0.0 is"
            " UNINFORMATIVE (not a mismatch signal); similarity = 1H only."
            if c_absent else ""
        )
        obs = {
            "similarity": round(sim, 4),
            "c_sim": round(c_sim, 4),
            "h_sim": round(h_sim, 4),
            "n_pred_c": len(pred_c),
            "n_exp_c": len(self.c_peaks),
            "n_pred_h": len(pred_h),
            "n_exp_h": len(self.h_peaks),
            "c_data_absent": c_absent,
            "ok": True,
            "note": (
                f"predicted-vs-experimental spectral similarity = {sim:.3f} in [0,1] "
                f"(higher = better match; 13C sim {c_sim:.3f}, 1H sim {h_sim:.3f})."
                + c_absent_note
            ),
        }
        return obs, {"similarity": sim, "c_sim": c_sim, "h_sim": h_sim, "pred_ok": True}

    # --- check_substructure -------------------------------------------------
    def _handle_check_substructure(self, args: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        pattern = str(args.get("pattern", "")).strip()
        if not pattern:
            return {"error": "pattern is required (a group name like 'ester' or a raw SMARTS)"}, {}
        smarts, resolved = resolve_pattern(pattern) if resolve_pattern else (pattern, pattern)
        patt = Chem.MolFromSmarts(smarts)
        if patt is None:
            return {"error": f"could not parse pattern as SMARTS: {smarts!r}", "input_pattern": pattern}, {}
        matches = self._mol.GetSubstructMatches(patt)
        n = len(matches)
        obs = {
            "pattern": pattern,
            "resolved_name": resolved,
            "smarts": smarts,
            "present_in_candidate": n > 0,
            "n_matches": n,
            "note": (
                f"substructure {resolved!r} {'IS' if n else 'is NOT'} present in the candidate "
                f"({n} match{'es' if n != 1 else ''})."
            ),
        }
        return obs, {"present": n > 0, "n_matches": n}

    # --- functional_group_scan ---------------------------------------------
    def _handle_functional_group_scan(self, args: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        present: List[str] = []
        for name, smarts in NAMED_SMARTS.items():
            patt = Chem.MolFromSmarts(smarts)
            if patt is None:
                continue
            if self._mol.GetSubstructMatches(patt):
                present.append(name)
        obs = {
            "groups_present": present,
            "n_groups": len(present),
            "note": f"candidate contains these known functional groups: {present}" if present
                    else "no known functional groups from the scan table were detected.",
        }
        return obs, {"n_groups": len(present)}

    # --- peak_assignment ----------------------------------------------------
    def _handle_peak_assignment(self, args: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Globally optimal one-to-one shift assignment (contract v2)."""
        nucleus = str(args.get("nucleus", "C")).strip().upper()
        if nucleus not in ("C", "H"):
            return {"error": "nucleus must be 'C' or 'H'"}, {}
        tol = 5.0 if nucleus == "C" else 0.3
        large_tol = 8.0 if nucleus == "C" else 0.35
        pred = self._resolve_prediction()
        if not pred.get("ok"):
            raw = self.c_peaks if nucleus == "C" else self.h_peaks
            values = []
            for item in raw:
                value = item.get("delta (ppm)") if nucleus == "C" else item.get("centroid", item.get("delta"))
                if value is not None: values.append(float(value))
            return {"contract_version": 2, "algorithm": "hungarian_max_in_tolerance_then_min_abs", "nucleus": nucleus, "tol_ppm": tol, "ok": False, "status": "prediction_unavailable", "n_exp": len(values), "n_pred": 0, "n_assigned": 0, "n_matched": 0, "n_unmatched": len(values), "n_unmatched_exp": len(values), "n_unmatched_pred": 0, "coverage": 0.0 if values else 1.0, "exp_coverage": 0.0 if values else 1.0, "pred_coverage": 1.0, "symmetric_coverage": 0.0 if values else 1.0, "assignments": [{"exp_index": i, "exp_ppm": v, "pred_index": None, "pred_ppm": None, "residual": None, "abs_residual": None, "matched": False} for i, v in enumerate(values)], "unmatched_exp_ppm": sorted(values), "unmatched_pred_ppm": [], "n_large_residual": 0, "large_residual_threshold_ppm": large_tol, "large_residuals": [], "residual_stats": {"n": 0, "mae": None, "rmse": None, "max_abs": None}, "note": "no predicted spectrum available"}, {}
        pred_peaks = [float(x) for x in pred.get("c_peaks" if nucleus == "C" else "h_peaks", [])]
        raw_exp = self.c_peaks if nucleus == "C" else self.h_peaks
        exp_peaks = []
        for item in raw_exp:
            value = item.get("delta (ppm)") if nucleus == "C" else item.get("centroid", item.get("delta"))
            if value is not None: exp_peaks.append(float(value))
        if not exp_peaks:
            return {"contract_version": 2, "algorithm": "hungarian_max_in_tolerance_then_min_abs", "nucleus": nucleus, "ok": True, "status": "no_experimental_data", "tol_ppm": tol, "c_data_absent": nucleus == "C", "n_exp": 0, "n_pred": len(pred_peaks), "n_assigned": 0, "n_matched": 0, "n_unmatched": 0, "n_unmatched_exp": 0, "n_unmatched_pred": len(pred_peaks), "coverage": 1.0, "exp_coverage": 1.0, "pred_coverage": 0.0 if pred_peaks else 1.0, "symmetric_coverage": 0.0 if pred_peaks else 1.0, "assignments": [], "unmatched_exp_ppm": [], "unmatched_pred_ppm": sorted(pred_peaks), "n_large_residual": 0, "large_residual_threshold_ppm": large_tol, "large_residuals": [], "residual_stats": {"n": 0, "mae": None, "rmse": None, "max_abs": None}}, {}
        if not pred_peaks:
            return {"contract_version": 2, "algorithm": "hungarian_max_in_tolerance_then_min_abs", "nucleus": nucleus, "ok": False, "status": "no_predicted_peaks", "tol_ppm": tol, "n_exp": len(exp_peaks), "n_pred": 0, "n_assigned": 0, "n_matched": 0, "n_unmatched": len(exp_peaks), "n_unmatched_exp": len(exp_peaks), "n_unmatched_pred": 0, "coverage": 0.0, "exp_coverage": 0.0, "pred_coverage": 1.0, "symmetric_coverage": 0.0, "assignments": [{"exp_index": i, "exp_ppm": v, "pred_index": None, "pred_ppm": None, "residual": None, "abs_residual": None, "matched": False} for i, v in enumerate(exp_peaks)], "unmatched_exp_ppm": sorted(exp_peaks), "unmatched_pred_ppm": [], "n_large_residual": 0, "large_residual_threshold_ppm": large_tol, "large_residuals": [], "residual_stats": {"n": 0, "mae": None, "rmse": None, "max_abs": None}, "note": f"no predicted {nucleus} peaks available"}, {}
        es=sorted(enumerate(exp_peaks),key=lambda x:(x[1],x[0]));ps=sorted(enumerate(pred_peaks),key=lambda x:(x[1],x[0]));cost=[[abs(e-p) for _,p in ps] for _,e in es];max_cost=max(max(row) for row in cost);penalty=(max_cost+1.0)*(min(len(es),len(ps))+1);objective=[[v if v<=tol else penalty+v for v in row] for row in cost];ri,ci=linear_sum_assignment(objective);paired={r:c for r,c in zip(ri,ci)};used=set(ci);assignments=[];unexp=[];resids=[];large=[]
        for r,(orig_e,e) in enumerate(es):
            if r not in paired:
                assignments.append({"exp_index":orig_e,"exp_ppm":e,"pred_index":None,"pred_ppm":None,"residual":None,"abs_residual":None,"matched":False});unexp.append(e);continue
            c=paired[r];orig_p,pv=ps[c];res=round(e-pv,4);ab=round(abs(res),4);matched=ab<=tol;row={"exp_index":orig_e,"exp_ppm":e,"pred_index":orig_p,"pred_ppm":pv,"residual":res,"abs_residual":ab,"matched":matched};assignments.append(row);resids.append(ab)
            if not matched:unexp.append(e)
            if ab>large_tol:large.append(row)
        unpred=[p for j,(_,p) in enumerate(ps) if j not in used or (j in used and cost[next(r for r,c in paired.items() if c==j)][j]>tol)]
        nm=sum(x["matched"] for x in assignments);ne=len(exp_peaks);np=len(pred_peaks);stats={"n":len(resids),"mae":round(sum(resids)/len(resids),4),"rmse":round(math.sqrt(sum(x*x for x in resids)/len(resids)),4),"max_abs":max(resids)}
        obs={"contract_version":2,"algorithm":"hungarian_max_in_tolerance_then_min_abs","nucleus":nucleus,"tol_ppm":tol,"n_exp":ne,"n_pred":np,"n_assigned":len(paired),"n_matched":nm,"n_unmatched":len(unexp),"n_unmatched_exp":len(unexp),"n_unmatched_pred":len(unpred),"coverage":round(nm/ne,4),"exp_coverage":round(nm/ne,4),"pred_coverage":round(nm/np,4),"symmetric_coverage":round(nm/max(ne,np),4),"assignments":assignments,"unmatched_exp_ppm":unexp,"unmatched_pred_ppm":unpred,"large_residual_threshold_ppm":large_tol,"n_large_residual":len(large),"large_residuals":large,"residual_stats":stats,"ok":True,"status":"ok"}
        if nucleus == "H" and pred.get("H_groups") is not None:
            obs["predicted_group_sizes"] = sorted(len(g) for g in pred["H_groups"])
            integrals = describe_integrals(self.h_peaks, self.record.get("experimental_total_nH"))
            obs["integral_complete"] = integrals["peak_table_complete"]
            if integrals["peak_table_complete"]:
                obs["experimental_integrations"] = sorted(int(p["nH"]) for p in self.h_peaks if p.get("nH") is not None)
            else:
                obs["experimental_integrations"] = None
                obs["known_nH_values"] = integrals["known_nH_values"]
        return obs,{"n_matched":nm,"n_unmatched_exp":len(unexp),"n_unmatched_pred":len(unpred),"n_large_residual":len(large),"max_abs_residual":stats["max_abs"]}


def molecule_formula(smiles: str) -> Optional[str]:
    """Convenience wrapper (mirrors repair_tool.candidate.molecule_formula)."""
    m = _mol_from_smiles(smiles)
    if m is None:
        return None
    try:
        return rdMolDescriptors.CalcMolFormula(m)
    except Exception:
        return None
