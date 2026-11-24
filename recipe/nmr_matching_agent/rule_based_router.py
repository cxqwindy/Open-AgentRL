"""Rule-based router for the NMR MATCHING agent.

When all deterministic tool signals are strongly positive, this router bypasses the
LLM reasoning loop entirely and constructs the <think>/<answer> output from a template.
For ambiguous cases it signals the caller to fall through to the fine-tuned model.

Design rationale
----------------
The fine-tuned Qwen3-32B agent suffers from "Turn-1 hallucination pollution": before
calling any tool it guesses a concrete integer in its first <think> block. Even when
tools later return the correct value the final key_evidence can silently revert to that
Turn-1 guess. This router short-circuits the problem for high-confidence positives by:
  1. Running the four base tools synchronously, then C/H assignment for would-be templates.
  2. Classifying C/H independently as available+pass, available+mismatch, or missing.
  3. Requiring every available count check plus a successful thresholded spectral score.
  4. Filling a language variant that references only actual tool outputs.
  5. Returning complete answer and transcript-ready evidence with zero LLM calls.

Dual-threshold strategy (method A)
-----------------------------------
Availability is tracked separately from pass/fail. A missing C or H channel may be
waived when the other atom-count check passes, but an available mismatch is a hard
conflict and always falls through to the model. Spectral evidence is mandatory.

  - C available+pass AND H available+pass → sim ≥ sim_threshold_both (default 0.85)
  - C missing, H available+pass           → sim ≥ sim_threshold_strict (default 0.92)
  - H missing, C available+pass           → sim ≥ sim_threshold_strict (default 0.92)
  - both C/H missing, either mismatch, or spectral ok=false → model

Each successful evidence scenario has English, Chinese, and Japanese wording variants.
A SHA-256 hash of the full sample identity, scenario ID, and configured seed selects the
variant, giving diverse language across samples and identical output across repeated or
multi-process runs.

Usage
-----
    from rule_based_router import RuleBasedRouter, load_prediction_cache

    cache = load_prediction_cache()   # optional but strongly recommended
    router = RuleBasedRouter(case_dict, prediction_cache=cache)
    result = router.analyze()

    if result.route == "template":
        # Use result.full_assistant_turn directly; no LLM call needed.
        print(result.full_assistant_turn)
    else:
        # Fall through to standalone_infer_matching.run_agent(...)
        ...
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_RECIPE_DIR = Path(__file__).resolve().parent
_SPECMASTER = "/fs_mol/cxq/SpecMaster_jp/perturbation_generator"
for _p in [str(_RECIPE_DIR), _SPECMASTER]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from nmr_matching_runtime import NmrMatchingRuntime  # noqa: E402

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ToolResults:
    count_c:  Dict[str, Any]
    count_h:  Dict[str, Any]
    spectral: Dict[str, Any]
    fg_scan:  Dict[str, Any]
    peak_c:   Dict[str, Any] = field(default_factory=dict)
    peak_h:   Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TemplateVariant:
    variant_id: str
    language: str
    text: str


@dataclass
class RouteDecision:
    route:                   str    # "template" | "model"
    n_positive:              int    # available count passes + spectral threshold pass
    confidence:              float  # weighted 0..1
    template_id:             str    # evidence scenario "T1".."T5", or ""
    template_variant_id:     str    # selected wording variant, e.g. "T3-ja"
    template_language:       str    # "en" | "zh" | "ja"
    template_variant_digest: str    # stable SHA-256 prefix used for audit
    c_available:             bool
    h_available:             bool
    c_match:                 bool
    h_match:                 bool
    sim:                     float
    would_template:          bool = False
    assignment_vetoed:       bool = False
    assignment_veto_reasons: List[str] = field(default_factory=list)
    assignment_shadow_mode:  bool = True


@dataclass
class RouterResult:
    route:               str           # "template" | "model"
    final_answer:        Optional[str] # "Yes" for template; None for model
    confidence:          float
    key_evidence:        str           # short factual sentence
    think_block:         str           # filled <think> text
    full_assistant_turn: str           # complete <think>...<answer>...</answer>
    tool_results:        ToolResults
    decision:            RouteDecision


# ---------------------------------------------------------------------------
# Evidence-scenario template pools. Each scenario has English, Chinese, and
# Japanese variants with identical placeholders and decision semantics.
# ---------------------------------------------------------------------------

_TEMPLATE_POOLS: Dict[str, List[TemplateVariant]] = {
    # T1: both C/H available; C exact signal count and H exact integration match.
    "T1": [
        TemplateVariant("T1-en", "en", (
            "count_atoms(C) returned candidate_count={c_count}, experimental_signals={c_signals}, match=true; "
            "count_atoms(H) returned candidate_count={h_count}, experimental_total_nH={h_exp}, match=true; "
            "spectral_match returned similarity={sim} (c_sim={c_sim}, h_sim={h_sim}); "
            "functional_group_scan returned {n_groups} group(s): [{groups}]. "
            "All available criteria pass. Therefore final_answer = Yes."
        )),
        TemplateVariant("T1-zh", "zh", (
            "count_atoms(C) 返回 candidate_count={c_count}、experimental_signals={c_signals}，match=true；"
            "count_atoms(H) 返回 candidate_count={h_count}、experimental_total_nH={h_exp}，match=true；"
            "spectral_match 返回 similarity={sim}（c_sim={c_sim}，h_sim={h_sim}）；"
            "functional_group_scan 返回 {n_groups} 个基团：[{groups}]。"
            "现有判据均通过。因此 final_answer = Yes。"
        )),
        TemplateVariant("T1-ja", "ja", (
            "count_atoms(C) は candidate_count={c_count}、experimental_signals={c_signals}、match=true を返した；"
            "count_atoms(H) は candidate_count={h_count}、experimental_total_nH={h_exp}、match=true を返した；"
            "spectral_match は similarity={sim}（c_sim={c_sim}、h_sim={h_sim}）を返した；"
            "functional_group_scan は {n_groups} 個の基を返した：[{groups}]。"
            "利用可能な判定条件はすべて成立する。Therefore final_answer = Yes."
        )),
    ],
    # T2: both available; C passes and H difference is within configured tolerance.
    "T2": [
        TemplateVariant("T2-en", "en", (
            "count_atoms(C) returned candidate_count={c_count}, experimental_signals={c_signals}, match=true; "
            "count_atoms(H) returned candidate_count={h_count}, experimental_total_nH={h_exp}; "
            "the observed difference {h_diff} is within integration tolerance; "
            "spectral_match returned similarity={sim} (c_sim={c_sim}, h_sim={h_sim}); "
            "functional_group_scan returned [{groups}]. Therefore final_answer = Yes."
        )),
        TemplateVariant("T2-zh", "zh", (
            "count_atoms(C) 返回 candidate_count={c_count}、experimental_signals={c_signals}，match=true；"
            "count_atoms(H) 返回 candidate_count={h_count}、experimental_total_nH={h_exp}，"
            "差值 {h_diff} 位于积分容差内；spectral_match 返回 similarity={sim}（c_sim={c_sim}，h_sim={h_sim}）；"
            "functional_group_scan 返回 [{groups}]。因此 final_answer = Yes。"
        )),
        TemplateVariant("T2-ja", "ja", (
            "count_atoms(C) は candidate_count={c_count}、experimental_signals={c_signals}、match=true；"
            "count_atoms(H) は candidate_count={h_count}、experimental_total_nH={h_exp} を返し、"
            "差 {h_diff} は積分許容範囲内である；spectral_match は similarity={sim}（c_sim={c_sim}、h_sim={h_sim}）；"
            "functional_group_scan は [{groups}] を返した。Therefore final_answer = Yes."
        )),
    ],
    # T3: both available; fewer resolved C signals than candidate carbons is non-contradictory.
    "T3": [
        TemplateVariant("T3-en", "en", (
            "count_atoms(C) returned candidate_count={c_count} and experimental_signals={c_signals}; "
            "candidate_count >= experimental_signals, so the carbon count is non-contradictory; "
            "count_atoms(H) returned candidate_count={h_count}, experimental_total_nH={h_exp}, match=true; "
            "spectral_match returned similarity={sim} (c_sim={c_sim}, h_sim={h_sim}); "
            "functional_group_scan returned [{groups}]. Therefore final_answer = Yes."
        )),
        TemplateVariant("T3-zh", "zh", (
            "count_atoms(C) 返回 candidate_count={c_count}、experimental_signals={c_signals}；"
            "由于 candidate_count >= experimental_signals，碳计数不存在硬冲突；"
            "count_atoms(H) 返回 candidate_count={h_count}、experimental_total_nH={h_exp}，match=true；"
            "spectral_match 返回 similarity={sim}（c_sim={c_sim}，h_sim={h_sim}）；"
            "functional_group_scan 返回 [{groups}]。因此 final_answer = Yes。"
        )),
        TemplateVariant("T3-ja", "ja", (
            "count_atoms(C) は candidate_count={c_count}、experimental_signals={c_signals} を返した；"
            "candidate_count >= experimental_signals のため炭素数に明確な矛盾はない；"
            "count_atoms(H) は candidate_count={h_count}、experimental_total_nH={h_exp}、match=true；"
            "spectral_match は similarity={sim}（c_sim={c_sim}、h_sim={h_sim}）；"
            "functional_group_scan は [{groups}] を返した。Therefore final_answer = Yes."
        )),
    ],
    # T4: C data missing; H and H-only spectral evidence pass the strict threshold.
    "T4": [
        TemplateVariant("T4-en", "en", (
            "Experimental 13C data are unavailable, so the C criterion is not evaluated; "
            "count_atoms(H) returned candidate_count={h_count}, experimental_total_nH={h_exp}, match=true; "
            "spectral_match returned similarity={sim}, h_sim={h_sim}, c_data_absent=true; "
            "functional_group_scan returned [{groups}]. The two available criteria pass. "
            "Therefore final_answer = Yes."
        )),
        TemplateVariant("T4-zh", "zh", (
            "实验 13C 数据缺失，因此不评估 C 判据；count_atoms(H) 返回 candidate_count={h_count}、"
            "experimental_total_nH={h_exp}，match=true；spectral_match 返回 similarity={sim}、"
            "h_sim={h_sim}、c_data_absent=true；functional_group_scan 返回 [{groups}]。"
            "两个可用判据均通过。因此 final_answer = Yes。"
        )),
        TemplateVariant("T4-ja", "ja", (
            "実験 13C データがないため C 判定は行わない；count_atoms(H) は candidate_count={h_count}、"
            "experimental_total_nH={h_exp}、match=true；spectral_match は similarity={sim}、"
            "h_sim={h_sim}、c_data_absent=true；functional_group_scan は [{groups}] を返した。"
            "利用可能な二つの条件は成立する。Therefore final_answer = Yes."
        )),
    ],
    # T5: H integrations missing; C and C-containing spectral evidence pass strict threshold.
    "T5": [
        TemplateVariant("T5-en", "en", (
            "Experimental 1H integration is unavailable, so the H criterion is not evaluated; "
            "count_atoms(C) returned candidate_count={c_count}, experimental_signals={c_signals}, match=true; "
            "spectral_match returned similarity={sim} (c_sim={c_sim}, h_sim={h_sim}); "
            "functional_group_scan returned [{groups}]. The two available criteria pass. "
            "Therefore final_answer = Yes."
        )),
        TemplateVariant("T5-zh", "zh", (
            "实验 1H 积分缺失，因此不评估 H 判据；count_atoms(C) 返回 candidate_count={c_count}、"
            "experimental_signals={c_signals}，match=true；spectral_match 返回 similarity={sim}"
            "（c_sim={c_sim}，h_sim={h_sim}）；functional_group_scan 返回 [{groups}]。"
            "两个可用判据均通过。因此 final_answer = Yes。"
        )),
        TemplateVariant("T5-ja", "ja", (
            "実験 1H 積分がないため H 判定は行わない；count_atoms(C) は candidate_count={c_count}、"
            "experimental_signals={c_signals}、match=true；spectral_match は similarity={sim}"
            "（c_sim={c_sim}、h_sim={h_sim}）；functional_group_scan は [{groups}] を返した。"
            "利用可能な二つの条件は成立する。Therefore final_answer = Yes."
        )),
    ],
}

_TEMPLATE_VARIANTS = {
    variant.variant_id: variant
    for pool in _TEMPLATE_POOLS.values()
    for variant in pool
}

# Default thresholds and deterministic wording-selection settings.
_DEFAULT_CONFIG: Dict[str, Any] = {
    "sim_threshold_both":   0.85,
    "sim_threshold_strict": 0.92,
    "h_tolerance": 2,
    "template_variant_seed": 0,
    "template_languages": ["en", "zh", "ja"],
    "assignment_veto": {
        "enabled": True, "shadow_mode": True,
        "c_max_abs_ppm": 8.0, "h_max_abs_ppm": 0.35,
        "fail_closed": True,
    },
    "confidence_weights": {"sim": 0.40, "c_sim": 0.30, "h_sim": 0.30},
}


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    if path is None:
        path = str(_RECIPE_DIR / "router_config.json")
    p = Path(path)
    if p.exists():
        return json.loads(p.read_text())
    return _DEFAULT_CONFIG.copy()


def load_prediction_cache(path: Optional[str] = None) -> Dict[str, Any]:
    if path is None:
        path = str(_RECIPE_DIR / "prediction_cache.json")
    p = Path(path)
    if p.exists():
        return json.loads(p.read_text())
    return {}


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class RuleBasedRouter:
    """Deterministic pre-filter for the NMR matching agent.

    Instantiate once per case. Call analyze() to get a RouterResult.
    """

    def __init__(
        self,
        record: Dict[str, Any],
        *,
        prediction_cache: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
        allow_remote_predict: bool = True,
    ) -> None:
        self.record = record
        cfg = config if config is not None else _DEFAULT_CONFIG
        # Dual-threshold: C&H both match → lower bar; only one matches → strict bar
        self.sim_threshold_both:   float = float(cfg.get("sim_threshold_both",   0.85))
        self.sim_threshold_strict: float = float(cfg.get("sim_threshold_strict", 0.92))
        # Back-compat: single "sim_threshold" key maps to both thresholds
        if "sim_threshold" in cfg and "sim_threshold_both" not in cfg:
            self.sim_threshold_both   = float(cfg["sim_threshold"])
            self.sim_threshold_strict = float(cfg["sim_threshold"])
        self.h_tol: int = int(cfg.get("h_tolerance", 2))
        self.template_variant_seed: int = int(cfg.get("template_variant_seed", 0))
        configured_languages = cfg.get("template_languages", ["en", "zh", "ja"])
        if not isinstance(configured_languages, list):
            raise ValueError("template_languages must be a list")
        self.template_languages = [str(language) for language in configured_languages]
        supported_languages = {variant.language for variant in _TEMPLATE_VARIANTS.values()}
        if not self.template_languages or not set(self.template_languages) <= supported_languages:
            raise ValueError(
                f"template_languages must be a non-empty subset of {sorted(supported_languages)}"
            )
        self.weights: Dict[str, float] = cfg.get(
            "confidence_weights", {"sim": 0.40, "c_sim": 0.30, "h_sim": 0.30}
        )
        self.assignment_veto: Dict[str, Any] = cfg.get("assignment_veto", {})
        self.runtime = NmrMatchingRuntime(
            record,
            prediction_cache=prediction_cache or {},
            allow_remote_predict=allow_remote_predict,
            h_tolerance=self.h_tol,
        )

    def _select_template_variant(self, template_id: str) -> Tuple[TemplateVariant, str]:
        candidates = [
            variant for variant in _TEMPLATE_POOLS[template_id]
            if variant.language in self.template_languages
        ]
        sample_identity = {
            "candidate_smiles": self.record.get("candidate_smiles"),
            "c_nmr_peaks": self.record.get("c_nmr_peaks") or [],
            "h_nmr_peaks": self.record.get("h_nmr_peaks") or [],
            "experimental_total_nH": self.record.get("experimental_total_nH"),
            "nmr_solvent": self.record.get("nmr_solvent"),
            "nmr_frequency": self.record.get("nmr_frequency"),
        }
        material = json.dumps(
            {
                "selector_version": 1,
                "seed": self.template_variant_seed,
                "template_id": template_id,
                "sample": sample_identity,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        digest = hashlib.sha256(material.encode("utf-8")).hexdigest()
        return candidates[int(digest, 16) % len(candidates)], digest

    # ------------------------------------------------------------------
    # Step 1: run tools (fully synchronous, zero LLM calls)
    # ------------------------------------------------------------------

    def run_all_tools(self) -> ToolResults:
        rt = self.runtime
        if rt._mol is None:
            _err: Dict[str, Any] = {"match": False, "error": "invalid SMILES", "ok": False}
            return ToolResults(_err, _err, {"similarity": 0.0, "ok": False, "c_data_absent": False}, {"groups_present": [], "n_groups": 0})

        count_c, _ = rt._handle_count_atoms({"element": "C"})
        count_h, _ = rt._handle_count_atoms({"element": "H"})
        spectral, _ = rt._handle_spectral_match({})
        fg_scan, _ = rt._handle_functional_group_scan({})
        return ToolResults(count_c, count_h, spectral, fg_scan)

    # ------------------------------------------------------------------
    # Step 2: evaluate routing criteria
    # ------------------------------------------------------------------

    def evaluate_criteria(self, tr: ToolResults) -> RouteDecision:
        # Availability is distinct from pass/fail. Missing evidence may be waived;
        # an available count mismatch is a hard conflict and can never use a Yes template.
        c_exp_raw = tr.count_c.get("experimental_distinct_signals")
        try:
            c_available = int(c_exp_raw) > 0 and not bool(tr.spectral.get("c_data_absent", False))
        except (TypeError, ValueError):
            c_available = False
        c_match = c_available and tr.count_c.get("match") is True

        h_match_exact = tr.count_h.get("exact_match", tr.count_h.get("match"))
        h_count_raw = tr.count_h.get("candidate_count")
        h_exp_raw = tr.count_h.get("experimental_total_nH")
        h_available = h_exp_raw is not None
        if not h_available:
            h_match = False
        elif h_match_exact is True:
            h_match = True
        else:
            try:
                h_match = abs(int(h_count_raw) - int(h_exp_raw)) <= self.h_tol
            except (TypeError, ValueError):
                h_match = False

        spectral_ok = bool(tr.spectral.get("ok", False))
        sim = float(tr.spectral.get("similarity", 0.0))
        if c_available and h_available:
            sim_threshold_eff: Optional[float] = self.sim_threshold_both
        elif c_available or h_available:
            sim_threshold_eff = self.sim_threshold_strict
        else:
            sim_threshold_eff = None
        spectral_positive = (
            spectral_ok
            and sim_threshold_eff is not None
            and sim >= sim_threshold_eff
        )

        count_mismatch = (c_available and not c_match) or (h_available and not h_match)
        available_count_passes = (
            (c_available or h_available)
            and (not c_available or c_match)
            and (not h_available or h_match)
        )
        is_template = not count_mismatch and available_count_passes and spectral_positive
        n_positive = sum([c_match, h_match, spectral_positive])

        c_sim = float(tr.spectral.get("c_sim", 0.0))
        h_sim = float(tr.spectral.get("h_sim", 0.0))
        if spectral_ok:
            confidence = round(
                self.weights["sim"] * sim
                + self.weights["c_sim"] * c_sim
                + self.weights["h_sim"] * h_sim,
                3,
            )
        else:
            confidence = 0.0

        template_id = ""
        template_variant_id = ""
        template_language = ""
        template_variant_digest = ""
        if is_template:
            if not c_available:
                template_id = "T4"
            elif not h_available:
                template_id = "T5"
            elif h_match_exact is not True:
                template_id = "T2"
            else:
                try:
                    has_reduced_c_signals = int(tr.count_c.get("candidate_count")) > int(c_exp_raw)
                except (TypeError, ValueError):
                    has_reduced_c_signals = False
                template_id = "T3" if has_reduced_c_signals else "T1"
            variant, digest = self._select_template_variant(template_id)
            template_variant_id = variant.variant_id
            template_language = variant.language
            template_variant_digest = digest[:16]

        return RouteDecision(
            route="template" if is_template else "model",
            n_positive=n_positive,
            confidence=confidence,
            template_id=template_id,
            template_variant_id=template_variant_id,
            template_language=template_language,
            template_variant_digest=template_variant_digest,
            c_available=c_available,
            h_available=h_available,
            c_match=c_match,
            h_match=h_match,
            sim=sim,
            would_template=is_template,
            assignment_shadow_mode=bool(getattr(self, "assignment_veto", {}).get("shadow_mode", True)),
        )

    def run_assignment_checks(self, tr: ToolResults, decision: RouteDecision) -> None:
        if decision.c_available:
            tr.peak_c, _ = self.runtime._handle_peak_assignment({"nucleus": "C"})
        if decision.h_available:
            tr.peak_h, _ = self.runtime._handle_peak_assignment({"nucleus": "H"})

    def apply_assignment_veto(self, tr: ToolResults, decision: RouteDecision) -> RouteDecision:
        cfg = self.assignment_veto
        if not cfg.get("enabled", False) or not decision.would_template:
            return decision
        reasons: List[str] = []
        for nucleus, obs, threshold in (
            ("C", tr.peak_c, float(cfg.get("c_max_abs_ppm", 8.0))),
            ("H", tr.peak_h, float(cfg.get("h_max_abs_ppm", 0.35))),
        ):
            if not obs:
                continue
            if not obs.get("ok"):
                if cfg.get("fail_closed", True): reasons.append(f"{nucleus}:assignment_unavailable")
                continue
            max_abs = (obs.get("residual_stats") or {}).get("max_abs")
            if isinstance(max_abs, (int, float)) and float(max_abs) > threshold:
                reasons.append(f"{nucleus}:max_abs={max_abs}>{threshold}")
        decision.assignment_vetoed = bool(reasons)
        decision.assignment_veto_reasons = reasons
        if reasons and not decision.assignment_shadow_mode:
            decision.route = "model"
            decision.template_id = decision.template_variant_id = decision.template_language = ""
            decision.template_variant_digest = ""
        return decision

    # ------------------------------------------------------------------
    # Step 3: fill template with actual tool values
    # ------------------------------------------------------------------

    def fill_template(self, template_variant_id: str, tr: ToolResults) -> str:
        tpl = _TEMPLATE_VARIANTS[template_variant_id].text

        c_count   = tr.count_c.get("candidate_count", "?")
        c_signals = tr.count_c.get("experimental_distinct_signals", "?")
        h_count   = tr.count_h.get("candidate_count", "?")
        h_exp     = tr.count_h.get("experimental_total_nH", "?")
        try:
            h_diff = abs(int(h_count) - int(h_exp))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            h_diff = "?"

        # Preserve the exact numeric values returned by spectral_match.  The complete
        # template transcript uses these observations for strict provenance checks.
        sim   = str(tr.spectral.get("similarity", 0.0))
        c_sim = str(tr.spectral.get("c_sim", 0.0))
        h_sim = str(tr.spectral.get("h_sim", 0.0))

        groups_list: List[str] = tr.fg_scan.get("groups_present", [])
        n_groups   = tr.fg_scan.get("n_groups", 0)
        groups_str = ", ".join(groups_list) if groups_list else "(none)"

        return tpl.format(
            c_count=c_count,  c_signals=c_signals,
            h_count=h_count,  h_exp=h_exp,  h_diff=h_diff,
            sim=sim,  c_sim=c_sim,  h_sim=h_sim,
            n_groups=n_groups,  groups=groups_str,
        )

    # ------------------------------------------------------------------
    # Step 4: derive key_evidence string from tool outputs
    # ------------------------------------------------------------------

    def _build_key_evidence(self, tr: ToolResults, decision: RouteDecision) -> str:
        sim = tr.spectral.get("similarity", 0.0)
        c_sim = tr.spectral.get("c_sim", 0.0)
        h_sim = tr.spectral.get("h_sim", 0.0)
        c_count = tr.count_c.get("candidate_count", "?")
        c_signals = tr.count_c.get("experimental_distinct_signals", "?")
        h_count = tr.count_h.get("candidate_count", "?")
        h_exp = tr.count_h.get("experimental_total_nH", "?")
        language = decision.template_language

        if language == "zh":
            parts = [f"spectral_match similarity={sim}（c_sim={c_sim}，h_sim={h_sim}）"]
            if decision.c_available:
                parts.append(f"C计数通过（{c_count} atoms / {c_signals} signals）")
            else:
                parts.append("实验13C数据缺失")
            if decision.h_available:
                parts.append(f"H计数通过（{h_count}/{h_exp} protons）")
            else:
                parts.append("实验1H积分缺失")
            return "；".join(parts) + "。"

        if language == "ja":
            parts = [f"spectral_match similarity={sim}（c_sim={c_sim}、h_sim={h_sim}）"]
            if decision.c_available:
                parts.append(f"C数は整合（{c_count} atoms / {c_signals} signals）")
            else:
                parts.append("実験13Cデータなし")
            if decision.h_available:
                parts.append(f"H数は整合（{h_count}/{h_exp} protons）")
            else:
                parts.append("実験1H積分なし")
            return "；".join(parts) + "。"

        parts = [f"spectral_match similarity={sim} (c_sim={c_sim}, h_sim={h_sim})"]
        if decision.c_available:
            parts.append(f"C count passed ({c_count} atoms / {c_signals} signals)")
        else:
            parts.append("experimental 13C data unavailable")
        if decision.h_available:
            parts.append(f"H count passed ({h_count}/{h_exp} protons)")
        else:
            parts.append("experimental 1H integration unavailable")
        return "; ".join(parts) + "."

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def analyze(self) -> RouterResult:
        """Run all tools, decide route, return RouterResult.

        For route="template": full_assistant_turn is ready to use as the final
            model output — no LLM call needed.
        For route="model": full_assistant_turn is empty; caller should run the
            fine-tuned model via standalone_infer_matching.run_agent().
        """
        tr       = self.run_all_tools()
        decision = self.evaluate_criteria(tr)
        if decision.would_template and self.assignment_veto.get("enabled", False):
            self.run_assignment_checks(tr, decision)
            decision = self.apply_assignment_veto(tr, decision)

        if decision.route == "template":
            think_block  = self.fill_template(decision.template_variant_id, tr)
            key_evidence = self._build_key_evidence(tr, decision)
            conf_str     = f"{decision.confidence:.2f}"
            # Escape quotes in key_evidence for embedding in JSON
            ev_safe      = key_evidence.replace('"', "'")
            full_turn = (
                f"<think>{think_block}</think>\n"
                f'<answer>{{"final_answer": "Yes", "confidence": {conf_str}, '
                f'"key_evidence": "{ev_safe}"}}</answer>'
            )
            return RouterResult(
                route="template",
                final_answer="Yes",
                confidence=decision.confidence,
                key_evidence=key_evidence,
                think_block=think_block,
                full_assistant_turn=full_turn,
                tool_results=tr,
                decision=decision,
            )

        return RouterResult(
            route="model",
            final_answer=None,
            confidence=decision.confidence,
            key_evidence="",
            think_block="",
            full_assistant_turn="",
            tool_results=tr,
            decision=decision,
        )


# ---------------------------------------------------------------------------
# Convenience: build a messages list with tool results pre-injected
# (Experimental — use when you want the model to skip tool calling and
#  directly generate its summary from pre-computed evidence.)
# ---------------------------------------------------------------------------

def build_prefilled_messages(
    case: Dict[str, Any],
    router_result: RouterResult,
    system_prompt: str,
) -> List[Dict[str, Any]]:
    """Return messages with 4 tool call+result turns pre-injected.

    Useful for the model fallback path: the model sees all tool outputs before
    generating Turn-1, so its first <think> block has nothing left to guess.
    The model should then proceed directly to a final <answer>.
    """
    tr = router_result.tool_results
    msgs: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": json.dumps(case, ensure_ascii=False)},
    ]
    tool_sequence: List[Tuple[str, Dict[str, Any], Dict[str, Any]]] = [
        ("count_atoms",         {"element": "C"}, tr.count_c),
        ("count_atoms",         {"element": "H"}, tr.count_h),
        ("spectral_match",      {},               tr.spectral),
        ("functional_group_scan", {},             tr.fg_scan),
    ]
    _decision = getattr(router_result, "decision", None)
    include_assignment = not (
        bool(getattr(_decision, "assignment_vetoed", False))
        and bool(getattr(_decision, "assignment_shadow_mode", True))
    )
    if include_assignment and getattr(tr, "peak_c", None):
        tool_sequence.append(("peak_assignment", {"nucleus": "C"}, tr.peak_c))
    if include_assignment and getattr(tr, "peak_h", None):
        tool_sequence.append(("peak_assignment", {"nucleus": "H"}, tr.peak_h))
    for name, args, result_dict in tool_sequence:
        msgs.append({
            "role": "assistant",
            "content": (
                "<think>Pre-computed tool result follows.</think>\n"
                f'<tool_call>{{"name": "{name}", "arguments": {json.dumps(args)}}}</tool_call>'
            ),
        })
        msgs.append({
            "role": "tool",
            "name": name,
            "content": json.dumps(result_dict, ensure_ascii=False),
        })
    return msgs


# ---------------------------------------------------------------------------
# CLI for quick standalone testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Rule-based router — standalone test")
    ap.add_argument("--input",  required=True,
                    help="JSON file with candidate_smiles + c_nmr_peaks + h_nmr_peaks")
    ap.add_argument("--cache",  default=str(_RECIPE_DIR / "prediction_cache.json"),
                    help="Prediction cache JSON (default: prediction_cache.json in recipe dir)")
    ap.add_argument("--config", default=str(_RECIPE_DIR / "router_config.json"),
                    help="Router config JSON")
    ap.add_argument("--sim_threshold_both",   type=float, default=None,
                    help="Override sim_threshold_both (C&H both match)")
    ap.add_argument("--sim_threshold_strict", type=float, default=None,
                    help="Override sim_threshold_strict (only one C/H channel available)")
    ap.add_argument("--h_tol", type=int, default=None,
                    help="Override h_tolerance in config")
    ap.add_argument("--template_variant_seed", type=int, default=None,
                    help="Override deterministic wording seed")
    ap.add_argument("--template_languages", nargs="+", choices=["en", "zh", "ja"],
                    help="Restrict enabled template languages")
    args = ap.parse_args()

    record = json.loads(Path(args.input).read_text())
    cache  = load_prediction_cache(args.cache)
    cfg    = load_config(args.config)
    if args.sim_threshold_both is not None:
        cfg["sim_threshold_both"] = args.sim_threshold_both
    if args.sim_threshold_strict is not None:
        cfg["sim_threshold_strict"] = args.sim_threshold_strict
    if args.h_tol is not None:
        cfg["h_tolerance"] = args.h_tol
    if args.template_variant_seed is not None:
        cfg["template_variant_seed"] = args.template_variant_seed
    if args.template_languages is not None:
        cfg["template_languages"] = args.template_languages

    router = RuleBasedRouter(record, prediction_cache=cache, config=cfg)
    result = router.analyze()

    th_both   = router.sim_threshold_both
    th_strict = router.sim_threshold_strict
    c_state = "missing" if not result.decision.c_available else (
        "pass" if result.decision.c_match else "mismatch"
    )
    h_state = "missing" if not result.decision.h_available else (
        "pass" if result.decision.h_match else "mismatch"
    )
    threshold = th_both if result.decision.c_available and result.decision.h_available else th_strict
    print(f"\n{'='*60}")
    print(f"Route      : {result.route}")
    print(f"Confidence : {result.confidence:.3f}")
    print(f"Evidence   : C={c_state}, H={h_state}, sim={result.decision.sim:.3f}, "
          f"threshold={threshold}")
    if result.route == "template":
        print(f"Template   : {result.decision.template_id}")
        print(f"Variant    : {result.decision.template_variant_id} "
              f"(language={result.decision.template_language}, "
              f"digest={result.decision.template_variant_digest})")
        print(f"\n--- think block ---\n{result.think_block}")
        print(f"\n--- key_evidence ---\n{result.key_evidence}")
        print(f"\n--- full assistant turn ---\n{result.full_assistant_turn}")
    else:
        print("→ Routing to fine-tuned model (run standalone_infer_matching.py).")
    print('='*60)
