"""Deterministic final-evidence reconstruction for NMR matching inference.

Only parsed tool observations from the transcript are used. Model-generated reasoning is
never consulted when constructing the final think or key_evidence.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


class EvidenceReconstructionError(ValueError):
    """A fail-closed reconstruction error with a stable machine-readable code."""

    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        super().__init__(f"{code}: {detail}" if detail else code)


@dataclass(frozen=True)
class TranscriptEvidence:
    turn: int
    tool_name: str
    observation: Dict[str, Any]


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _load_json_strict(text: str) -> Any:
    return json.loads(text, parse_constant=_reject_constant)


def _render(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


_STRONG_MATCH_SIMILARITY = 0.90


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
    return None


def _as_finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _as_nonnegative_int(value: Any) -> int | None:
    number = _as_finite_float(value)
    if number is None or number < 0 or not number.is_integer():
        return None
    return int(number)


def _assignment_unmatched(obs: Dict[str, Any]) -> Tuple[int, Any]:
    for field in ("n_unmatched_exp", "n_unmatched"):
        count = _as_nonnegative_int(obs.get(field))
        if count is not None:
            values = obs.get("unmatched_exp_ppm", obs.get("unmatched_exp", []))
            return count, values
    for field in ("unmatched_exp_ppm", "unmatched_exp"):
        values = obs.get(field)
        if isinstance(values, list):
            return len(values), values
    return 0, []


def parse_final_answer_object(answer_body: str) -> Dict[str, Any]:
    try:
        parsed = _load_json_strict(answer_body)
    except (TypeError, json.JSONDecodeError, ValueError) as exc:
        raise EvidenceReconstructionError("invalid_final_answer_json", str(exc)) from exc
    if not isinstance(parsed, dict):
        raise EvidenceReconstructionError("invalid_final_answer_object", type(parsed).__name__)
    if parsed.get("final_answer") not in {"Yes", "No"}:
        raise EvidenceReconstructionError(
            "invalid_final_answer_value", repr(parsed.get("final_answer"))
        )
    return dict(parsed)


def collect_transcript_evidence(
    transcript: List[Dict[str, Any]],
) -> Tuple[List[TranscriptEvidence], List[str]]:
    evidence: List[TranscriptEvidence] = []
    diagnostics: List[str] = []
    for index, row in enumerate(transcript):
        action = row.get("action") or {}
        if action.get("type") != "tool":
            continue
        turn = int(row.get("turn", index + 1))
        tool_name = action.get("name")
        if not isinstance(tool_name, str) or not tool_name:
            diagnostics.append(f"turn={turn}:missing_tool_name")
            continue
        raw = row.get("tool_observation")
        if isinstance(raw, dict):
            observation = dict(raw)
        elif isinstance(raw, str):
            try:
                observation = _load_json_strict(raw)
            except (json.JSONDecodeError, ValueError) as exc:
                diagnostics.append(f"turn={turn},tool={tool_name}:invalid_observation_json:{exc}")
                continue
        else:
            diagnostics.append(f"turn={turn},tool={tool_name}:missing_observation")
            continue
        if not isinstance(observation, dict):
            diagnostics.append(
                f"turn={turn},tool={tool_name}:non_object_observation:{type(observation).__name__}"
            )
            continue
        evidence.append(TranscriptEvidence(turn, tool_name, observation))
    return evidence, diagnostics


def summarize_tool_observation(item: TranscriptEvidence) -> str:
    name, obs = item.tool_name, item.observation
    if "error" in obs:
        return f"{name} returned error={_render(obs.get('error'))}"

    if name == "count_atoms":
        element = obs.get("element")
        if element == "C":
            return (
                "count_atoms(C) returned "
                f"candidate_count={_render(obs.get('candidate_count'))}, "
                f"experimental_signals={_render(obs.get('experimental_distinct_signals'))}, "
                f"match={_render(obs.get('match'))}"
            )
        if element == "H":
            extra = ""
            if "exact_match" in obs:
                extra = (
                    f", exact_match={_render(obs.get('exact_match'))}, "
                    f"difference={_render(obs.get('difference'))}, "
                    f"tolerance={_render(obs.get('tolerance'))}"
                )
            return (
                "count_atoms(H) returned "
                f"candidate_count={_render(obs.get('candidate_count'))}, "
                f"experimental_total_nH={_render(obs.get('experimental_total_nH'))}, "
                f"match={_render(obs.get('match'))}{extra}"
            )

    if name == "spectral_match":
        if obs.get("ok") is not True:
            fields = [f"ok={_render(obs.get('ok'))}"]
            if "similarity" in obs:
                fields.append(f"similarity={_render(obs.get('similarity'))}")
            return "spectral_match returned " + ", ".join(fields)
        text = (
            "spectral_match returned "
            f"similarity={_render(obs.get('similarity'))}, "
            f"c_sim={_render(obs.get('c_sim'))}, h_sim={_render(obs.get('h_sim'))}, "
            f"n_pred_c={_render(obs.get('n_pred_c'))}, n_exp_c={_render(obs.get('n_exp_c'))}, "
            f"n_pred_h={_render(obs.get('n_pred_h'))}, n_exp_h={_render(obs.get('n_exp_h'))}, "
            f"c_data_absent={_render(obs.get('c_data_absent'))}, ok=true"
        )
        return text

    if name == "check_substructure":
        return (
            f"check_substructure({_render(obs.get('resolved_name') or obs.get('pattern'))}) "
            f"returned present_in_candidate={_render(obs.get('present_in_candidate'))}, "
            f"n_matches={_render(obs.get('n_matches'))}"
        )

    if name == "functional_group_scan":
        return (
            "functional_group_scan returned "
            f"n_groups={_render(obs.get('n_groups'))}, "
            f"groups_present={_render(obs.get('groups_present'))}"
        )

    if name == "peak_assignment":
        nucleus = obs.get("nucleus")
        if obs.get("c_data_absent") is True:
            return "peak_assignment(C) returned ok=true, c_data_absent=true"
        if obs.get("ok") is not True:
            return f"peak_assignment({_render(nucleus)}) returned ok={_render(obs.get('ok'))}"
        if "n_exp" not in obs:
            unmatched = obs.get("unmatched_exp", obs.get("unmatched_exp_ppm", []))
            return (
                f"peak_assignment({_render(nucleus)}) returned ok=true, "
                f"assignments={_render(obs.get('assignments', []))}, "
                f"unmatched_exp={_render(unmatched)}"
            )
        return (
            f"peak_assignment({_render(nucleus)}) returned "
            f"n_exp={_render(obs.get('n_exp'))}, n_pred={_render(obs.get('n_pred'))}, "
            f"n_matched={_render(obs.get('n_matched'))}, "
            f"n_unmatched_exp={_render(obs.get('n_unmatched_exp', obs.get('n_unmatched')))}, "
            f"n_unmatched_pred={_render(obs.get('n_unmatched_pred'))}, "
            f"n_large_residual={_render(obs.get('n_large_residual'))}, "
            f"max_abs={_render((obs.get('residual_stats') or {}).get('max_abs'))}, "
            f"symmetric_coverage={_render(obs.get('symmetric_coverage'))}, ok=true"
        )

    return f"{name} returned observation={_render(obs)}"


def _latest_by_kind(evidence: List[TranscriptEvidence]) -> Dict[str, TranscriptEvidence]:
    latest: Dict[str, TranscriptEvidence] = {}
    for item in evidence:
        obs = item.observation
        if item.tool_name == "count_atoms" and obs.get("element") in {"C", "H"}:
            key = f"count_{obs['element']}"
        elif item.tool_name == "peak_assignment":
            key = f"peak_assignment_{obs.get('nucleus')}"
        else:
            key = item.tool_name
        latest[key] = item
    return latest


def build_grounded_key_evidence(
    final_answer: str, evidence: List[TranscriptEvidence]
) -> str:
    latest = _latest_by_kind(evidence)

    # Priority 1: a hard atom-count contradiction, independent of tool-call order.
    for key, label, experimental_field in (
        ("count_C", "carbon", "experimental_distinct_signals"),
        ("count_H", "hydrogen", "experimental_total_nH"),
    ):
        item = latest.get(key)
        if not item or _as_bool(item.observation.get("match")) is not False:
            continue
        obs = item.observation
        fields = []
        candidate = _as_nonnegative_int(obs.get("candidate_count"))
        experimental = _as_nonnegative_int(obs.get(experimental_field))
        if candidate is not None:
            fields.append(f"candidate_count={candidate}")
        if experimental is not None:
            public_name = "experimental_signals" if label == "carbon" else experimental_field
            fields.append(f"{public_name}={experimental}")
        fields.append("match=false")
        return f"Decisive {label}-count contradiction: {', '.join(fields)}."

    # Priority 2: a direct one-to-one peak-assignment contradiction.
    for key in ("peak_assignment_C", "peak_assignment_H"):
        item = latest.get(key)
        if not item or _as_bool(item.observation.get("ok")) is not True:
            continue
        obs = item.observation
        unmatched_count, unmatched_values = _assignment_unmatched(obs)
        large_count = _as_nonnegative_int(obs.get("n_large_residual")) or 0
        if unmatched_count <= 0 and large_count <= 0:
            continue
        nucleus = str(obs.get("nucleus") or key.rsplit("_", 1)[-1])
        details = [f"n_unmatched={unmatched_count}"]
        coverage = _as_finite_float(obs.get("coverage"))
        if coverage is not None:
            details.append(f"coverage={_render(coverage)}")
        if unmatched_values:
            details.append(f"unmatched_exp_ppm={_render(unmatched_values)}")
        if large_count > 0:
            details.append(f"n_large_residual={large_count}")
        return f"Direct {nucleus} peak-assignment contradiction: {', '.join(details)}."

    # Priority 3: insufficient aggregate spectral agreement.
    spectral = latest.get("spectral_match")
    if spectral and _as_bool(spectral.observation.get("ok")) is True and "error" not in spectral.observation:
        obs = spectral.observation
        c_absent = _as_bool(obs.get("c_data_absent")) is True
        similarity = _as_finite_float(obs.get("similarity"))
        if (
            final_answer == "No"
            and similarity is not None
            and similarity < _STRONG_MATCH_SIMILARITY
        ):
            count_pairs = []
            count_details = []
            count_labels = []
            if not c_absent:
                c_pred = _as_nonnegative_int(obs.get("n_pred_c"))
                c_exp = _as_nonnegative_int(obs.get("n_exp_c"))
                if c_pred is not None and c_exp is not None:
                    count_pairs.append(c_pred == c_exp)
                    count_details.append(f"13C {c_pred}/{c_exp}")
                    count_labels.append("13C")
            h_pred = _as_nonnegative_int(obs.get("n_pred_h"))
            h_exp = _as_nonnegative_int(obs.get("n_exp_h"))
            if h_pred is not None and h_exp is not None:
                count_pairs.append(h_pred == h_exp)
                count_details.append(f"1H {h_pred}/{h_exp}")
                count_labels.append("1H")

            if count_pairs and all(count_pairs):
                prefix = (
                    "Peak counts agree, but "
                    if len(count_pairs) > 1
                    else f"The available {count_labels[0]} peak count agrees, but "
                )
            else:
                prefix = ""
            clauses = [
                f"{prefix}spectral_match shows low chemical-shift agreement: "
                f"similarity={_render(similarity)}"
            ]
            if count_details and not all(count_pairs):
                clauses.append(
                    "predicted/experimental peak counts are " + " and ".join(count_details)
                )

            component_scores = []
            c_sim = None if c_absent else _as_finite_float(obs.get("c_sim"))
            h_sim = _as_finite_float(obs.get("h_sim"))
            if c_sim is not None:
                component_scores.append((c_sim, "13C", "c_sim"))
            if h_sim is not None:
                component_scores.append((h_sim, "1H", "h_sim"))
            if len(component_scores) >= 2:
                weakest, other = sorted(component_scores, key=lambda item: item[0])[:2]
                if weakest[0] == other[0]:
                    clauses.append(
                        f"component scores are tied ({weakest[2]}={_render(weakest[0])}, "
                        f"{other[2]}={_render(other[0])})"
                    )
                else:
                    clauses.append(
                        f"the weakest component is {weakest[1]} "
                        f"({weakest[2]}={_render(weakest[0])} versus "
                        f"{other[2]}={_render(other[0])})"
                    )
            elif component_scores:
                only = component_scores[0]
                clauses.append(f"{only[1]} {only[2]}={_render(only[0])}")
            if c_absent:
                clauses.append("experimental 13C data are absent")
            return "; ".join(clauses) + "."

        if final_answer == "Yes":
            suffix = ", with c_data_absent=true" if c_absent else ""
            return (
                f"spectral_match returned similarity={_render(obs.get('similarity'))}, "
                f"c_sim={_render(obs.get('c_sim'))}, and h_sim={_render(obs.get('h_sim'))}{suffix}."
            )

    if final_answer == "No":
        similarity_detail = (
            "missing_or_invalid"
            if not spectral
            else repr(spectral.observation.get("similarity"))
        )
        raise EvidenceReconstructionError(
            "no_grounded_negative_evidence",
            f"no count/assignment contradiction and spectral similarity={similarity_detail}",
        )

    for key in ("peak_assignment_C", "peak_assignment_H"):
        item = latest.get(key)
        if item and _as_bool(item.observation.get("ok")) is True:
            obs = item.observation
            if _as_bool(obs.get("c_data_absent")) is True:
                return "peak_assignment(C) returned ok=true with c_data_absent=true."
            return (
                f"peak_assignment({_render(obs.get('nucleus'))}) returned "
                f"n_matched={_render(obs.get('n_matched'))}, "
                f"n_unmatched={_render(obs.get('n_unmatched'))}, and "
                f"coverage={_render(obs.get('coverage'))}."
            )

    for key in ("count_C", "count_H"):
        item = latest.get(key)
        if item:
            return summarize_tool_observation(item) + "."

    for key in ("check_substructure", "functional_group_scan"):
        item = latest.get(key)
        if item:
            return summarize_tool_observation(item) + "."

    first = evidence[0]
    return summarize_tool_observation(first) + "."


def reconstruct_model_final_turn(
    answer_body: str, transcript: List[Dict[str, Any]]
) -> Tuple[str, List[str]]:
    answer = parse_final_answer_object(answer_body)
    evidence, diagnostics = collect_transcript_evidence(transcript)
    if not evidence:
        detail = ";".join(diagnostics) if diagnostics else "transcript contains no tool result"
        raise EvidenceReconstructionError("no_parseable_tool_observation", detail)

    final_answer = answer["final_answer"]
    key_evidence = build_grounded_key_evidence(final_answer, evidence)
    answer["key_evidence"] = key_evidence
    if final_answer == "No":
        think = (
            f"The decisive negative evidence is: {key_evidence} "
            "Therefore final_answer = No."
        )
    else:
        summaries = " ; ".join(summarize_tool_observation(item) for item in evidence)
        think = f"{summaries}. Therefore final_answer = Yes."
    tagged = (
        f"<think>{think}</think>\n"
        f"<answer>{json.dumps(answer, ensure_ascii=False, allow_nan=False)}</answer>"
    )
    return tagged, diagnostics
