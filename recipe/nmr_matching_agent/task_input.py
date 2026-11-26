"""Canonical NMR matching task object.

Only physical task fields may be serialized into a model user message, a router
record, or a tool raw_prompt. Labels, sample identifiers, ground-truth SMILES,
and other evaluation metadata stay outside this object.

``normalize_task_input`` does not modify the caller's objects. Calling it on
its own result returns an equal dictionary.

Schema ``nmr-task-v2`` adds ``nmr_frequency_nucleus``. When the input does not
itself declare a nucleus, the value is ``unknown``. A numeric magnitude is
never used to guess ``1H`` or ``13C``.
"""

from __future__ import annotations

import copy
import json
import math
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

SCHEMA_VERSION = "nmr-task-v2"

# Fields the model, router, and tools are allowed to see.
TASK_FIELDS: Tuple[str, ...] = (
    "candidate_smiles",
    "c_nmr_peaks",
    "h_nmr_peaks",
    "nmr_solvent",
    "nmr_frequency",
    "nmr_frequency_nucleus",
    "experimental_total_nH",
)

# v6.4 training user payloads and the matching runtime both use this 13C key.
C_PEAK_FIELDS: Tuple[str, ...] = ("delta (ppm)",)

# v6.4 training 1H peaks. The runtime reads centroid/delta/nH; the other keys
# are retained so the model sees the same physical peak it was trained on.
H_PEAK_FIELDS: Tuple[str, ...] = (
    "category",
    "centroid",
    "delta",
    "j_values",
    "nH",
    "rangeMax",
    "rangeMin",
)

_C_PEAK_FIELD_SET = frozenset(C_PEAK_FIELDS)
_H_PEAK_FIELD_SET = frozenset(H_PEAK_FIELDS)
_KNOWN_FREQUENCY_NUCLEI = frozenset({"1H", "13C", "spectrometer", "unknown"})
_NUMERIC_PEAK_FIELDS = frozenset({
    "delta (ppm)",
    "centroid",
    "delta",
    "nH",
    "rangeMax",
    "rangeMin",
})

# Harness bookkeeping. These keys are not task fields and must not be copied
# into a model message, router record, or tool raw_prompt.
HARNESS_PAIR_ID_KEY = "_harness_pair_id"
HARNESS_RUN_ID_KEY = "_harness_run_id"

_CAND_RE = re.compile(r"^Candidate SMILES:\s*(\S+)\s*$", re.IGNORECASE)
_HEADER_RE = re.compile(
    r"^(13C|1H)\s+NMR\s*(?:\(([^)]*)\))?\s*:\s*(.*)$",
    re.IGNORECASE,
)
_H_PEAK_RE = re.compile(
    r"^([+-]?(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+))"
    r"(?:\s*-\s*([+-]?(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+)))?"
    r"(?:\s*\((.*)\))?$"
)
_NUMBER_TOKEN = r"[+-]?(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+)"
_NH_RE = re.compile(r"^(\d+)\s*H$", re.IGNORECASE)
_J_EQ_RE = re.compile(rf"^J\s*=\s*({_NUMBER_TOKEN})$", re.IGNORECASE)
_J_CONT_RE = re.compile(rf"^{_NUMBER_TOKEN}$")
_CATEGORY_RE = re.compile(r"^[A-Za-z][A-Za-z0-9]*(?:[ \t]+[A-Za-z][A-Za-z0-9]*)*$")


class TaskInputError(ValueError):
    """A task could not be read without inventing or dropping physical data."""


def project_peaks(peaks: Iterable[Any], allowed: Iterable[str]) -> List[Any]:
    """Copy peaks, keeping only supported physical keys on dict entries.

    Non-dict entries are deep-copied as they are. Missing keys are not added,
    so an unknown integration is not rewritten as 0. A present null integration
    stays null.
    """
    allowed_set = set(allowed)
    projected: List[Any] = []
    for peak in peaks:
        if not isinstance(peak, dict):
            projected.append(copy.deepcopy(peak))
            continue
        kept: Dict[str, Any] = {}
        for key, value in peak.items():
            if key in allowed_set:
                kept[key] = copy.deepcopy(value)
        projected.append(kept)
    return projected


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _require_finite(value: Any, where: str) -> Any:
    if value is None:
        return None
    if not _is_number(value):
        raise TaskInputError(f"{where} must be a finite number or null, got {type(value).__name__}")
    if isinstance(value, float) and not math.isfinite(value):
        raise TaskInputError(f"{where} is not a finite number")
    return value


def _require_nonnegative(value: Any, where: str) -> Any:
    number = _require_finite(value, where)
    if number is not None and number < 0:
        raise TaskInputError(f"{where} must be zero or positive, got {number}")
    return number


def _check_peak_numbers(peak: Any, where: str) -> None:
    if _is_number(peak):
        _require_finite(peak, where)
        return
    if not isinstance(peak, dict):
        raise TaskInputError(f"{where} must be a peak object or a finite number")
    for key, value in peak.items():
        if key not in _NUMERIC_PEAK_FIELDS:
            if key == "j_values":
                _check_j_values(value, f"{where}.j_values")
            continue
        if key == "nH" and value is None:
            continue
        if key not in peak:
            continue
        _require_finite(value, f"{where}.{key}")


def _check_j_values(value: Any, where: str) -> None:
    if value is None or isinstance(value, str):
        return
    if _is_number(value):
        _require_finite(value, where)
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            if isinstance(item, str):
                continue
            _require_finite(item, f"{where}[{index}]")
        return
    raise TaskInputError(f"{where} must be a string, number, list, or null")


def _frequency_nucleus(raw: Mapping[str, Any]) -> str:
    if "nmr_frequency_nucleus" not in raw:
        return "unknown"
    value = raw.get("nmr_frequency_nucleus")
    if value is None or value == "":
        return "unknown"
    if value in _KNOWN_FREQUENCY_NUCLEI:
        return str(value)
    raise TaskInputError(
        "nmr_frequency_nucleus must be 1H, 13C, spectrometer, or unknown; "
        f"got {value!r}. A frequency magnitude is not a nucleus assignment."
    )


def _candidate_string(raw: Mapping[str, Any]) -> str:
    """Return the original SMILES string. Dicts, lists, and bools are not coerced."""
    if "candidate_smiles" in raw and raw.get("candidate_smiles") not in (None, ""):
        value = raw.get("candidate_smiles")
        field = "candidate_smiles"
    elif raw.get("candidate") not in (None, ""):
        value = raw.get("candidate")
        field = "candidate"
    else:
        raise TaskInputError("input must contain candidate_smiles")
    if not isinstance(value, str):
        raise TaskInputError(f"{field} must be a SMILES string, got {type(value).__name__}")
    from rdkit import Chem
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.error")
    if Chem.MolFromSmiles(value) is None:
        raise TaskInputError(f"{field} is not a parseable SMILES: {value!r}")
    return value


def _optional_text(raw: Mapping[str, Any], primary: str, alias: str) -> Optional[str]:
    if primary in raw and raw.get(primary) not in (None, ""):
        value = raw.get(primary)
        field = primary
    elif raw.get(alias) not in (None, ""):
        value = raw.get(alias)
        field = alias
    else:
        return None
    if not isinstance(value, str):
        raise TaskInputError(f"{field} must be a string or null, got {type(value).__name__}")
    return value


def _scalar_peak(value: Any, nucleus: str, where: str) -> Dict[str, Any]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TaskInputError(f"{where} must be a peak object or a finite shift, got {type(value).__name__}")
    if isinstance(value, float) and not math.isfinite(value):
        raise TaskInputError(f"{where} is not a finite number")
    if nucleus == "C":
        return {"delta (ppm)": value}
    # A bare number has no written integral. Null keeps it unknown; it is not 1 or 0.
    return {"delta": value, "centroid": value, "nH": None}


def _complete_range(peak: Mapping[str, Any], where: str) -> bool:
    has_min = "rangeMin" in peak and peak.get("rangeMin") is not None
    has_max = "rangeMax" in peak and peak.get("rangeMax") is not None
    if not has_min and not has_max:
        return False
    if not has_min or not has_max:
        raise TaskInputError(f"{where} has an incomplete chemical-shift range")
    left = _require_finite(peak.get("rangeMin"), f"{where}.rangeMin")
    right = _require_finite(peak.get("rangeMax"), f"{where}.rangeMax")
    if left > right:
        raise TaskInputError(f"{where} has rangeMin greater than rangeMax")
    return True


def _validate_peak_object(peak: Mapping[str, Any], nucleus: str, where: str) -> None:
    if nucleus == "C":
        if "delta (ppm)" not in peak or peak.get("delta (ppm)") is None:
            raise TaskInputError(f"{where} has no finite chemical shift")
        _require_finite(peak.get("delta (ppm)"), f"{where}.delta (ppm)")
    else:
        ranged = _complete_range(peak, where)
        shift = False
        for key in ("delta", "centroid"):
            if key in peak and peak.get(key) is not None:
                _require_finite(peak.get(key), f"{where}.{key}")
                shift = True
        if not shift and not ranged:
            raise TaskInputError(f"{where} has no finite chemical shift or complete range")
        if "nH" in peak:
            _require_nonnegative(peak.get("nH"), f"{where}.nH")
        if "category" in peak and peak.get("category") is not None and not isinstance(peak.get("category"), str):
            raise TaskInputError(
                f"{where}.category must be a string or null, got {type(peak.get('category')).__name__}"
            )
        if "j_values" in peak:
            _check_j_values(peak.get("j_values"), f"{where}.j_values")


def peak_container(raw: Mapping[str, Any], key: str) -> Tuple[str, Any]:
    """Classify a peak field before alias fallback.

    A missing key, JSON null, and a list (including ``[]``) stay distinct.
    ``{}``, ``False``, ``0``, and ``""`` are illegal containers. They are not
    treated as a missing spectrum.
    """
    if key not in raw:
        return "missing", None
    value = raw.get(key)
    if value is None:
        return "null", None
    if isinstance(value, list):
        return "list", value
    raise TaskInputError(f"{key} must be a list when provided, got {type(value).__name__}")


def _peak_list(raw: Mapping[str, Any], primary: str, alias: str, nucleus: str) -> List[Any]:
    primary_state, primary_value = peak_container(raw, primary)
    alias_state, alias_value = peak_container(raw, alias)
    # An empty list still falls through to the alias. An illegal container does not.
    use_alias = primary_state in {"missing", "null"} or primary_value == []
    if use_alias:
        if alias_state != "list" or not alias_value:
            return []
        peaks = []
        for index, value in enumerate(alias_value):
            where = f"{alias}[{index}]"
            peak = _scalar_peak(value, nucleus, where)
            _validate_peak_object(peak, nucleus, where)
            peaks.append(peak)
        return peaks
    allowed = _C_PEAK_FIELD_SET if nucleus == "C" else _H_PEAK_FIELD_SET
    peaks = []
    for index, value in enumerate(primary_value):
        where = f"{primary}[{index}]"
        if isinstance(value, dict):
            peak = project_peaks([value], allowed)[0]
        else:
            peak = _scalar_peak(value, nucleus, where)
        _validate_peak_object(peak, nucleus, where)
        peaks.append(peak)
    return peaks


def task_sample_id(raw: Mapping[str, Any]) -> Any:
    """Return a caller-supplied sample locator. It is not a physical task field."""
    if not isinstance(raw, Mapping):
        return None
    for key in ("record_id", "case_id", "source_id", "idx", "id"):
        if key in raw and raw.get(key) not in (None, ""):
            return raw.get(key)
    return None


def annotate_task_error(exc: Exception, sample: Any, where: str) -> Exception:
    """Keep the original error type and add the sample plus the field location."""
    message = f"{where} sample {sample!r}: {exc}"
    if isinstance(exc, TaskInputError):
        return TaskInputError(message)
    return type(exc)(message)


def normalize_task_input(raw: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a new task dict with aliases resolved and metadata removed.

    Priority matches the historical ``build_user_case`` rules. A missing or
    empty value falls through to its alias; a present value wins:

    - ``candidate_smiles`` over ``candidate``
    - ``c_nmr_peaks`` over ``experimental_13c_peaks``
    - ``h_nmr_peaks`` over ``experimental_1h_peaks``
    - ``nmr_solvent`` over ``solvent``
    - ``nmr_frequency`` over ``nmr_freq``

    ``experimental_total_nH`` of 0 stays 0. A missing integration key is not
    inserted. ``nmr_frequency_nucleus`` defaults to ``unknown`` and is never
    inferred from the frequency number. Every peak that remains must carry a
    finite shift or a complete range. Empty peak objects are rejected at their
    index instead of being dropped or counted. A present peak field that is not
    a list, including ``{}``, ``False``, ``0``, and ``""``, raises before alias
    fallback. It is not treated as a missing spectrum.
    """
    if not isinstance(raw, Mapping):
        raise TaskInputError("task input must be a mapping")

    total = raw.get("experimental_total_nH")
    if "experimental_total_nH" in raw:
        _require_nonnegative(total, "experimental_total_nH")

    return {
        "candidate_smiles": _candidate_string(raw),
        "c_nmr_peaks": _peak_list(raw, "c_nmr_peaks", "experimental_13c_peaks", "C"),
        "h_nmr_peaks": _peak_list(raw, "h_nmr_peaks", "experimental_1h_peaks", "H"),
        "nmr_solvent": _optional_text(raw, "nmr_solvent", "solvent"),
        "nmr_frequency": _optional_text(raw, "nmr_frequency", "nmr_freq"),
        "nmr_frequency_nucleus": _frequency_nucleus(raw),
        "experimental_total_nH": total,
    }


_PRESENCE_FIELDS = (
    "candidate_smiles",
    "c_nmr_peaks",
    "h_nmr_peaks",
    "nmr_solvent",
    "nmr_frequency",
    "nmr_frequency_nucleus",
    "experimental_total_nH",
)


def describe_input_presence(rec: Mapping[str, Any]) -> Dict[str, Any]:
    """Record raw field presence before alias resolution or normalization.

    Outer structured fields, a JSON task string, and legacy Alpaca headers are
    separate layers. The input hash does not replace these states.
    """
    outer = {key: value_presence(rec, key) for key in _PRESENCE_FIELDS}
    json_layer = None
    alpaca_layer = None
    text = rec.get("input")
    if isinstance(text, str) and text.strip():
        stripped = text.strip()
        if stripped[0] in "{[":
            try:
                loaded = json.loads(stripped, parse_constant=_reject_json_constant)
            except TaskInputError:
                raise
            except json.JSONDecodeError as exc:
                raise TaskInputError(f"invalid task JSON: {exc.msg} at line {exc.lineno}") from exc
            if not isinstance(loaded, dict):
                raise TaskInputError("task JSON must be an object")
            json_layer = {key: value_presence(loaded, key) for key in _PRESENCE_FIELDS}
        else:
            alpaca_layer = _alpaca_presence(text)
    return {
        "outer_structured": outer,
        "json_string": json_layer,
        "alpaca_text": alpaca_layer,
    }


def _alpaca_presence(text: str) -> Dict[str, Any]:
    found: Dict[str, Any] = {
        "c_nmr_peaks": "missing",
        "h_nmr_peaks": "missing",
        "headers": [],
    }
    for raw_line in text.splitlines():
        header = _HEADER_RE.match(raw_line.strip())
        if header is None:
            continue
        section = header.group(1).upper()
        frequency, solvent = _split_header_parenthetical(header.group(2))
        body = header.group(3).strip()
        key = "c_nmr_peaks" if section == "13C" else "h_nmr_peaks"
        found[key] = "empty_list" if not body or body.casefold() == "n/a" else "present"
        found["headers"].append({
            "section": section,
            "frequency_raw": frequency,
            "solvent_raw": solvent,
        })
    return found


def value_presence(source: Mapping[str, Any], key: str) -> str:
    """Distinguish a missing key, JSON null, and an empty list.

    The runtime still projects null peak lists to ``[]``. That projection is
    not this presence value and is not an exact spectrum fingerprint.
    """
    if key not in source:
        return "missing"
    value = source.get(key)
    if value is None:
        return "null"
    if value == []:
        return "empty_list"
    return "present"


def peak_has_signal(peak: Any, nucleus: str) -> bool:
    """A finite shift, or an explicit finite range, is a usable experimental peak."""
    if _is_number(peak):
        return isinstance(peak, float) is False or math.isfinite(peak)
    if not isinstance(peak, dict):
        return False
    if nucleus == "C":
        value = peak.get("delta (ppm)")
        return _is_number(value) and (not isinstance(value, float) or math.isfinite(value))
    for key in ("centroid", "delta", "rangeMin", "rangeMax"):
        value = peak.get(key)
        if _is_number(value) and (not isinstance(value, float) or math.isfinite(value)):
            return True
    return False


def has_experimental_signal(task: Mapping[str, Any]) -> bool:
    carbons = task.get("c_nmr_peaks") or []
    protons = task.get("h_nmr_peaks") or []
    return any(peak_has_signal(peak, "C") for peak in carbons) or any(
        peak_has_signal(peak, "H") for peak in protons
    )


def task_to_lossless_text(task: Mapping[str, Any]) -> str:
    """Serialize the normalized task. This does not round chemical shifts."""
    normalized = normalize_task_input(task)
    return json.dumps(normalized, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _reject_json_constant(name: str) -> None:
    raise TaskInputError(f"non-finite JSON constant: {name}")


def _split_top_level_commas(text: str) -> List[str]:
    parts: List[str] = []
    buffer: List[str] = []
    depth = 0
    for char in text:
        if char == "(":
            depth += 1
            buffer.append(char)
        elif char == ")":
            if depth == 0:
                raise TaskInputError(f"unbalanced parenthesis in {text!r}")
            depth -= 1
            buffer.append(char)
        elif char == "," and depth == 0:
            parts.append("".join(buffer).strip())
            buffer = []
        else:
            buffer.append(char)
    if depth != 0:
        raise TaskInputError(f"unbalanced parenthesis in {text!r}")
    tail = "".join(buffer).strip()
    if tail:
        parts.append(tail)
    return [part for part in parts if part]


def _parse_explicit_float(token: str, where: str) -> float:
    try:
        value = float(token)
    except ValueError as exc:
        raise TaskInputError(f"invalid number at {where}: {token!r}") from exc
    if not math.isfinite(value):
        raise TaskInputError(f"non-finite number at {where}: {token!r}")
    return value


def _parse_h_inner(inner: Optional[str], where: str) -> Dict[str, Any]:
    """Split the parenthesis into whole tokens before reading J or nH.

    A J number must occupy its own comma-separated token. ``12H`` is one nH
    token, so the leading ``1`` cannot be pulled into the J list.
    """
    if inner is None or inner.strip() == "":
        return {}
    kept: Dict[str, Any] = {}
    category = None
    j_numbers: List[str] = []
    expecting_j = False
    for piece in _split_top_level_commas(inner):
        token = piece.strip()
        if not token:
            continue
        nh_match = _NH_RE.fullmatch(token)
        if nh_match:
            if "nH" in kept:
                raise TaskInputError(f"multiple nH values at {where}: {token!r}")
            kept["nH"] = int(nh_match.group(1))
            expecting_j = False
            continue
        j_eq = _J_EQ_RE.fullmatch(token)
        if j_eq:
            if j_numbers:
                raise TaskInputError(f"multiple J lists at {where}: {token!r}")
            j_numbers.append(j_eq.group(1))
            expecting_j = True
            continue
        if expecting_j and _J_CONT_RE.fullmatch(token):
            j_numbers.append(token)
            continue
        if _CATEGORY_RE.fullmatch(token) and token.casefold() != "h":
            if category is not None:
                raise TaskInputError(f"multiple categories at {where}: {token!r}")
            category = token
            expecting_j = False
            continue
        raise TaskInputError(f"unsupported token at {where}: {token!r}")
    if category is not None:
        kept["category"] = category
    if j_numbers:
        kept["j_values"] = ", ".join(j_numbers)
    return kept


def parse_h_peak_token(token: str, where: str) -> Dict[str, Any]:
    """Parse one proton token. A missing integral stays missing, including ``9.85()``."""
    match = _H_PEAK_RE.fullmatch(token.strip())
    if match is None:
        raise TaskInputError(f"unsupported 1H peak at {where}: {token!r}")
    left, right, inner = match.group(1), match.group(2), match.group(3)
    peak = _parse_h_inner(inner, where)
    if right is None:
        shift = _parse_explicit_float(left, where)
        peak["delta"] = shift
        peak["centroid"] = shift
        return peak
    peak["rangeMin"] = _parse_explicit_float(left, where)
    peak["rangeMax"] = _parse_explicit_float(right, where)
    return peak


def parse_h_peak_list(text: str, where: str = "1H") -> List[Dict[str, Any]]:
    if text is None:
        return []
    stripped = text.strip()
    if not stripped or stripped.casefold() == "n/a":
        return []
    peaks = []
    for index, token in enumerate(_split_top_level_commas(stripped), 1):
        peaks.append(parse_h_peak_token(token, f"{where} peak {index}"))
    return peaks


def parse_c_peak_list(text: str, where: str = "13C") -> List[Dict[str, Any]]:
    if text is None:
        return []
    stripped = text.strip()
    if not stripped or stripped.casefold() == "n/a":
        return []
    peaks = []
    for index, token in enumerate(_split_top_level_commas(stripped), 1):
        peaks.append({
            "delta (ppm)": _parse_explicit_float(token, f"{where} peak {index}"),
        })
    return peaks


def _split_header_parenthetical(inner: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    if inner is None or not inner.strip():
        return None, None
    frequency = None
    solvent_parts: List[str] = []
    for part in (piece.strip() for piece in inner.split(",")):
        if not part:
            continue
        if frequency is None and re.search(r"Hz|not_known", part, re.IGNORECASE):
            frequency = part
        else:
            solvent_parts.append(part)
    solvent = ", ".join(solvent_parts) or None
    return frequency, solvent


def _same_or_conflict(
    left: Optional[str],
    right: Optional[str],
    field: str,
    diagnostics: Dict[str, Any],
) -> Optional[str]:
    if left and right:
        if left == right:
            diagnostics["warnings"].append({
                "code": "header_value_copied",
                "field": field,
                "value": left,
                "note": (
                    "The 13C and 1H headers carry the same string. "
                    "It is kept once and the nucleus assignment stays unknown."
                ),
            })
            return left
        diagnostics["conflicts"].append({
            "code": "header_values_differ",
            "field": field,
            "13C_header": left,
            "1H_header": right,
            "note": "Neither value is selected and no 4x conversion is applied.",
        })
        return None
    return left or right


def _parse_alpaca_text(text: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    diagnostics: Dict[str, Any] = {
        "format": "alpaca_text",
        "warnings": [],
        "conflicts": [],
        "headers": [],
        "nucleus_assignment": "unknown",
    }
    candidate = None
    carbon_lines = []
    proton_lines = []
    for line_number, raw_line in enumerate(text.splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        candidate_match = _CAND_RE.match(line)
        if candidate_match:
            found = candidate_match.group(1).strip()
            if candidate is not None and found != candidate:
                raise TaskInputError(
                    f"line {line_number}: candidate SMILES {found!r} conflicts with {candidate!r}"
                )
            candidate = found
            continue
        header = _HEADER_RE.match(line)
        if header:
            section = header.group(1).upper()
            parenthetical = header.group(2)
            body = header.group(3).strip()
            frequency, solvent = _split_header_parenthetical(parenthetical)
            diagnostics["headers"].append({
                "line": line_number,
                "section": section,
                "parenthetical": parenthetical,
                "frequency_raw": frequency,
                "solvent_raw": solvent,
            })
            target = carbon_lines if section == "13C" else proton_lines
            target.append((line_number, body, frequency, solvent))
            continue
        raise TaskInputError(f"line {line_number}: unsupported task text: {line!r}")
    if len(carbon_lines) > 1 or len(proton_lines) > 1:
        raise TaskInputError("repeated 13C or 1H NMR line")
    task: Dict[str, Any] = {
        "candidate_smiles": candidate or "",
        "c_nmr_peaks": [],
        "h_nmr_peaks": [],
    }
    carbon_frequency = proton_frequency = None
    carbon_solvent = proton_solvent = None
    if carbon_lines:
        line_number, body, carbon_frequency, carbon_solvent = carbon_lines[0]
        task["c_nmr_peaks"] = parse_c_peak_list(body, f"line {line_number} 13C")
    if proton_lines:
        line_number, body, proton_frequency, proton_solvent = proton_lines[0]
        task["h_nmr_peaks"] = parse_h_peak_list(body, f"line {line_number} 1H")
    frequency = _same_or_conflict(carbon_frequency, proton_frequency, "nmr_frequency", diagnostics)
    solvent = _same_or_conflict(carbon_solvent, proton_solvent, "nmr_solvent", diagnostics)
    if frequency is not None:
        task["nmr_frequency"] = frequency
    if solvent is not None:
        task["nmr_solvent"] = solvent
    return task, diagnostics


def parse_task_text(text: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Parse a lossless task JSON string or a legacy Alpaca spectrum string.

    JSON gold keys are removed by normalization. Alpaca text does not assign a
    nucleus just because a header says 13C or 1H. Data the old text cannot
    express stays absent.
    """
    if not isinstance(text, str) or not text.strip():
        raise TaskInputError("task text is empty")
    stripped = text.strip()
    if stripped.startswith("{") or stripped.startswith("["):
        try:
            loaded = json.loads(stripped, parse_constant=_reject_json_constant)
        except TaskInputError:
            raise
        except json.JSONDecodeError as exc:
            raise TaskInputError(f"invalid task JSON: {exc.msg} at line {exc.lineno}") from exc
        if not isinstance(loaded, dict):
            raise TaskInputError("task JSON must be an object")
        return normalize_task_input(loaded), {
            "format": "lossless_json",
            "warnings": [],
            "conflicts": [],
            "nucleus_assignment": _frequency_nucleus(loaded),
        }
    return _parse_alpaca_text(text)


def parse_alpaca_input(input_text: str) -> Dict[str, Any]:
    """Return physical fields from an Alpaca spectrum string.

    This is the compatibility reader. It preserves an explicit J value,
    multiplicity, range, solvent, and the raw frequency string. It does not
    turn one copied header frequency into two measured frequencies.
    """
    task, _diagnostics = parse_task_text(input_text)
    return task


def describe_integrals(h_peaks: Optional[Iterable[Any]], explicit: Any) -> Dict[str, Any]:
    """Describe proton integrals without promoting a partial sum to the total.

    A complete peak-table sum exists only when every peak has a known finite
    ``nH``. ``0`` is a known value. Missing and null stay unknown. An explicit
    ``experimental_total_nH`` is a separate observation. When both exist and
    differ, neither is selected as the comparison total.
    """
    peaks = list(h_peaks or [])
    known_values: List[Any] = []
    missing = 0
    invalid = 0
    for index, peak in enumerate(peaks):
        if not isinstance(peak, dict) or "nH" not in peak:
            missing += 1
            continue
        value = peak.get("nH")
        if value is None:
            missing += 1
            continue
        if (
            not _is_number(value)
            or (isinstance(value, float) and not math.isfinite(value))
            or value < 0
        ):
            invalid += 1
            continue
        known_values.append(value)
    complete = bool(peaks) and missing == 0 and invalid == 0 and len(known_values) == len(peaks)
    known_sum: Any = None
    if known_values:
        known_sum = known_values[0]
        for value in known_values[1:]:
            known_sum += value
    peak_sum = known_sum if complete else None
    explicit_value = None
    explicit_invalid = False
    if explicit is not None:
        if (
            not _is_number(explicit)
            or (isinstance(explicit, float) and not math.isfinite(explicit))
            or explicit < 0
        ):
            explicit_invalid = True
        else:
            explicit_value = explicit
    conflict = peak_sum is not None and explicit_value is not None and peak_sum != explicit_value
    if conflict or explicit_invalid:
        used = None
        completeness = "conflict" if conflict else "invalid"
    elif explicit_value is not None:
        used = explicit_value
        completeness = "explicit"
    elif peak_sum is not None:
        used = peak_sum
        completeness = "complete"
    elif known_values:
        used = None
        completeness = "partial"
    else:
        used = None
        completeness = "unknown"
    return {
        "peak_count": len(peaks),
        "known_nH_count": len(known_values),
        "missing_nH_count": missing,
        "invalid_nH_count": invalid,
        "known_nH_values": known_values,
        "known_nH_sum": known_sum if not complete else None,
        "peak_table_complete": complete,
        "peak_table_nH_sum": peak_sum,
        "explicit_experimental_total_nH": explicit_value,
        "explicit_invalid": explicit_invalid,
        "integral_conflict": conflict,
        "experimental_total_nH": used,
        "completeness": completeness,
    }
