"""Canonical NMR matching task object.

Only physical task fields may be serialized into a model user message, a router
record, or a tool raw_prompt. Labels, sample identifiers, ground-truth SMILES,
and other evaluation metadata stay outside this object.

``normalize_task_input`` does not modify the caller's objects. Calling it on
its own result returns an equal dictionary.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Iterable, List, Mapping, Tuple

# Fields the model, router, and tools are allowed to see.
TASK_FIELDS: Tuple[str, ...] = (
    "candidate_smiles",
    "c_nmr_peaks",
    "h_nmr_peaks",
    "nmr_solvent",
    "nmr_frequency",
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


def normalize_task_input(raw: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a new task dict with aliases resolved and metadata removed.

    Priority matches the historical ``build_user_case`` rules. A missing or
    empty value falls through to its alias; a present value wins:

    - ``candidate_smiles`` over ``candidate``
    - ``c_nmr_peaks`` over ``experimental_13c_peaks``
    - ``h_nmr_peaks`` over ``experimental_1h_peaks``
    - ``nmr_solvent`` over ``solvent``
    - ``nmr_frequency`` over ``nmr_freq``
    """
    if not isinstance(raw, Mapping):
        raise TypeError("task input must be a mapping")

    candidate = raw.get("candidate_smiles") or raw.get("candidate")
    if not candidate:
        raise ValueError("input must contain candidate_smiles")

    c_nmr = raw.get("c_nmr_peaks")
    if not c_nmr:
        flat_c = raw.get("experimental_13c_peaks") or []
        c_peaks = [{"delta (ppm)": float(value)} for value in flat_c]
    elif isinstance(c_nmr, list):
        c_peaks = project_peaks(c_nmr, _C_PEAK_FIELD_SET)
    else:
        raise TypeError("c_nmr_peaks must be a list when provided")

    h_nmr = raw.get("h_nmr_peaks")
    if not h_nmr:
        flat_h = raw.get("experimental_1h_peaks") or []
        h_peaks = [
            {"delta": float(value), "centroid": float(value), "nH": None}
            for value in flat_h
        ]
    elif isinstance(h_nmr, list):
        h_peaks = project_peaks(h_nmr, _H_PEAK_FIELD_SET)
    else:
        raise TypeError("h_nmr_peaks must be a list when provided")

    return {
        "candidate_smiles": candidate,
        "c_nmr_peaks": c_peaks,
        "h_nmr_peaks": h_peaks,
        "nmr_solvent": raw.get("nmr_solvent") or raw.get("solvent"),
        "nmr_frequency": raw.get("nmr_frequency") or raw.get("nmr_freq"),
        "experimental_total_nH": raw.get("experimental_total_nH"),
    }
