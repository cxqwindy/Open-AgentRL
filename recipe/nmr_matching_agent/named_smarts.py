"""Named SMARTS patterns for the matching agent's check_substructure / functional_group_scan tools.

Lifted from SpecMaster_jp/perturbation_generator/operators.py (the perturbation operators embed
these SMARTS inline in their detectors; there is no standalone match helper there). We collect the
functional-group / heteroatom patterns that the negative-sample generator perturbs, because those
are exactly the groups a matching model must confirm present/absent to catch func_group / hetero /
protecting-group negatives.

`check_substructure` accepts either a friendly NAME (looked up here) or a raw SMARTS string.
`functional_group_scan` iterates NAMED_SMARTS and reports which groups are present in the candidate.
"""

from __future__ import annotations

# name -> SMARTS. Names chosen to match the vocabulary Gemini uses in the CoT `check_substructure`
# observations (methoxy/ethoxy/ester/amide/carbonyl/aromatic/TMS/...), so the CoT->trajectory
# converter (M2) can map prose keywords to a pattern with high hit rate.
NAMED_SMARTS = {
    # carbonyl family
    "ester": "[CX3;!R](=O)[OX2;!R][#6]",
    "amide": "[CX3](=O)[NX3]",
    "aldehyde": "[CX3H1](=O)[#6]",
    "ketone": "[#6][CX3](=O)[#6]",
    "carboxylic_acid": "[CX3](=O)[OX2H1]",
    "carbonyl": "[CX3]=[OX1]",
    "anhydride": "[CX3](=O)[OX2][CX3](=O)",
    # oxygen groups
    "hydroxyl": "[OX2H]",
    "methoxy": "[OX2;!R][CH3]",
    "ethoxy": "[OX2;!R][CH2][CH3]",
    "ether": "[OD2;!$([OX2][CX3]=O);!R]([#6])[#6]",
    "phenol": "[c][OX2H]",
    # nitrogen groups
    "amine_primary": "[NX3;H2][#6]",
    "amine_secondary": "[NX3;H1]([#6])[#6]",
    "nitrile": "[NX1]#[CX2]",
    "nitro": "[NX3+](=O)[O-]",
    "isocyanate": "[NX2]=[CX2]=[OX1]",
    # sulfur groups
    "thiol": "[SX2H]",
    "thioether": "[#16X2;!R]([#6])[#6]",
    "sulfone": "[SX4](=O)(=O)",
    "sulfoxide": "[SX3](=O)",
    # halogens
    "fluorine": "[F]",
    "chlorine": "[Cl]",
    "bromine": "[Br]",
    "iodine": "[I]",
    # rings / unsaturation
    "aromatic_ring": "c1ccccc1",
    "furan": "c1ccoc1",
    "pyrrole": "[nH1;R1]",
    "pyridine": "c1ccncc1",
    "thiophene": "c1cccs1",
    "alkene": "[CX3]=[CX3]",
    "alkyne": "[CX2]#[CX2]",
    # protecting groups (procedural in operators.py; approximate SMARTS here)
    "tms": "[Si]([CH3])([CH3])[CH3]",
    "tbs": "[Si]([CH3])([CH3])[C]([CH3])([CH3])[CH3]",
    "acetyl": "[CH3;!R][CX3;!R](=O)[OX2;!R]",
    # heteroatom variants perturbed by the negative generator
    "ring_oxygen": "[o;R1]",
    "ring_nh": "[nH1;R1]",
}

# keyword (as seen in CoT prose) -> canonical NAMED_SMARTS name. Used by the M2 CoT converter
# to turn a check_substructure step's prose into a concrete pattern.
KEYWORD_TO_NAME = {
    "methoxy": "methoxy", "ome": "methoxy", "-och3": "methoxy",
    "ethoxy": "ethoxy", "oet": "ethoxy",
    "ester": "ester",
    "amide": "amide",
    "aldehyde": "aldehyde", "cho": "aldehyde",
    "ketone": "ketone",
    "carboxylic": "carboxylic_acid", "cooh": "carboxylic_acid",
    "carbonyl": "carbonyl", "c=o": "carbonyl",
    "hydroxyl": "hydroxyl", "alcohol": "hydroxyl", "-oh": "hydroxyl",
    "phenol": "phenol",
    "ether": "ether",
    "amine": "amine_primary",
    "nitrile": "nitrile", "cyano": "nitrile", "c#n": "nitrile",
    "nitro": "nitro",
    "thiol": "thiol", "-sh": "thiol",
    "thioether": "thioether", "sulfide": "thioether",
    "sulfone": "sulfone",
    "sulfoxide": "sulfoxide",
    "fluor": "fluorine", "fluorine": "fluorine",
    "chlor": "chlorine", "chlorine": "chlorine",
    "brom": "bromine", "bromine": "bromine",
    "iod": "iodine", "iodine": "iodine",
    "aromatic": "aromatic_ring", "benzene": "aromatic_ring", "phenyl": "aromatic_ring",
    "furan": "furan",
    "pyrrole": "pyrrole",
    "pyridine": "pyridine",
    "thiophene": "thiophene",
    "alkene": "alkene", "vinyl": "alkene", "olefin": "alkene",
    "alkyne": "alkyne", "acetylen": "alkyne",
    "tms": "tms", "trimethylsilyl": "tms",
    "tbs": "tbs", "tert-butyldimethylsilyl": "tbs",
    "acetyl": "acetyl",
}


def resolve_pattern(pattern: str) -> tuple[str, str]:
    """Return (smarts, resolved_name). If `pattern` is a known name, look it up; else treat as SMARTS."""
    key = (pattern or "").strip()
    low = key.lower()
    if low in NAMED_SMARTS:
        return NAMED_SMARTS[low], low
    # maybe a keyword synonym
    if low in KEYWORD_TO_NAME:
        name = KEYWORD_TO_NAME[low]
        return NAMED_SMARTS[name], name
    # otherwise assume the caller passed a raw SMARTS
    return key, key
