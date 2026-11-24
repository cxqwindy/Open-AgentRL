#!/usr/bin/env python3
"""Offline NMR forward-prediction cache for the MATCHING tool agent.

The `spectral_match` tool (M2) needs the candidate's predicted spectrum. The candidate SMILES
set across train/val parquet + the test_1000 alpaca file + cot_traces_v4/v6 is finite and known
ahead of time, so we predict every unique candidate ONCE and cache it -> the RL rollout / tool
runtime makes zero network calls.

Extends recipe/nmr_matching/precompute_spectral_sim.py to also harvest candidate SMILES from
parquet files (reward_model.ground_truth.candidate_smiles). Output format is identical
({smiles: {c_peaks, h_peaks, ok}}) and this can refresh the existing prediction_cache.json.

Sources of candidate SMILES:
  --parquet    : verl parquet(s); reads reward_model.ground_truth.candidate_smiles
  --cot        : cot_traces_*.jsonl; reads input.candidate_smiles
  --alpaca     : nmr_matching_test_1000.json; reads _candidate_smiles
  --jsonl      : *_benchmark.jsonl; reads candidate_smiles

Usage (coverage dry-run, no network):
    python precompute_matching_cache.py \
        --parquet recipe/nmr_matching/data/rl_train_expanded.parquet \
                  recipe/nmr_matching/data/rl_val_1000.parquet \
        --alpaca /fs_mol/cxq/.../nmr_matching_test_1000.json \
        --cot /fs_mol/cxq/SpecMaster_jp/cot_output/cot_traces_v4.jsonl \
              /fs_mol/cxq/SpecMaster_jp/cot_output/cot_traces_v6.jsonl \
        --cache recipe/nmr_matching/prediction_cache.json \
        --dry_run

    # actually predict the missing ones (touches the remote Uni-Mol service):
    python precompute_matching_cache.py ... --out recipe/nmr_matching_agent/prediction_cache.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

_PREDICTOR_PATH = "/fs_mol/cxq/SpecMaster_jp/perturbation_generator"
if _PREDICTOR_PATH not in sys.path:
    sys.path.insert(0, _PREDICTOR_PATH)


def _canon(smiles: str) -> Optional[str]:
    """Canonical SMILES matching how the predictor keys results (order-independent)."""
    try:
        from nmr_predictor import _canonical  # reuse the exact canonicalization
        return _canonical(smiles)
    except Exception:
        return smiles


def collect_from_parquet(path: str) -> List[str]:
    import pandas as pd

    df = pd.read_parquet(path)
    out = []
    for rm in df.get("reward_model", []):
        if isinstance(rm, dict):
            gt = rm.get("ground_truth", {})
            if isinstance(gt, dict):
                cand = gt.get("candidate_smiles")
                if cand:
                    out.append(cand)
    return out


def collect_from_cot(path: str) -> List[str]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            inp = rec.get("input", {})
            cand = inp.get("candidate_smiles") if isinstance(inp, dict) else None
            if cand:
                out.append(cand)
    return out


def collect_from_alpaca(path: str) -> List[str]:
    out = []
    with open(path) as f:
        data = json.load(f)
    for rec in data:
        cand = rec.get("_candidate_smiles")
        if cand:
            out.append(cand)
    return out


def collect_from_jsonl(path: str) -> List[str]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            cand = rec.get("candidate_smiles")
            if cand:
                out.append(cand)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", nargs="*", default=[])
    ap.add_argument("--cot", nargs="*", default=[])
    ap.add_argument("--alpaca", nargs="*", default=[])
    ap.add_argument("--jsonl", nargs="*", default=[])
    ap.add_argument("--cache", default="", help="existing cache to seed/resume from (read)")
    ap.add_argument("--out", default="", help="output cache path (required unless --dry_run)")
    ap.add_argument("--batch_size", type=int, default=50)
    ap.add_argument("--dry_run", action="store_true",
                    help="only report coverage vs --cache; no network prediction")
    args = ap.parse_args()

    # Collect all candidate SMILES.
    all_smiles: List[str] = []
    for p in args.parquet:
        s = collect_from_parquet(p)
        print(f"[collect] parquet {p}: {len(s)} candidates")
        all_smiles += s
    for p in args.cot:
        s = collect_from_cot(p)
        print(f"[collect] cot {p}: {len(s)} candidates")
        all_smiles += s
    for p in args.alpaca:
        s = collect_from_alpaca(p)
        print(f"[collect] alpaca {p}: {len(s)} candidates")
        all_smiles += s
    for p in args.jsonl:
        s = collect_from_jsonl(p)
        print(f"[collect] jsonl {p}: {len(s)} candidates")
        all_smiles += s

    # Dedup preserving order.
    seen = set()
    unique = []
    for s in all_smiles:
        if s not in seen:
            seen.add(s)
            unique.append(s)
    print(f"[collect] total unique (raw SMILES): {len(unique)}")

    # Load existing cache (raw-keyed).
    cache: Dict[str, Dict] = {}
    if args.cache and os.path.exists(args.cache):
        with open(args.cache) as f:
            cache = json.load(f)
        print(f"[cache] loaded {len(cache)} entries from {args.cache}")

    # Coverage: a candidate is covered if its raw key OR its canonical key is in the cache.
    # Build a canonical index of the cache once.
    cache_canon = set()
    for k in cache:
        c = _canon(k)
        if c:
            cache_canon.add(c)

    covered_raw = [s for s in unique if s in cache]
    missing = []
    for s in unique:
        if s in cache:
            continue
        c = _canon(s)
        if c and c in cache_canon:
            continue
        missing.append(s)

    n = len(unique)
    n_cov = n - len(missing)
    print(f"[coverage] {n_cov}/{n} = {100.0*n_cov/max(n,1):.1f}% covered "
          f"(raw-hit {len(covered_raw)}, canonical-hit {n_cov-len(covered_raw)})")
    print(f"[coverage] missing: {len(missing)}")

    if args.dry_run:
        if missing:
            print("[dry_run] sample missing:", missing[:5])
        return

    if not args.out:
        raise SystemExit("--out required unless --dry_run")

    # Seed output cache with the existing one.
    out_cache = dict(cache)
    if missing:
        from nmr_predictor import predict_nmr_batch  # noqa: E402
        print(f"[predict] predicting {len(missing)} missing candidates (remote Uni-Mol)...")
        preds = predict_nmr_batch(missing, batch_size=args.batch_size)
        n_ok = 0
        for smi, pred in zip(missing, preds):
            out_cache[smi] = {
                "c_peaks": pred.get("c_peaks", []),
                "h_peaks": pred.get("h_peaks", []),
                "ok": bool(pred.get("ok", False)),
            }
            if out_cache[smi]["ok"]:
                n_ok += 1
        print(f"[predict] newly predicted ok: {n_ok}/{len(missing)}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out_cache, f)
    n_ok_total = sum(1 for v in out_cache.values() if v.get("ok"))
    print(f"[write] {len(out_cache)} entries ({n_ok_total} ok) -> {args.out}")


if __name__ == "__main__":
    main()
