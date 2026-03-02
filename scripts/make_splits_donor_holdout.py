#!/usr/bin/env python
from __future__ import annotations
import argparse, os
from pathlib import Path
import numpy as np
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs_csv", required=True, help="Cell-level obs CSV with at least sampleID, donor_id, CoVID-19 severity")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train_frac", type=float, default=0.7)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--bag_col", default="sampleID")
    ap.add_argument("--donor_col", default="donor_id")
    ap.add_argument("--sev_col", default="CoVID-19 severity")
    ap.add_argument("--min_ctrl_val", type=int, default=5)
    ap.add_argument("--min_ctrl_test", type=int, default=5)
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.obs_csv)
    for c in [args.bag_col, args.donor_col, args.sev_col]:
        if c not in df.columns:
            raise ValueError(f"missing column {c} in obs_csv")

    # bag table: one row per sampleID
    g = df.groupby(args.bag_col)
    bag = pd.DataFrame({
        "sample_id": g[args.bag_col].first().astype(str),
        "donor_id": g[args.donor_col].nunique(),
        "sev_n": g[args.sev_col].nunique(),
        "donor": g[args.donor_col].first().astype(str),
        "sev": g[args.sev_col].first().astype(str),
        "n_cells": g.size().astype(int),
    }).reset_index(drop=True)

    bad = bag[(bag["donor_id"]!=1) | (bag["sev_n"]!=1)]
    if len(bad)>0:
        raise RuntimeError(f"non-unique donor/severity within sampleID; fix upstream. bad rows={len(bad)}")

    # donor-level stats by class (count bags)
    donors = bag["donor"].unique().tolist()
    rng = np.random.default_rng(args.seed)
    rng.shuffle(donors)

    # Greedy assignment by bag counts, control minima constraints
    donor_stats = []
    for d in donors:
        sub = bag[bag["donor"]==d]
        donor_stats.append((d, len(sub), (sub["sev"]=="control").sum()))
    donor_stats.sort(key=lambda x: x[1], reverse=True)

    n_bags = len(bag)
    tgt_train = int(round(args.train_frac * n_bags))
    tgt_val = int(round(args.val_frac * n_bags))
    tgt_test = n_bags - tgt_train - tgt_val
    targets = {"train": tgt_train, "val": tgt_val, "test": tgt_test}
    counts = {k:0 for k in targets}
    ctrl = {k:0 for k in targets}
    assign = {}

    for d, nb, nc in donor_stats:
        # score each split by how far below target it is (bigger deficit => preferred)
        def score(split):
            deficit = targets[split] - counts[split]
            # enforce control minima first for val/test
            bonus = 0
            if split=="val" and ctrl["val"] < args.min_ctrl_val and nc>0:
                bonus += 1e6
            if split=="test" and ctrl["test"] < args.min_ctrl_test and nc>0:
                bonus += 1e6
            return bonus + deficit
        best = max(targets.keys(), key=score)
        assign[d]=best
        counts[best]+=nb
        ctrl[best]+=nc

    # write donor lists
    for split in ["train","val","test"]:
        ds = [d for d,s in assign.items() if s==split]
        (out/f"donors_{split}.txt").write_text("\n".join(ds)+"\n", encoding="utf-8")

    meta = {"targets": targets, "counts": counts, "ctrl_counts": ctrl, "seed": args.seed}
    (out/"split_meta.json").write_text(pd.Series(meta).to_json(), encoding="utf-8")
    print("wrote", out)

if __name__ == "__main__":
    main()
