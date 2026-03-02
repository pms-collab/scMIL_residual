#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_h5ad", required=True)
    ap.add_argument("--out_h5ad", required=True)
    ap.add_argument("--bag_col", default="sampleID")
    ap.add_argument("--cap", type=int, default=4096)
    ap.add_argument("--min_cells", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--write_cell_selection_npy", default=None,
                    help="Optional path to save selected cell indices (int64) used in the capped h5ad.")
    args = ap.parse_args()

    import anndata as ad
    rng = np.random.default_rng(args.seed)
    adata = ad.read_h5ad(args.src_h5ad)

    if args.bag_col not in adata.obs.columns:
        raise ValueError(f"missing obs column {args.bag_col}")

    obs = adata.obs.reset_index(drop=True)
    idx_keep=[]
    for bag, sub_idx in obs.groupby(args.bag_col).indices.items():
        sub_idx = np.asarray(list(sub_idx), dtype=np.int64)
        if len(sub_idx) < args.min_cells:
            continue
        if len(sub_idx) > args.cap:
            sub_idx = rng.choice(sub_idx, size=args.cap, replace=False)
        idx_keep.append(sub_idx)
    if not idx_keep:
        raise RuntimeError("no bags survived; check min_cells/cap")

    idx_keep = np.concatenate(idx_keep).astype(np.int64)
    idx_keep.sort()
    adata2 = adata[idx_keep].copy()
    Path(args.out_h5ad).parent.mkdir(parents=True, exist_ok=True)
    adata2.write_h5ad(args.out_h5ad)
    if args.write_cell_selection_npy:
        Path(args.write_cell_selection_npy).parent.mkdir(parents=True, exist_ok=True)
        np.save(args.write_cell_selection_npy, idx_keep)
    print("wrote", args.out_h5ad, "cells=", adata2.n_obs)

if __name__ == "__main__":
    main()
