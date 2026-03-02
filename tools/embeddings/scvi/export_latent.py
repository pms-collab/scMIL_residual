#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5ad", required=True)
    ap.add_argument("--scvi_model_dir", required=True)
    ap.add_argument("--split_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--donor_col", default="donor_id")
    ap.add_argument("--batch_size", type=int, default=4096)
    args = ap.parse_args()

    import anndata as ad
    import scvi

    split_dir = Path(args.split_dir)
    donors = {}
    for name in ["train","val","test"]:
        donors[name] = set([l.strip() for l in (split_dir/f"donors_{name}.txt").read_text(encoding="utf-8").splitlines() if l.strip()])

    def donor_to_split(d: str) -> str:
        if d in donors["train"]: return "train"
        if d in donors["val"]: return "val"
        if d in donors["test"]: return "test"
        return "none"

    adata = ad.read_h5ad(args.h5ad)
    model = scvi.model.SCVI.load(args.scvi_model_dir, adata=adata)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    split = adata.obs[args.donor_col].astype(str).map(donor_to_split)
    for name in ["train","val","test"]:
        mask = (split == name).values
        ad_sub = adata[mask].copy()
        z = model.get_latent_representation(adata=ad_sub)
        np.save(Path(args.out_dir)/f"z_{name}.npy", np.asarray(z, np.float32))
        ad_sub.obs.to_csv(Path(args.out_dir)/f"obs_{name}.csv", index=False)
        print("wrote", name, "cells=", ad_sub.n_obs)

if __name__ == "__main__":
    main()
