#!/usr/bin/env python
from __future__ import annotations
import argparse, os
from pathlib import Path
import numpy as np
import pandas as pd

YMAP_DEFAULT = {
    "control": 0,
    "mild": 1,
    "moderate": 1,
    "mild/moderate": 1,
    "severe": 2,
    "critical": 2,
    "severe/critical": 2,
}

def _load_donor_list(path: Path) -> set[str]:
    return set([l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()])

def _normalize_sev(s: str) -> str:
    return str(s).strip().lower()

def _subsample_idx(n: int, cap: int, rng: np.random.Generator, mode: str) -> np.ndarray:
    if n <= cap:
        return np.arange(n)
    if mode == "first":
        return np.arange(cap)
    if mode == "random":
        return rng.choice(n, size=cap, replace=False)
    raise ValueError(f"unknown subsample_mode={mode}")

def main():
    ap = argparse.ArgumentParser()
    # input modes
    ap.add_argument("--latent_prefix", default=None,
                    help="Directory containing z_{train,val,test}.npy and obs_{train,val,test}.csv")
    ap.add_argument("--h5ad", default=None, help="h5ad path (requires scvi-tools if using --scvi_model_dir)")
    ap.add_argument("--scvi_model_dir", default=None, help="trained scVI model dir (optional; used when latent_prefix not given)")
    # split
    ap.add_argument("--split_dir", required=True, help="Directory containing donors_train.txt, donors_val.txt, donors_test.txt")
    # columns
    ap.add_argument("--bag_col", default="sampleID")
    ap.add_argument("--donor_col", default="donor_id")
    ap.add_argument("--sev_col", default="CoVID-19 severity")
    # bagging
    ap.add_argument("--cap", type=int, default=4096)
    ap.add_argument("--min_cells", type=int, default=100)
    ap.add_argument("--subsample_mode", choices=["random","first"], default="random")
    ap.add_argument("--seed", type=int, default=0)
    # output
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    split_dir = Path(args.split_dir)
    donors_train = _load_donor_list(split_dir/"donors_train.txt")
    donors_val   = _load_donor_list(split_dir/"donors_val.txt")
    donors_test  = _load_donor_list(split_dir/"donors_test.txt")

    def donor_to_split(d: str) -> int:
        if d in donors_train: return 0
        if d in donors_val: return 1
        if d in donors_test: return 2
        return -1

    # ---------- load cell embeddings + obs ----------
    def load_split(name: str):
        if args.latent_prefix is not None:
            pre = Path(args.latent_prefix)
            z = np.load(pre/f"z_{name}.npy")
            obs = pd.read_csv(pre/f"obs_{name}.csv")
            return z, obs
        # on-the-fly scVI
        if args.h5ad is None or args.scvi_model_dir is None:
            raise ValueError("Provide --latent_prefix OR (--h5ad and --scvi_model_dir).")
        import anndata as ad
        import scvi
        adata = ad.read_h5ad(args.h5ad)
        # Filter cells by donor split
        if args.donor_col not in adata.obs.columns:
            raise ValueError(f"missing donor col {args.donor_col} in adata.obs")
        dsplit = adata.obs[args.donor_col].astype(str).map(donor_to_split)
        keep = dsplit == {"train":0,"val":1,"test":2}[name]
        adata = adata[keep].copy()
        # load model and get latent
        model = scvi.model.SCVI.load(args.scvi_model_dir, adata=adata)
        z = model.get_latent_representation()
        obs = adata.obs.reset_index(drop=True).copy()
        return np.asarray(z), obs

    Zs, OBSs = {}, {}
    for name in ["train","val","test"]:
        z, obs = load_split(name)
        Zs[name] = np.asarray(z, dtype=np.float32)
        OBSs[name] = obs.reset_index(drop=True)

    # concat
    Z = np.concatenate([Zs["train"], Zs["val"], Zs["test"]], axis=0)
    OBS = pd.concat([OBSs["train"], OBSs["val"], OBSs["test"]], axis=0, ignore_index=True)

    # sanity: required columns
    for c in [args.bag_col, args.donor_col, args.sev_col]:
        if c not in OBS.columns:
            raise ValueError(f"missing obs column: {c}. available={list(OBS.columns)[:30]}...")

    # basic cleanup
    OBS = OBS.dropna(subset=[args.bag_col, args.donor_col, args.sev_col]).copy()
    OBS[args.bag_col] = OBS[args.bag_col].astype(str)
    OBS[args.donor_col] = OBS[args.donor_col].astype(str)
    OBS[args.sev_col] = OBS[args.sev_col].astype(str)

    # ensure Z aligns with OBS after dropna: drop same rows
    if len(OBS) != len(Z):
        # assume dropna removed rows; align by keeping non-null mask from original
        # robust path: recompute mask on original concat
        OBS0 = pd.concat([OBSs["train"], OBSs["val"], OBSs["test"]], axis=0, ignore_index=True)
        good = (~OBS0[args.bag_col].isna()) & (~OBS0[args.donor_col].isna()) & (~OBS0[args.sev_col].isna())
        Z = Z[good.values]
        OBS = OBS0.loc[good.values].copy()
        OBS[args.bag_col] = OBS[args.bag_col].astype(str)
        OBS[args.donor_col] = OBS[args.donor_col].astype(str)
        OBS[args.sev_col] = OBS[args.sev_col].astype(str)

    # map donor -> split
    split = OBS[args.donor_col].map(donor_to_split).astype(int)
    if (split < 0).any():
        bad = OBS.loc[split < 0, args.donor_col].unique()[:20]
        raise RuntimeError(f"donors not found in split lists (showing up to 20): {bad}")

    # label mapping
    sev_norm = OBS[args.sev_col].map(_normalize_sev)
    y = sev_norm.map(YMAP_DEFAULT).astype("Int64")
    if y.isna().any():
        bad = sev_norm[y.isna()].unique()[:20]
        raise RuntimeError(f"unmapped severity labels (up to 20): {bad}")
    y = y.astype(np.int64).values

    # build bag index
    bag_ids = OBS[args.bag_col].values
    donor_ids = OBS[args.donor_col].values
    split_arr = split.values.astype(np.int8)

    # group cells by bag
    # we must ensure each bag has single donor + single split + single y
    df = pd.DataFrame({
        "bag": bag_ids,
        "donor": donor_ids,
        "split": split_arr,
        "y": y,
        "cell_idx": np.arange(len(OBS), dtype=np.int64),
    })
    g = df.groupby("bag", sort=False)

    bags = []
    meta = []
    for bag, sub in g:
        if sub["donor"].nunique() != 1:
            raise RuntimeError(f"bag {bag} has multiple donors")
        if sub["split"].nunique() != 1:
            raise RuntimeError(f"bag {bag} has multiple splits (donor leakage?)")
        if sub["y"].nunique() != 1:
            raise RuntimeError(f"bag {bag} has multiple labels")
        n_cells = len(sub)
        if n_cells < args.min_cells:
            continue
        cell_idx = sub["cell_idx"].to_numpy()
        take = _subsample_idx(len(cell_idx), args.cap, rng, args.subsample_mode)
        cell_idx = cell_idx[take]
        bags.append(cell_idx)
        meta.append((bag, sub["donor"].iloc[0], int(sub["split"].iloc[0]), int(sub["y"].iloc[0]), n_cells))

    if len(bags) == 0:
        raise RuntimeError("no bags after filtering; check min_cells/cap and inputs")

    D = Z.shape[1]
    N = len(bags)
    X = np.zeros((N, args.cap, D), np.float32)
    M = np.zeros((N, args.cap), bool)
    sample_id = np.empty((N,), dtype=object)
    donor_id  = np.empty((N,), dtype=object)
    split_b   = np.zeros((N,), np.int8)
    y_b       = np.zeros((N,), np.int64)

    for i, (cell_idx, (bag, donor, spl, yy, n_cells)) in enumerate(zip(bags, meta)):
        n = len(cell_idx)
        X[i,:n] = Z[cell_idx]
        M[i,:n] = True
        sample_id[i] = str(bag)
        donor_id[i]  = str(donor)
        split_b[i]   = np.int8(spl)
        y_b[i]       = np.int64(yy)

    idx_train = np.where(split_b==0)[0].astype(np.int64)
    idx_val   = np.where(split_b==1)[0].astype(np.int64)
    idx_test  = np.where(split_b==2)[0].astype(np.int64)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path,
        X=X, mask=M, y=y_b, split=split_b,
        sample_id=sample_id, donor_id=donor_id,
        idx_train=idx_train, idx_val=idx_val, idx_test=idx_test,
    )
    print("wrote", out_path, "N_bags=", N, "D=", D)

if __name__ == "__main__":
    main()
