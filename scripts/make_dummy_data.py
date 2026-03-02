#!/usr/bin/env python
from __future__ import annotations
import argparse, os
import numpy as np
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_bags", type=int, default=60)
    ap.add_argument("--cap", type=int, default=512)
    ap.add_argument("--d", type=int, default=64)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    # create dummy splits at donor-level
    donors = np.array([f"D{i:03d}" for i in range(args.n_bags)])
    sample = np.array([f"S{i:03d}" for i in range(args.n_bags)])
    split = np.zeros(args.n_bags, np.int8)
    split[int(0.7*args.n_bags):int(0.85*args.n_bags)] = 1
    split[int(0.85*args.n_bags):] = 2

    # labels
    y = rng.choice([0,1,2], size=args.n_bags, p=[0.4,0.35,0.25]).astype(np.int64)

    # generate bags with variable n_cells
    X = np.zeros((args.n_bags, args.cap, args.d), np.float32)
    mask = np.zeros((args.n_bags, args.cap), bool)
    for i in range(args.n_bags):
        n = rng.integers(low=int(args.cap*0.2), high=args.cap)
        x = rng.normal(size=(n, args.d)).astype(np.float32)
        # add stage signal
        x += (y[i] * 0.3) * rng.normal(size=(1,args.d)).astype(np.float32)
        X[i,:n]=x
        mask[i,:n]=True

    idx_train = np.where(split==0)[0].astype(np.int64)
    idx_val = np.where(split==1)[0].astype(np.int64)
    idx_test = np.where(split==2)[0].astype(np.int64)

    np.savez_compressed(out/"bags.npz",
        X=X, mask=mask, y=y, split=split,
        sample_id=sample.astype(object), donor_id=donors.astype(object),
        idx_train=idx_train, idx_val=idx_val, idx_test=idx_test,
    )
    print("wrote", out/"bags.npz")

if __name__ == "__main__":
    main()
