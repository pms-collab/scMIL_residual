#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5ad", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_latent", type=int, default=64)
    ap.add_argument("--batch_key", default=None, help="e.g. site/batch column in adata.obs")
    ap.add_argument("--max_epochs", type=int, default=200)
    args = ap.parse_args()

    import anndata as ad
    import scvi

    adata = ad.read_h5ad(args.h5ad)
    scvi.model.SCVI.setup_anndata(adata, batch_key=args.batch_key)
    model = scvi.model.SCVI(adata, n_latent=args.n_latent)
    model.train(max_epochs=args.max_epochs)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    model.save(args.out_dir, overwrite=True)
    print("saved scVI model to", args.out_dir)

if __name__ == "__main__":
    main()
