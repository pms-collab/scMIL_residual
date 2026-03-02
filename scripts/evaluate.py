#!/usr/bin/env python
from __future__ import annotations
import argparse, os
from pathlib import Path
import yaml
import torch

# allow running without `pip install -e .`
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

import numpy as np

from scmil_residual.data import load_bags_npz, make_loaders
from scmil_residual.models import GORHierMIL
from scmil_residual.eval import compute_metrics, export_predictions
from scmil_residual.utils import save_json

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="runs/<EXP_ID>")
    ap.add_argument("--data", default=None, help="bags.npz (if omitted, will use run/config.json only for model params)")
    ap.add_argument("--ckpt", default="best", choices=["best","last"])
    ap.add_argument("--num_workers", type=int, default=0)
    args = ap.parse_args()

    run_dir = Path(args.run)
    cfg = yaml.safe_load((run_dir/"config.json").read_text(encoding="utf-8"))

    data_path = args.data
    if data_path is None:
        raise ValueError("Provide --data bags.npz for evaluation (kept explicit to avoid silent mismatch).")

    bags = load_bags_npz(data_path)
    d_in = int(bags.X.shape[-1])
    device = torch.device(cfg.get("device","cuda") if torch.cuda.is_available() else "cpu")
    _, val_loader, test_loader = make_loaders(bags, batch_size=16, num_workers=args.num_workers)

    model = GORHierMIL(
        d_in=d_in,
        attn_hidden=int(cfg["model"]["attn_hidden"]),
        attn_dropout=float(cfg["model"]["attn_dropout"]),
        k_prototypes=int(cfg["model"]["k_prototypes"]),
        disease_pool=str(cfg["model"].get("disease_pool","max")),
        severity_head=str(cfg["model"].get("severity_head","residual")),
        sev_mlp_hidden=int(cfg["model"]["sev_mlp_hidden"]),
        sev_mlp_dropout=float(cfg["model"]["sev_mlp_dropout"]),
        gor_cfg=cfg["model"].get("gor", None),
        proto_mode=str(cfg["model"].get("prototype_mode", "general")),
        proto_eps=float(cfg["model"].get("prototype_eps", 1e-5)),
    ).to(device)

    ckpt_path = run_dir/"checkpoints"/f"{args.ckpt}.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)

    val_m = compute_metrics(model, val_loader, device)
    test_m = compute_metrics(model, test_loader, device)
    out = {"val": val_m, "test": test_m, "ckpt": args.ckpt}
    save_json(out, str(run_dir/"final_metrics.json"))

    meta = {"sample_id": bags.sample_id, "donor_id": bags.donor_id, "split": bags.split}
    export_predictions(model, val_loader, device, str(run_dir/"val_predictions.csv"), meta=meta)
    export_predictions(model, test_loader, device, str(run_dir/"test_predictions.csv"), meta=meta)

    print("wrote", run_dir/"final_metrics.json")

if __name__ == "__main__":
    main()
