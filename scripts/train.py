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


from scmil_residual.utils import set_seed, ensure_dir, save_json
from scmil_residual.data import load_bags_npz, make_loaders
from scmil_residual.models import GORHierMIL
from scmil_residual.train import train_model
from scmil_residual.audit import write_audit

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", required=True, help="bags.npz")
    ap.add_argument("--exp_id", required=True)
    ap.add_argument("--runs_dir", default="runs")
    ap.add_argument("--num_workers", type=int, default=0)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    seed = int(cfg.get("seed", 11))
    set_seed(seed)

    run_dir = Path(args.runs_dir)/args.exp_id
    ensure_dir(str(run_dir))
    save_json(cfg, str(run_dir/"config.json"))

    bags = load_bags_npz(args.data)
    write_audit(str(run_dir/"audit"), bags.donor_id, bags.split, bags.y)

    # infer d_in from data
    d_in = int(bags.X.shape[-1])
    cfg["model"]["d_in"] = d_in

    device = torch.device(cfg.get("device","cuda") if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = make_loaders(bags, batch_size=int(cfg["train"]["batch_size"]),
                                                         num_workers=args.num_workers)

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

    train_summary = train_model(model, (train_loader,val_loader,test_loader), device, str(run_dir), cfg)
    print("train_summary:", train_summary)

if __name__ == "__main__":
    main()
