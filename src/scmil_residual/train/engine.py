from __future__ import annotations
import os, time
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
import torch
from torch.optim import AdamW
from tqdm.auto import tqdm

from ..utils import ensure_dir, save_json, to_device
from ..eval.metrics import compute_metrics
from .losses import compute_losses

@dataclass
class TrainState:
    best_val: float
    best_epoch: int
    bad_epochs: int

def train_model(model: torch.nn.Module,
                loaders: Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader],
                device: torch.device,
                out_dir: str,
                cfg: dict) -> Dict:
    train_loader, val_loader, _ = loaders
    ensure_dir(out_dir)
    ckpt_dir = ensure_dir(os.path.join(out_dir, "checkpoints"))
    log_dir  = ensure_dir(os.path.join(out_dir, "logs"))

    optim = AdamW(model.parameters(),
                  lr=float(cfg["train"]["lr"]),
                  weight_decay=float(cfg["train"]["weight_decay"]))
    epochs = int(cfg["train"]["epochs"])
    patience = int(cfg["train"]["patience"])
    warmup = int(cfg["train"].get("stage1_warmup_epochs", 0))

    lw = cfg["train"]["loss_weights"]
    beta_ent_w = float(cfg.get("model", {}).get("gor", {}).get("beta_entropy_weight", 0.0))
    beta_l2_w  = float(cfg.get("model", {}).get("gor", {}).get("beta_l2_weight", 0.0))

    state = TrainState(best_val=-1e9, best_epoch=-1, bad_epochs=0)
    history = []

    for epoch in range(1, epochs+1):
        model.train()
        t0=time.time()
        pbar = tqdm(train_loader, desc=f"train e{epoch}", leave=False)
        running = {}

        # stage1: freeze severity branch + head
        if warmup > 0 and epoch <= warmup:
            for name, p in model.named_parameters():
                if ("severity_branch" in name) or ("sev_head" in name):
                    p.requires_grad_(False)
                else:
                    p.requires_grad_(True)
        else:
            for p in model.parameters():
                p.requires_grad_(True)

        for batch in pbar:
            batch = to_device(batch, device)
            X, mask, y = batch["X"], batch["mask"], batch["y"]
            out = model(X, mask)
            lo = compute_losses(
                out, y,
                w_sick=float(lw["sick_bce"]),
                w_sev=float(lw["sev_bce"]),
                beta_entropy_weight=beta_ent_w,
                beta_l2_weight=beta_l2_w,
            )
            optim.zero_grad(set_to_none=True)
            lo.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optim.step()

            for k,v in lo.parts.items():
                running[k] = running.get(k, 0.0) + float(v.detach().cpu())
            running["total"] = running.get("total", 0.0) + float(lo.total.detach().cpu())

        # normalize
        n_batches = max(1, len(train_loader))
        train_log = {k: v/n_batches for k,v in running.items()}

        # val
        val_metrics = compute_metrics(model, val_loader, device)
        val_key = val_metrics["staging"]["macro_auc"]  # robust-ish for imbalanced 3-class
        epoch_log = {"epoch": epoch, "time_sec": time.time()-t0, "train": train_log, "val": val_metrics}
        history.append(epoch_log)
        save_json(epoch_log, os.path.join(log_dir, f"epoch_{epoch:03d}.json"))

        # ckpt
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optim": optim.state_dict(),
            "cfg": cfg,
            "val_key": val_key,
        }
        torch.save(ckpt, os.path.join(ckpt_dir, "last.pt"))

        if val_key > state.best_val + 1e-6:
            state.best_val = val_key
            state.best_epoch = epoch
            state.bad_epochs = 0
            torch.save(ckpt, os.path.join(ckpt_dir, "best.pt"))
        else:
            state.bad_epochs += 1

        if state.bad_epochs >= patience:
            break

    final = {"best_epoch": state.best_epoch, "best_val_key": state.best_val, "history_len": len(history)}
    save_json(final, os.path.join(out_dir, "train_summary.json"))
    return final
