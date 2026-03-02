from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional
import torch
import torch.nn.functional as F

@dataclass
class LossOut:
    total: torch.Tensor
    parts: Dict[str, torch.Tensor]

def bce_from_probs(p: torch.Tensor, y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = p.clamp(eps, 1 - eps)
    return F.binary_cross_entropy(p, y.float())

def compute_losses(out: Dict[str, torch.Tensor], y3: torch.Tensor,
                   w_sick: float = 1.0, w_sev: float = 1.0,
                   beta_entropy_weight: float = 0.0, beta_l2_weight: float = 0.0) -> LossOut:
    # y3: 0/1/2
    y_sick = (y3 > 0).float()
    y_sev  = (y3 == 2).float()

    p_sick = out["p_sick"]
    p_sev_given_sick = out["p_sev_given_sick"]

    loss_sick = bce_from_probs(p_sick, y_sick)
    # conditional severity defined only for sick bags; use mask
    sick_mask = (y_sick > 0.5)
    if sick_mask.any():
        loss_sev = bce_from_probs(p_sev_given_sick[sick_mask], y_sev[sick_mask])
    else:
        loss_sev = torch.tensor(0.0, device=y3.device)

    total = w_sick * loss_sick + w_sev * loss_sev
    parts = {"loss_sick": loss_sick, "loss_sev": loss_sev}

    if "beta" in out and (beta_entropy_weight > 0 or beta_l2_weight > 0):
        beta = out["beta"].clamp(1e-6, 1 - 1e-6)  # [K]
        ent = -(beta * beta.log() + (1 - beta) * (1 - beta).log()).mean()
        l2 = (beta ** 2).mean()
        total = total + beta_entropy_weight * (-ent) + beta_l2_weight * l2
        parts["beta_entropy"] = ent
        parts["beta_l2"] = l2

    return LossOut(total=total, parts=parts)
