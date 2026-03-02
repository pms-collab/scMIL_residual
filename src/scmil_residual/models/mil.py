from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import noisy_or, sigmoid_safe

def masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int = 1) -> torch.Tensor:
    # mask: True for valid
    neg_inf = torch.finfo(logits.dtype).min
    logits = logits.masked_fill(~mask, neg_inf)
    return torch.softmax(logits, dim=dim)

class AttnMILBranch(nn.Module):
    def __init__(self, d_in: int, attn_hidden: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, attn_hidden),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(attn_hidden, 1),
        )

    def forward(self, X: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # X: [B, C, D], mask: [B, C]
        logits = self.net(X).squeeze(-1)            # [B, C]
        alpha = masked_softmax(logits, mask, dim=1) # [B, C]
        z = torch.einsum("bc,bcd->bd", alpha, X)    # [B, D]
        return z, alpha

class PrototypeSubspace(nn.Module):
    """Prototype subspace for disease programs.

    Two modes:
    - mode='qr'      : hard orthonormal rows via QR on V (D x K); projector P = Q Q^T.
                      (differentiable QR can be expensive on some CPU builds)
    - mode='general' : learn W directly (K x D); projector P = W^T (W W^T + eps I)^{-1} W.
                      This is an exact projector onto row-space(W) without requiring orthonormality.
    """
    def __init__(self, d_in: int, k: int, mode: str = "general", eps: float = 1e-5):
        super().__init__()
        assert mode in ("qr", "general")
        self.mode = mode
        self.eps = float(eps)
        if mode == "qr":
            self.V = nn.Parameter(torch.randn(d_in, k) * 0.02)  # D x K
            self.W_raw = None
        else:
            self.W_raw = nn.Parameter(torch.randn(k, d_in) * 0.02)  # K x D
            self.V = None

    def W(self) -> torch.Tensor:
        """Return prototype matrix W with shape [K, D]."""
        if self.mode == "qr":
            Q, _ = torch.linalg.qr(self.V, mode="reduced")  # [D,K]
            return Q.T                                       # [K,D]
        return self.W_raw

    def gram(self, W: torch.Tensor) -> torch.Tensor:
        # [K,K]
        K = W.shape[0]
        return W @ W.T + (self.eps * torch.eye(K, device=W.device, dtype=W.dtype))

    def projector_apply(self, z: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        """Apply the projector P to z.

        z: [B,D]
        W: [K,D]
        returns: Pz [B,D]
        """
        if self.mode == "qr":
            # rows of W are orthonormal => Pz = W^T (W z)
            b = torch.einsum("kd,bd->bk", W, z)            # [B,K]
            return torch.einsum("bk,kd->bd", b, W)        # [B,D]

        # general projector onto row-space(W): Pz = W^T (W W^T)^{-1} W z
        G = self.gram(W)                                    # [K,K]
        b = torch.einsum("kd,bd->bk", W, z)                # [B,K]  (W z)
        a = torch.linalg.solve(G, b.T).T                    # [B,K]  a = (W W^T)^{-1} (W z)
        return torch.einsum("bk,kd->bd", a, W)             # [B,D]

    def coefficients(self, z: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        """Return coefficients 'a' such that Pz = a W (batch).

        For qr mode: a = W z.
        For general: a = (W W^T)^{-1} W z.
        """
        b = torch.einsum("kd,bd->bk", W, z)                # [B,K]
        if self.mode == "qr":
            return b
        G = self.gram(W)
        return torch.linalg.solve(G, b.T).T                 # [B,K]

class DiseaseHead(nn.Module):
    def __init__(self, d_in: int, k: int, pool: str = "max", proto_mode: str = "general", proto_eps: float = 1e-5):
        super().__init__()
        assert pool in ("max", "noisy_or")
        self.pool = pool
        self.proto = PrototypeSubspace(d_in, k, mode=proto_mode, eps=proto_eps)
        self.bias = nn.Parameter(torch.zeros(k))

    def forward(self, z_d: torch.Tensor) -> Dict[str, torch.Tensor]:
        # z_d: [B, D]
        W = self.proto.W()                       # [K, D]
        logits_k = torch.einsum("kd,bd->bk", W, z_d) + self.bias  # [B, K]
        p_k = sigmoid_safe(logits_k)             # [B, K]
        if self.pool == "max":
            p_sick = p_k.max(dim=1).values
        else:
            p_sick = noisy_or(p_k, dim=1)
        return {"W": W, "logits_k": logits_k, "p_k": p_k, "p_sick": p_sick}

class SeverityHeadResidual(nn.Module):
    def __init__(self, d_in: int, hidden: int, dropout: float = 0.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        return self.mlp(r).squeeze(-1)  # logit

class SeverityHeadGOR(nn.Module):
    def __init__(self, d_in: int, k: int, hidden: int, dropout: float = 0.0,
                 beta_init_logit: float = -2.0):
        super().__init__()
        self.beta_logit = nn.Parameter(torch.ones(k) * beta_init_logit)
        self.mlp = nn.Sequential(
            nn.Linear(d_in + k, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def beta(self) -> torch.Tensor:
        return torch.sigmoid(self.beta_logit)  # [K]

    def forward(self, r: torch.Tensor, a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # r: [B, D], a: [B, K]
        beta = self.beta().unsqueeze(0)        # [1,K]
        a_keep = a * (1.0 - beta)              # [B,K]
        x = torch.cat([r, a_keep], dim=1)
        logit = self.mlp(x).squeeze(-1)
        return logit, beta.squeeze(0), a_keep

class GORHierMIL(nn.Module):
    def __init__(self, d_in: int, attn_hidden: int, attn_dropout: float,
                 k_prototypes: int, disease_pool: str,
                 severity_head: str, sev_mlp_hidden: int, sev_mlp_dropout: float,
                 gor_cfg: Optional[dict] = None,
                 proto_mode: str = "general",
                 proto_eps: float = 1e-5):
        super().__init__()
        self.disease_branch = AttnMILBranch(d_in, attn_hidden, attn_dropout)
        self.severity_branch = AttnMILBranch(d_in, attn_hidden, attn_dropout)
        self.disease_head = DiseaseHead(d_in, k_prototypes, pool=disease_pool, proto_mode=proto_mode, proto_eps=proto_eps)

        assert severity_head in ("residual", "gor")
        self.severity_mode = severity_head
        if severity_head == "residual":
            self.sev_head = SeverityHeadResidual(d_in, sev_mlp_hidden, sev_mlp_dropout)
        else:
            gor_cfg = gor_cfg or {}
            self.sev_head = SeverityHeadGOR(
                d_in=d_in, k=k_prototypes, hidden=sev_mlp_hidden, dropout=sev_mlp_dropout,
                beta_init_logit=float(gor_cfg.get("beta_init_logit", -2.0))
            )

    def forward(self, X: torch.Tensor, mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        # branches
        z_d, alpha_d = self.disease_branch(X, mask)   # [B,D], [B,C]
        z_s, alpha_s = self.severity_branch(X, mask)  # [B,D], [B,C]

        # disease head
        dh = self.disease_head(z_d)
        W = dh["W"]                                   # [K,D]

        # residualization: project severity summary onto the disease prototype row-space
        a = self.disease_head.proto.coefficients(z_s, W)              # [B,K]
        z_parallel = torch.einsum("bk,kd->bd", a, W)                 # [B,D]
        r = z_s - z_parallel                           # [B,D]

        if self.severity_mode == "residual":
            sev_logit = self.sev_head(r)
            beta = None
            a_keep = None
        else:
            sev_logit, beta, a_keep = self.sev_head(r, a)

        p_sev_given_sick = sigmoid_safe(sev_logit)

        # compose C/M/S probabilities
        p_sick = dh["p_sick"]
        pC = 1.0 - p_sick
        pM = p_sick * (1.0 - p_sev_given_sick)
        pS = p_sick * p_sev_given_sick
        probs = torch.stack([pC, pM, pS], dim=1)  # [B,3]

        out = {
            "z_d": z_d, "z_s": z_s,
            "alpha_d": alpha_d, "alpha_s": alpha_s,
            "W": W, "a": a, "r": r,
            "p_k": dh["p_k"], "p_sick": p_sick,
            "sev_logit": sev_logit, "p_sev_given_sick": p_sev_given_sick,
            "probs_cms": probs,
        }
        if beta is not None:
            out["beta"] = beta
            out["a_keep"] = a_keep
        return out
