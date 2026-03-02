from __future__ import annotations
from typing import Dict
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

@torch.no_grad()
def _collect(model, loader, device):
    model.eval()
    ys=[]
    p_sick=[]
    p_sev=[]
    p_cms=[]
    for batch in loader:
        X=batch["X"].to(device)
        mask=batch["mask"].to(device)
        y=batch["y"].cpu().numpy()
        out=model(X, mask)
        ys.append(y)
        p_sick.append(out["p_sick"].detach().cpu().numpy())
        p_sev.append(out["p_sev_given_sick"].detach().cpu().numpy())
        p_cms.append(out["probs_cms"].detach().cpu().numpy())
    y=np.concatenate(ys)
    return y, np.concatenate(p_sick), np.concatenate(p_sev), np.concatenate(p_cms)

def _safe_auc(y_true, y_score):
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return float("nan")

def _safe_ap(y_true, y_score):
    try:
        return float(average_precision_score(y_true, y_score))
    except Exception:
        return float("nan")

def _macro_ovr_auc(y3: np.ndarray, p3: np.ndarray) -> float:
    # one-vs-rest macro AUC across present classes
    aucs=[]
    for c in [0,1,2]:
        y=(y3==c).astype(int)
        if y.sum()==0 or y.sum()==len(y):
            continue
        aucs.append(_safe_auc(y, p3[:,c]))
    if not aucs:
        return float("nan")
    return float(np.nanmean(aucs))

def compute_metrics(model, loader, device) -> Dict:
    y3, psick, psev, p3 = _collect(model, loader, device)

    y_sick=(y3>0).astype(int)
    y_sev=(y3==2).astype(int)

    m = {
        "n": int(len(y3)),
        "sick": {
            "n": int(len(y3)),
            "roc_auc": _safe_auc(y_sick, psick),
            "pr_auc": _safe_ap(y_sick, psick),
        },
        "sev_cond": {
            "n_sick": int(y_sick.sum()),
            "roc_auc": _safe_auc(y_sev[y_sick==1], psev[y_sick==1]) if y_sick.sum()>0 else float("nan"),
            "pr_auc": _safe_ap(y_sev[y_sick==1], psev[y_sick==1]) if y_sick.sum()>0 else float("nan"),
        },
        "staging": {
            "macro_auc": _macro_ovr_auc(y3, p3),
        }
    }
    return m
