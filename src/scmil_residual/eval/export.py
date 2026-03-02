from __future__ import annotations
import os
from typing import Dict
import numpy as np
import pandas as pd
import torch
from ..utils import ensure_dir

@torch.no_grad()
def export_predictions(model, loader, device, out_csv: str, meta: Dict[str, np.ndarray] | None = None) -> None:
    model.eval()
    rows=[]
    for batch in loader:
        idx = batch["idx"].cpu().numpy()
        X=batch["X"].to(device)
        mask=batch["mask"].to(device)
        y=batch["y"].cpu().numpy()
        out=model(X, mask)
        p_sick=out["p_sick"].detach().cpu().numpy()
        p_sev=out["p_sev_given_sick"].detach().cpu().numpy()
        p3=out["probs_cms"].detach().cpu().numpy()
        for i,j in enumerate(idx):
            r={"idx": int(j), "y": int(y[i]), "p_sick": float(p_sick[i]), "p_sev_given_sick": float(p_sev[i]),
               "pC": float(p3[i,0]), "pM": float(p3[i,1]), "pS": float(p3[i,2])}
            rows.append(r)
    df=pd.DataFrame(rows).sort_values("idx")
    if meta is not None:
        df["sample_id"] = meta["sample_id"][df["idx"].values]
        df["donor_id"]  = meta["donor_id"][df["idx"].values]
        df["split"]     = meta["split"][df["idx"].values]
    ensure_dir(os.path.dirname(out_csv) or ".")
    df.to_csv(out_csv, index=False)
