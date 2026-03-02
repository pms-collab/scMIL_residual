from __future__ import annotations
import os
import numpy as np
import pandas as pd
from ..utils import ensure_dir

def donor_overlap_table(donor_id: np.ndarray, split: np.ndarray) -> pd.DataFrame:
    # split: 0/1/2
    df = pd.DataFrame({"donor_id": donor_id, "split": split})
    piv = pd.crosstab(df["donor_id"], df["split"])
    # donors with >1 nonzero splits are leakage
    nonzero = (piv > 0).sum(axis=1)
    leaked = piv[nonzero > 1].copy()
    leaked.columns = [f"split_{c}" for c in leaked.columns]
    return leaked.reset_index()

def class_count_by_split(y: np.ndarray, split: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame({"y": y, "split": split})
    return pd.crosstab(df["split"], df["y"]).reset_index()

def write_audit(out_dir: str, donor_id: np.ndarray, split: np.ndarray, y: np.ndarray) -> None:
    ensure_dir(out_dir)
    leak = donor_overlap_table(donor_id, split)
    leak.to_csv(os.path.join(out_dir, "donor_overlap.csv"), index=False)
    cc = class_count_by_split(y, split)
    cc.to_csv(os.path.join(out_dir, "class_counts.csv"), index=False)
