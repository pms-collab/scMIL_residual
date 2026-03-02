from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

SPLIT_MAP = {"train": 0, "val": 1, "test": 2}

@dataclass
class Bags:
    X: np.ndarray         # [N, CAP, D] float32
    mask: np.ndarray      # [N, CAP] bool
    y: np.ndarray         # [N] int64 (0/1/2)
    split: np.ndarray     # [N] int8 (0/1/2)
    sample_id: np.ndarray # [N] str
    donor_id: np.ndarray  # [N] str
    idx_train: np.ndarray
    idx_val: np.ndarray
    idx_test: np.ndarray

def load_bags_npz(path: str) -> Bags:
    z = np.load(path, allow_pickle=True)
    return Bags(
        X=z["X"].astype(np.float32),
        mask=z["mask"].astype(bool),
        y=z["y"].astype(np.int64),
        split=z["split"].astype(np.int8),
        sample_id=z["sample_id"].astype(str),
        donor_id=z["donor_id"].astype(str),
        idx_train=z["idx_train"].astype(np.int64),
        idx_val=z["idx_val"].astype(np.int64),
        idx_test=z["idx_test"].astype(np.int64),
    )

class BagDataset(Dataset):
    def __init__(self, bags: Bags, indices: np.ndarray):
        self.bags=bags
        self.indices=indices

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        j = int(self.indices[i])
        X = torch.from_numpy(self.bags.X[j])
        mask = torch.from_numpy(self.bags.mask[j])
        y = torch.tensor(self.bags.y[j], dtype=torch.long)
        split = torch.tensor(self.bags.split[j], dtype=torch.long)
        return {"X": X, "mask": mask, "y": y, "split": split, "idx": torch.tensor(j, dtype=torch.long)}

def make_loaders(bags: Bags, batch_size: int, num_workers: int = 0) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = BagDataset(bags, bags.idx_train)
    val_ds   = BagDataset(bags, bags.idx_val)
    test_ds  = BagDataset(bags, bags.idx_test)

    def _dl(ds, shuffle):
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                          pin_memory=True, drop_last=False)
    return _dl(train_ds, True), _dl(val_ds, False), _dl(test_ds, False)
