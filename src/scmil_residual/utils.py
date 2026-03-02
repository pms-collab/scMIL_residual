from __future__ import annotations
import os, json, random
from dataclasses import dataclass, asdict
from typing import Any, Dict
import numpy as np
import torch

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path

def save_json(obj: Any, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    out={}
    for k,v in batch.items():
        if torch.is_tensor(v):
            out[k]=v.to(device, non_blocking=True)
        else:
            out[k]=v
    return out

def sigmoid_safe(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x.clamp(-30, 30))

def noisy_or(p: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    # p in [0,1]
    p = p.clamp(eps, 1 - eps)
    return 1.0 - torch.prod(1.0 - p, dim=dim)
