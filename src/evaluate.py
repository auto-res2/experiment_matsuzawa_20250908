"""evaluate.py – metrics used by the experiments"""
from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = [
    "accuracy",
    "row_diff",
    "dirichlet_energy",
]


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    """Top-1 accuracy in percent."""
    return (logits.argmax(dim=1) == y).float().mean().item() * 100.0


def row_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    """Simplified PairNorm row-difference statistic (higher = less smooth)."""
    with torch.no_grad():
        intra, inter = [], []
        classes = torch.unique(y)
        for c in classes:
            idx = (y == c).nonzero(as_tuple=True)[0]
            if idx.numel() < 2:
                continue
            intra.append(torch.cdist(x[idx], x[idx]).mean())
        for i, c1 in enumerate(classes):
            idx1 = (y == c1).nonzero(as_tuple=True)[0]
            for c2 in classes[i + 1 :]:
                idx2 = (y == c2).nonzero(as_tuple=True)[0]
                inter.append(torch.cdist(x[idx1], x[idx2]).mean())
        if not intra or not inter:
            return 0.0
        return (torch.stack(intra).mean() / torch.stack(inter).mean()).item()


def dirichlet_energy(x: torch.Tensor, edge_index: torch.Tensor) -> float:
    """Mean squared neighbour-difference (Dirichlet energy)."""
    row, col = edge_index
    diff = (x[row] - x[col]).pow(2).sum(dim=1)
    return diff.mean().item()
