"""train.py – model training utilities
Only the logic required by main.py is kept.  Multi-GPU / DDP code from the
original monolithic script has been removed to keep the reference project as
small and dependency–free as possible while still faithfully reproducing the
original behaviour on a single GPU / CPU.
"""
from __future__ import annotations

import os
import time
from types import SimpleNamespace
from typing import Dict, Any

import torch
import torch.nn.functional as F

from .evaluate import accuracy, row_diff, dirichlet_energy

__all__ = [
    "set_seed",
    "train_fullbatch",
]


def set_seed(seed: int) -> None:
    """Re-seed python, numpy and torch RNGs for reproducibility."""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------------------------------------------------------
# Full-batch supervised node-classification trainer
# -----------------------------------------------------------------------------


def _prepare_hist(num_nodes: int, dim: int, device: torch.device) -> torch.Tensor:
    """Returns an all-zeros history tensor used by the SmuSH controller."""
    return torch.zeros(num_nodes, dim, device=device)


def train_fullbatch(
    model: torch.nn.Module,
    data,
    cfg: Dict[str, Any] | SimpleNamespace,
    device: str | torch.device = "cpu",
) -> Dict[str, float]:
    """Generic full-batch training loop.

    Parameters
    ----------
    model : nn.Module
        Any torch-geometric GNN (GCN, GAT, SmuSHGCN, …)
    data  : Data object from torch-geometric   (already holds masks etc.)
    cfg   : mapping or object that must expose ``max_epochs`` and may expose
             ``lr`` and ``weight_decay``.
    device: "cpu" | "cuda" | torch.device

    Returns
    -------
    Dict[str, float]
        test accuracy and two over-smoothing statistics.
    """

    device = torch.device(device)
    model = model.to(device)
    data = data.to(device)

    lr = getattr(cfg, "lr", 5e-3)
    wd = getattr(cfg, "weight_decay", 5e-4)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    best_val = 0.0
    patience = getattr(cfg, "patience", 50)
    wait = 0

    hist = _prepare_hist(data.num_nodes, 16, device)
    struct_feat = torch.zeros(data.num_nodes, 3, device=device)  # placeholder

    for epoch in range(int(cfg.max_epochs)):
        model.train()
        opt.zero_grad(set_to_none=True)

        out = model(data.x, data.edge_index, struct_feat, hist)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        opt.step()

        # ------------------------------------------------------------------
        # Early-stopping on validation accuracy every 10 epochs
        # ------------------------------------------------------------------
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                logits = model(data.x, data.edge_index, struct_feat, hist)
                val = accuracy(logits[data.val_mask], data.y[data.val_mask])
            if val > best_val:
                best_val = val
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 10
            if wait > patience:
                break

    # ----------------------------------------------------------------------
    # Load best parameters and evaluate on the test split
    # ----------------------------------------------------------------------
    if "best_state" in locals():
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index, struct_feat, hist)
        test_acc = accuracy(logits[data.test_mask], data.y[data.test_mask])
        rd = row_diff(logits, data.y)
        de = dirichlet_energy(logits, data.edge_index)

    return {
        "test_acc": float(test_acc),
        "row_diff": float(rd),
        "dirichlet_energy": float(de),
    }
