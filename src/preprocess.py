"""preprocess.py – dataset download / preprocessing utilities."""
from __future__ import annotations

import pathlib
from functools import lru_cache
from typing import Tuple, Optional

import torch
from torch_geometric.datasets import Planetoid, WikipediaNetwork, Texas, Chameleon, Squirrel
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import remove_self_loops, add_self_loops
from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T

DATA_ROOT = pathlib.Path("data")
DATA_ROOT.mkdir(parents=True, exist_ok=True)

__all__ = ["get_dataset"]


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def _zscore(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mean = x[mask].mean(dim=0, keepdim=True)
    std = x[mask].std(dim=0, keepdim=True).clamp_min_(1e-9)
    return (x - mean) / std


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

@lru_cache(maxsize=None)
def get_dataset(name: str):  # noqa: C901 (complexity – mirrors original code)
    """Return a torch-geometric *Data* object with train/val/test masks."""
    name = name.lower()
    root = DATA_ROOT / name

    if name in {"cora", "citeseer", "pubmed"}:
        dataset = Planetoid(root=str(root), name=name.capitalize(), transform=NormalizeFeatures())
        data = dataset[0]

    elif name in {"chameleon", "squirrel"}:
        dataset = WikipediaNetwork(root=str(root), name=name, transform=NormalizeFeatures())
        data = dataset[0]

    elif name == "texas":
        dataset = Texas(root=str(root))
        data = dataset[0]

    elif name == "ogbn-arxiv":
        dataset = PygNodePropPredDataset(name="ogbn-arxiv", root=str(root))
        split = dataset.get_idx_split()
        data = dataset[0]
        data.y = data.y.squeeze()
        # Build boolean masks
        n = data.num_nodes
        data.train_mask = torch.zeros(n, dtype=torch.bool)
        data.val_mask = torch.zeros(n, dtype=torch.bool)
        data.test_mask = torch.zeros(n, dtype=torch.bool)
        data.train_mask[split["train"]] = True
        data.val_mask[split["valid"]] = True
        data.test_mask[split["test"]] = True
        data = T.ToSparseTensor()(data)

    elif name == "ogbn-products":
        dataset = PygNodePropPredDataset(name="ogbn-products", root=str(root))
        split = dataset.get_idx_split()
        data = dataset[0]
        data.y = data.y.squeeze()
        n = data.num_nodes
        data.train_mask = torch.zeros(n, dtype=torch.bool)
        data.val_mask = torch.zeros(n, dtype=torch.bool)
        data.test_mask = torch.zeros(n, dtype=torch.bool)
        data.train_mask[split["train"]] = True
        data.val_mask[split["valid"]] = True
        data.test_mask[split["test"]] = True
        data = T.ToSparseTensor()(data)
    else:
        raise ValueError(f"Unknown dataset '{name}'.")

    # ---------------------------------------------------------
    # Standard feature normalisation and self-loop handling
    # ---------------------------------------------------------
    data.x = _zscore(data.x, data.train_mask)
    data.edge_index, _ = remove_self_loops(data.edge_index)
    data.edge_index, _ = add_self_loops(data.edge_index)

    return data
