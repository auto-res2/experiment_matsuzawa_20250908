"""src/preprocess.py
Data loading, graph pre-processing and utility helpers that do **not** require
model awareness live here.
"""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Tuple

import networkx as nx
import numpy as np
import torch
from torch_geometric.datasets import Planetoid, WikipediaNetwork, WebKB
from torch_geometric.utils import degree
from ogb.nodeproppred import PygNodePropPredDataset

__all__ = ["seed_everything", "load_dataset"]


# ---------------------------------------------------------------------------
#  Deterministic seeding
# ---------------------------------------------------------------------------

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
#  Dataset loader
# ---------------------------------------------------------------------------

def load_dataset(name: str, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Download (if necessary) and return a processed `Data` object + structural feats."""

    name = name.lower()
    processed_path = Path(os.getenv("PYG_DATA", "~/.pyg")).expanduser()

    if name in {"cora", "citeseer", "pubmed"}:
        ds = Planetoid(root=str(processed_path), name=name.capitalize())
        data = ds[0]
    elif name in {"texas", "wisconsin"}:
        ds = WebKB(root=str(processed_path), name=name.capitalize())
        data = ds[0]
    elif name in {"chameleon", "squirrel"}:
        ds = WikipediaNetwork(root=str(processed_path), name=name.capitalize())
        data = ds[0]
    elif name in {"ogbn-arxiv", "ogbn-products"}:
        ds = PygNodePropPredDataset(name=name)
        data = ds[0]
        split_idx = ds.get_idx_split()
        data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.val_mask = torch.zeros_like(data.train_mask)
        data.test_mask = torch.zeros_like(data.train_mask)
        data.train_mask[split_idx["train"]] = True
        data.val_mask[split_idx["valid"]] = True
        data.test_mask[split_idx["test"]] = True
    else:
        raise RuntimeError(f"Dataset '{name}' is not supported or download failed.")

    if data.x is None:
        raise RuntimeError("Node features are missing – aborting run.")

    # ------------------------------------------------------------------
    #  Structural features (degree, clustering coeff., Ollivier-Ricci curvature)
    # ------------------------------------------------------------------
    deg = degree(data.edge_index[0]).unsqueeze(-1)

    G_nx = nx.Graph()
    G_nx.add_edges_from(data.edge_index.t().cpu().numpy())
    clustering = torch.tensor(list(nx.clustering(G_nx).values())).unsqueeze(-1)

    # optional curvature (fails gracefully if lib not present)
    try:
        from GraphRicciCurvature.OllivierRicci import OllivierRicci

        orc = OllivierRicci(G_nx, alpha=0.5, verbose="ERROR")
        orc.compute_ricci_curvature()
        curvature_vals = [orc.G.nodes[n]["ricciCurvature"] for n in G_nx.nodes()]
        curvature = torch.tensor(curvature_vals).unsqueeze(-1)
    except Exception:
        curvature = torch.zeros_like(clustering)

    struc_feat = torch.cat([deg, clustering, curvature], dim=-1).float()

    return data.to(device), struc_feat.to(device)
