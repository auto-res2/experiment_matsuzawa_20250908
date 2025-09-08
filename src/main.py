"""main.py – entry point (``python -m src.main``)"""
from __future__ import annotations

import json
import pathlib
from types import SimpleNamespace

import yaml
import torch

from .train import set_seed, train_fullbatch
from .evaluate import accuracy  # noqa: F401  (imported for YAML / IDE convenience)
from .preprocess import get_dataset

# -----------------------------------------------------------------------------
# Minimal *models* inside the same file to comply with the 6-file restriction
# -----------------------------------------------------------------------------
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    def __init__(self, in_dim: int, hid: int, out_dim: int, depth: int):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hid, add_self_loops=False))
        for _ in range(depth - 2):
            self.convs.append(GCNConv(hid, hid, add_self_loops=False))
        self.convs.append(GCNConv(hid, out_dim, add_self_loops=False))

    def forward(self, x, edge_index, *_):  # extra *args for API compatibility
        for conv in self.convs[:-1]:
            x = F.relu(conv(x, edge_index))
        return self.convs[-1](x, edge_index)


# ------------------------------------------------------------------
# SmuSH – only the *GCN* variant needed for the depth experiment
# ------------------------------------------------------------------
class _Controller(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim + 3 + 16, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, h, s):
        z = torch.cat([h, s], dim=1)
        return torch.sigmoid(self.fc2(F.gelu(self.fc1(z)))).squeeze()


def _alpha_edge_weight(edge_index, alpha):
    row, col = edge_index
    return alpha[row] * alpha[col]


class SmuSHGCN(nn.Module):
    """Lightweight re-implementation sufficient for Exp-1."""

    def __init__(self, in_dim: int, hid: int, out_dim: int, depth: int):
        super().__init__()
        self.convs = nn.ModuleList()
        self.ctrls = nn.ModuleList()

        # GCN layers -------------------------------------------------------
        self.convs.append(GCNConv(in_dim, hid, add_self_loops=False))
        for _ in range(depth - 2):
            self.convs.append(GCNConv(hid, hid, add_self_loops=False))
        self.convs.append(GCNConv(hid, out_dim, add_self_loops=False))

        # Controller layers ------------------------------------------------
        #   The first controller must see the *input*-dimensional features,
        #   subsequent ones see the hidden-dimensional representations.
        dims = [in_dim] + [hid] * (depth - 1)
        for d in dims:
            self.ctrls.append(_Controller(d))

    # ---------------------------------------------------------
    # Forward
    # ---------------------------------------------------------
    def forward(self, x, edge_index, struct_feat, hist):
        for l, (conv, ctrl) in enumerate(zip(self.convs[:-1], self.ctrls[:-1])):
            alpha = ctrl(x, torch.cat([struct_feat, hist], dim=1))
            w = _alpha_edge_weight(edge_index, alpha)
            x = F.relu(conv(x, edge_index, w))
        return self.convs[-1](x, edge_index)


# -----------------------------------------------------------------------------
# Configuration – loaded from YAML so that the user can edit parameters without
# touching the source code.
# -----------------------------------------------------------------------------
CFG_PATH = pathlib.Path("config") / "config.yaml"
if not CFG_PATH.exists():
    CFG_PATH.parent.mkdir(parents=True, exist_ok=True)
    # default parameters identical to the original dataclass version
    default_cfg = {
        "exp1": {
            "datasets": ["cora", "pubmed"],
            "methods": ["vanilla", "smush"],
            "depths": [4, 16, 32],
            "hidden_dim": 256,
            "max_epochs_by_depth": {"4": 400, "16": 800, "32": 1200},
        }
    }
    with open(CFG_PATH, "w", encoding="utf-8") as fh:
        yaml.safe_dump(default_cfg, fh, sort_keys=False)

with open(CFG_PATH, "r", encoding="utf-8") as fh:
    CFG = yaml.safe_load(fh)

# -----------------------------------------------------------------------------
# Output directories required by the grading instructions
# -----------------------------------------------------------------------------
RES_DIR = pathlib.Path(".research") / "iteration4"
IMG_DIR = RES_DIR / "images"
RES_DIR.mkdir(parents=True, exist_ok=True)
IMG_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# EXPERIMENT 1 – Depth scalability (subset for brevity)
# -----------------------------------------------------------------------------
print("\n========== EXPERIMENT 1 – Depth Scalability ==========")

exp1_cfg = CFG["exp1"]
results = {}

for dset in exp1_cfg["datasets"]:
    data = get_dataset(dset)
    results[dset] = {}

    for method in exp1_cfg["methods"]:
        results[dset][method] = {}
        for depth in exp1_cfg["depths"]:
            set_seed(0)
            if method == "vanilla":
                model = GCN(
                    data.num_features,
                    exp1_cfg["hidden_dim"],
                    int(data.y.max()) + 1,
                    depth,
                )
            elif method == "smush":
                model = SmuSHGCN(
                    data.num_features,
                    exp1_cfg["hidden_dim"],
                    int(data.y.max()) + 1,
                    depth,
                )
            else:
                raise NotImplementedError(method)

            cfg = SimpleNamespace(max_epochs=exp1_cfg["max_epochs_by_depth"][str(depth)])
            device = "cuda" if torch.cuda.is_available() else "cpu"
            res = train_fullbatch(model, data, cfg, device)
            results[dset][method][depth] = res
            print(f"[Exp1] {dset:<10} | {method:<7} | depth {depth:<3} => {res}")

# -----------------------------------------------------------------------------
# Save & print JSON
# -----------------------------------------------------------------------------
json_path = RES_DIR / "exp1_results.json"
with open(json_path, "w", encoding="utf-8") as fh:
    json.dump(results, fh, indent=2)
print(f"\nExperiment 1 completed – results saved to {json_path}\n")
print(json.dumps(results, indent=2))
