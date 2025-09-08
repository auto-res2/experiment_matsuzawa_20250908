"""src/train.py
All model architectures and the main training logic live here.  The file
is intentionally dependency-free with respect to the local project – it
only relies on `src.preprocess` for the data loader and on
`src.evaluate` for the metric / plotting utilities.  This way circular
imports are avoided.
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from ptflops import get_model_complexity_info
from torch_geometric.nn import GCNConv

from .evaluate import (
    accuracy,
    dirichlet_energy,
    generate_figures,
    row_diff,
)
from .preprocess import load_dataset, seed_everything

__all__ = [
    "VanillaGCN",
    "SmuSHController",
    "SmuSHGCN",
    "GNNExperiment",
]


# ============================================================================
#  Models
# ============================================================================
class SmuSHController(nn.Module):
    """Light-weight two-layer MLP that outputs a scalar α∈(0,1) for each node."""

    def __init__(self, in_dim: int, hidden: int = 128, eps: float = 1e-3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self._sigmoid = nn.Sigmoid()
        self.eps = eps

    def forward(self, z: torch.Tensor) -> torch.Tensor:  # (N, in_dim) → (N,)
        a = self._sigmoid(self.net(z)).squeeze(-1)
        # strictly clamp to avoid vanishing edge-weights and numerical blow-ups
        return self.eps + (1.0 - 2 * self.eps) * a


class VanillaGCN(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, depth: int):
        if depth < 1:
            raise ValueError("Depth must be ≥1")
        super().__init__()
        self.convs: nn.ModuleList[nn.Module] = nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hidden, add_self_loops=False, cached=False))
        for _ in range(depth - 2):
            self.convs.append(GCNConv(hidden, hidden, add_self_loops=False, cached=False))
        self.convs.append(GCNConv(hidden, out_dim, add_self_loops=False, cached=False))

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, edge_index, edge_weight=None):  # type: ignore[override]
        for conv in self.convs[:-1]:
            x = conv(x, edge_index, edge_weight=edge_weight)
            x = self.act(x)
            x = self.dropout(x)
        x = self.convs[-1](x, edge_index, edge_weight=edge_weight)
        return x


class SmuSHGCN(nn.Module):
    """GCN enhanced with the SmuSH hyper-network controller."""

    def __init__(
        self,
        in_dim: int,
        hidden: int,
        out_dim: int,
        depth: int,
        struc_dim: int,
    ):
        if depth < 1:
            raise ValueError("Depth must be ≥1")
        super().__init__()
        self.depth = depth
        self.eps = 1e-4

        self.convs: nn.ModuleList[nn.Module] = nn.ModuleList()
        self.controllers: nn.ModuleList[nn.Module] = nn.ModuleList()
        # first layer – raw feature input
        self.convs.append(GCNConv(in_dim, hidden, add_self_loops=False, cached=False))
        self.controllers.append(SmuSHController(hidden + struc_dim + 1))
        # hidden layers
        for _ in range(depth - 2):
            self.convs.append(GCNConv(hidden, hidden, add_self_loops=False, cached=False))
            self.controllers.append(SmuSHController(hidden + struc_dim + 1))
        # final classifier layer
        self.convs.append(GCNConv(hidden, out_dim, add_self_loops=False, cached=False))
        self.controllers.append(SmuSHController(out_dim + struc_dim + 1))

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(self, x, edge_index, struc_feat):  # type: ignore[override]
        history = torch.zeros(x.size(0), 1, device=x.device)
        for idx, (conv, ctrl) in enumerate(zip(self.convs, self.controllers)):
            alpha_in = torch.cat([x, struc_feat, history], dim=-1)
            alpha = ctrl(alpha_in)  # (N,)
            edge_weight = torch.clamp(alpha[edge_index[0]] * alpha[edge_index[1]], min=self.eps)

            x_prev = x
            x = conv(x, edge_index, edge_weight=edge_weight)
            if idx != len(self.convs) - 1:
                x = self.act(x)
                x = self.dropout(x)
            # update history
            history = torch.norm(x - x_prev, p=2, dim=-1, keepdim=True)
        return x, alpha.detach()


# ============================================================================
#  Experiment harness
# ============================================================================
class GNNExperiment:
    """Thin wrapper that bundles data-loading, training and evaluation."""

    def __init__(self, cfg: Dict[str, Any], results_dir: Path, images_dir: Path):
        self.cfg = cfg
        self._results_dir = results_dir
        self._images_dir = images_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ------------------------------------------------------------------
        # Data – loaded immediately so we can already fail fast on download
        # ------------------------------------------------------------------
        self.data, self.struct_feat = load_dataset(cfg["dataset"], self.device)

        # ------------------------------------------------------------------
        #  Build model & optimiser
        # ------------------------------------------------------------------
        self.model = self._build_model().to(self.device)
        self.optimiser = optim.AdamW(
            self.model.parameters(),
            lr=float(cfg["optim"]["lr"]),
            weight_decay=float(cfg["optim"]["weight_decay"]),
        )
        self.criterion = nn.CrossEntropyLoss()
        self.scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    # ------------------------------------------------------------------
    # private helpers
    # ------------------------------------------------------------------
    def _build_model(self) -> nn.Module:
        mcfg = self.cfg["model"]
        in_dim = self.data.x.size(1)
        hidden = int(mcfg.get("hidden", 256))
        out_dim = int(self.data.y.max().item() + 1)
        depth = int(mcfg["depth"])

        if mcfg["type"] == "vanilla_gcn":
            return VanillaGCN(in_dim, hidden, out_dim, depth)
        if mcfg["type"] == "smush_gcn":
            return SmuSHGCN(in_dim, hidden, out_dim, depth, self.struct_feat.size(1))
        raise ValueError(f"Unknown model type: {mcfg['type']}")

    # ------------------------------------------------------------------
    # public API – one call does the entire run and returns a result-dict
    # ------------------------------------------------------------------
    def run(self) -> Dict[str, Any]:
        seed_everything(int(self.cfg["seed"]))
        max_epochs = int(self.cfg["optim"]["epochs"])
        patience = int(self.cfg["optim"].get("patience", 50))
        lambda_cons = float(self.cfg.get("lambda_cons", 0.0))

        history: Dict[str, List[float]] = {
            "train_acc": [],
            "val_acc": [],
            "test_acc": [],
            "row_diff": [],
            "energy_ratio": [],
        }

        E0 = dirichlet_energy(self.data.x, self.data.edge_index)
        best_val, best_state, best_epoch = -1.0, None, 0

        for epoch in range(max_epochs):
            # ------------------ training ------------------
            self.model.train()
            self.optimiser.zero_grad()
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                if isinstance(self.model, SmuSHGCN):
                    out, _ = self.model(self.data.x, self.data.edge_index, self.struct_feat)
                else:
                    out = self.model(self.data.x, self.data.edge_index)
                loss = self.criterion(
                    out[self.data.train_mask], self.data.y[self.data.train_mask]
                )
                if lambda_cons > 0:
                    rd = row_diff(out, self.data.y)
                    energy_ratio = dirichlet_energy(out, self.data.edge_index) / (E0 + 1e-9)
                    penalty = (0.3 - rd).clamp(min=0) + (energy_ratio - 0.4).clamp(min=0)
                    loss = loss + lambda_cons * penalty
            self.scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimiser)
            self.scaler.update()

            # ------------------ evaluation ------------------
            self.model.eval()
            with torch.no_grad():
                if isinstance(self.model, SmuSHGCN):
                    out, _ = self.model(self.data.x, self.data.edge_index, self.struct_feat)
                else:
                    out = self.model(self.data.x, self.data.edge_index)

            train_acc = accuracy(out[self.data.train_mask], self.data.y[self.data.train_mask])
            val_acc = accuracy(out[self.data.val_mask], self.data.y[self.data.val_mask])
            test_acc = accuracy(out[self.data.test_mask], self.data.y[self.data.test_mask])
            rd = row_diff(out, self.data.y)
            er = dirichlet_energy(out, self.data.edge_index) / (E0 + 1e-9)

            for k, v in zip(
                ["train_acc", "val_acc", "test_acc", "row_diff", "energy_ratio"],
                [train_acc, val_acc, test_acc, rd, er],
            ):
                history[k].append(float(v))

            # early stopping --------------------------------------------------
            if val_acc > best_val:
                best_val = val_acc
                best_state = self.model.state_dict()
                best_epoch = epoch
            if epoch - best_epoch > patience:
                break

        # restore best checkpoint -------------------------------------------
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()

        with torch.no_grad():
            if isinstance(self.model, SmuSHGCN):
                final_out, _ = self.model(self.data.x, self.data.edge_index, self.struct_feat)
            else:
                final_out = self.model(self.data.x, self.data.edge_index)
        final_test = accuracy(final_out[self.data.test_mask], self.data.y[self.data.test_mask])

        # complexity (can fail on unknown ops → fall back gracefully)
        try:
            flops, params = get_model_complexity_info(
                self.model, (self.data.num_node_features,), print_per_layer_stat=False
            )
        except Exception:
            flops, params = "N/A", "N/A"

        # figures -----------------------------------------------------------
        fig_files = generate_figures(history, self.cfg, self._images_dir)

        result = {
            "dataset": self.cfg["dataset"],
            "model": self.cfg["model"],
            "seed": self.cfg["seed"],
            "best_val_acc": best_val,
            "test_acc": final_test,
            "row_diff_last": history["row_diff"][-1],
            "energy_ratio_last": history["energy_ratio"][-1],
            "epochs_ran": len(history["train_acc"]),
            "flops": flops,
            "params": params,
            "figures": fig_files,
        }
        return result
