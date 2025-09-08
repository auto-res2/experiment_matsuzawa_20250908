"""src/evaluate.py
Evaluation utilities – metrics as well as helper routines for visualisation
and reporting.  All functions here are intentionally stateless and therefore
safe to import from any other module without causing side effects.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import torch

matplotlib.use("Agg")  # head-less back-end for clusters / CI
sns.set_style("whitegrid")

__all__ = [
    "accuracy",
    "row_diff",
    "dirichlet_energy",
    "generate_figures",
]


def accuracy(pred: torch.Tensor, y: torch.Tensor) -> float:
    return (pred.argmax(dim=-1) == y).float().mean().item() * 100.0


def row_diff(emb: torch.Tensor, y: torch.Tensor) -> float:
    """Mean pairwise distance between class centroids (PairNorm row-diff)."""

    classes = y.unique().tolist()
    centroids = []
    for c in classes:
        mask = y == c
        if mask.sum() == 0:
            continue
        centroids.append(emb[mask].mean(0, keepdim=True))
    if len(centroids) < 2:
        return 0.0
    centroids = torch.cat(centroids, dim=0)
    diffs = []
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            diffs.append(torch.norm(centroids[i] - centroids[j], p=2))
    return torch.stack(diffs).mean().item()


def dirichlet_energy(x: torch.Tensor, edge_index: torch.Tensor) -> float:
    src, dst = edge_index
    diff = x[src] - x[dst]
    return diff.pow(2).sum(1).mean().item()


# ---------------------------------------------------------------------------
#  Plotting helpers
# ---------------------------------------------------------------------------

def _annotate(ax, values: List[float], fmt: str):
    for idx, val in enumerate(values):
        ax.text(idx, val, fmt.format(val), fontsize=6)


def generate_figures(
    history: Dict[str, List[float]], cfg: Dict, images_dir: Path
) -> List[str]:
    """Save PDF figures for accuracy and row-diff curves.

    Returns
    -------
    List[str]
        File names (not paths) of the generated figures so callers can log them.
    """

    images_dir.mkdir(parents=True, exist_ok=True)
    base = f"{cfg['exp_name']}_{cfg['dataset']}_{cfg['model']['type']}_depth{cfg['model']['depth']}"

    # accuracy -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(history["train_acc"], label="train")
    ax.plot(history["val_acc"], label="val")
    ax.plot(history["test_acc"], label="test")
    _annotate(ax, history["test_acc"], "{:.1f}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.legend()
    acc_name = f"accuracy_{base}.pdf"
    fig.savefig(images_dir / acc_name, bbox_inches="tight")
    plt.close(fig)

    # row-diff --------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(history["row_diff"], label="row-diff")
    _annotate(ax, history["row_diff"], "{:.2f}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Row-Diff")
    ax.legend()
    rd_name = f"rowdiff_{base}.pdf"
    fig.savefig(images_dir / rd_name, bbox_inches="tight")
    plt.close(fig)

    return [acc_name, rd_name]
