"""src/evaluate.py
All evaluation and plotting helper functions live here.
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple

import torch
from sklearn.metrics import confusion_matrix

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402, isort:skip
import seaborn as sns  # noqa: E402, isort:skip

PLOT_STYLE = dict(marker="o", markersize=4, linewidth=1.4)

__all__ = [
    "plot_curve",
    "evaluate_classifier",
]


def plot_curve(values: List[float], title: str, ylabel: str, save_path: Path) -> None:
    """Save a simple line plot (.pdf)."""
    plt.figure(figsize=(6, 4))
    sns.set(style="whitegrid")
    plt.plot(range(1, len(values) + 1), values, **PLOT_STYLE)
    for i, v in enumerate(values, 1):
        plt.text(i, v, f"{v:.2f}", fontsize=6, ha="center", va="bottom")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()


def evaluate_classifier(module, dataloader) -> Tuple[float, List[List[int]]]:
    """Return overall accuracy and the confusion matrix for *module* on *dataloader*."""
    module.eval()
    preds, gts = [], []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = module.to(device)
    with torch.no_grad():
        for xb, yb in dataloader:
            xb = xb.to(device, non_blocking=True)
            logits = module(xb)
            preds.append(logits.argmax(1).cpu())
            gts.append(yb)
    preds = torch.cat(preds)
    gts = torch.cat(gts)
    acc = (preds == gts).float().mean().item()
    cm = confusion_matrix(gts.numpy(), preds.numpy()).tolist()
    return acc, cm
