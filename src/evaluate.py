# src/evaluate.py
"""Utility functions for model evaluation."""
from __future__ import annotations

from typing import Tuple, List

import torch
from sklearn.metrics import accuracy_score, confusion_matrix


def evaluate_model(model: torch.nn.Module, loader) -> Tuple[float, 'np.ndarray']:
    """Run *model* on *loader* and return ``(accuracy, confusion_matrix)``."""
    model.eval()
    preds: List[int] = []
    gts:   List[int] = []
    with torch.no_grad():
        for _tids, imgs, ys in loader:
            logits = model(imgs.to(model.device))
            p = logits.argmax(-1).cpu()
            preds.extend(p.tolist())
            gts.extend(ys.tolist())

    acc = accuracy_score(gts, preds)
    cm  = confusion_matrix(gts, preds)
    return acc, cm
