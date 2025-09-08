# src/preprocess.py
"""Data acquisition & preprocessing transforms.

Fixes (iteration 6):
1. Removed deprecated ``trust_remote_code`` flag from all ``load_dataset``
   calls (already addressed earlier).
2. **Label-handling bug-fix** – the use of ``or`` for selecting between
   ``fine_label`` and ``label`` dropped valid labels equal to ``0`` (because
   ``0`` is *falsy* in Python).  We now explicitly test for key presence rather
   than truthiness, eliminating the *KeyError: None* observed during
   ``DataLoader`` iteration.
3. Added inline comments for clarity.
"""
from __future__ import annotations

import itertools
from typing import Tuple, Dict

from datasets import load_dataset
from PIL import Image
import numpy as np
from torchvision import transforms

# -----------------------------------------------------------------------------
#  Normalisation statistics
# -----------------------------------------------------------------------------
_cifar_mean, _cifar_std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
_imnet_mean, _imnet_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

# -----------------------------------------------------------------------------
#  Transforms
# -----------------------------------------------------------------------------

def get_train_transform(img_res: int = 224, is_cifar: bool = False):
    mean, std = (_cifar_mean, _cifar_std) if is_cifar else (_imnet_mean, _imnet_std)
    return transforms.Compose([
        transforms.RandomResizedCrop(img_res, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def get_test_transform(img_res: int = 224, is_cifar: bool = False):
    mean, std = (_cifar_mean, _cifar_std) if is_cifar else (_imnet_mean, _imnet_std)
    return transforms.Compose([
        transforms.Resize(img_res),
        transforms.CenterCrop(img_res),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

# -----------------------------------------------------------------------------
#  ContinualDataset wrapper
# -----------------------------------------------------------------------------
class ContinualDataset:
    """HF-datasets wrapper returning ``(task_id, PIL.Image, label)``."""

    def __init__(self, name: str, split: str, task_map: Dict[int, int]):
        self.ds = load_dataset(name, split=split)  # trust_remote_code removed
        self.task_map = task_map

    def __len__(self):
        return len(self.ds)

    # ------------------------------------------------------------------
    # Utility: convert various HF image formats to PIL.Image -------------
    # ------------------------------------------------------------------
    def _ensure_pil(self, img):
        if isinstance(img, Image.Image):
            return img
        if isinstance(img, dict) and "bytes" in img:  # HF image feature (legacy)
            return Image.open(img["bytes"])
        if isinstance(img, np.ndarray):
            return Image.fromarray(img)
        raise TypeError(f"Unsupported image type: {type(img)}")

    # ------------------------------------------------------------------
    def __getitem__(self, idx):
        row = self.ds[idx]

        # ---- image (handle `img` vs. `image` column names) --------------
        img_val = row.get("img") if "img" in row else row.get("image")
        img = self._ensure_pil(img_val)

        # ---- label (handle presence without relying on truthiness) ------
        if "fine_label" in row and row["fine_label"] is not None:
            label = int(row["fine_label"])
        elif "label" in row and row["label"] is not None:
            label = int(row["label"])
        else:
            raise KeyError("No label column found in dataset row.")

        task_id = self.task_map[label]
        return task_id, img, label

# -----------------------------------------------------------------------------
#  public helper to construct one of our benchmark datasets
# -----------------------------------------------------------------------------

def build_continual_dataset(name: str):
    """Return ``(train_ds, test_ds, num_classes, img_res, is_cifar, task_map)``."""
    if name == "split_cifar100":
        # 20 tasks × 5 classes each
        class_order = list(range(100))
        tasks = [class_order[i: i + 5] for i in range(0, 100, 5)]
        task_map = {c: t for t, cls in enumerate(tasks) for c in cls}

        train_ds = ContinualDataset("uoft-cs/cifar100", "train", task_map)
        test_ds  = ContinualDataset("uoft-cs/cifar100", "test", task_map)
        return train_ds, test_ds, 100, 32, True, task_map

    # ------------------------------------------------------------------
    # unsupported dataset – handled upstream (Trainer.run_task_sequence)
    # ------------------------------------------------------------------
    raise NotImplementedError(f"Dataset '{name}' is not implemented.")
