# src/preprocess.py
"""Data acquisition & preprocessing transforms."""
from __future__ import annotations

import itertools
from typing import Tuple, Dict

from datasets import load_dataset
from PIL import Image
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
    """HF-datasets wrapper to return (task_id, PIL.Image, label)."""
    def __init__(self, name: str, split: str, task_map: Dict[int, int]):
        self.ds = load_dataset(name, split=split, trust_remote_code=True)
        self.task_map = task_map

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        row = self.ds[idx]
        img = row.get("img") or row.get("image")            # dataset compatibility
        label = row.get("fine_label") or row.get("label")
        task_id = self.task_map[label]
        return task_id, img, label

# -----------------------------------------------------------------------------
#  public helper to construct one of our benchmark datasets
# -----------------------------------------------------------------------------

def build_continual_dataset(name: str):
    """Return (train_ds, test_ds, num_classes, img_res, is_cifar, task_map)."""
    if name == "split_cifar100":
        full = load_dataset("uoft-cs/cifar100", split="train", trust_remote_code=True)
        class_order = list(range(100))
        tasks = [class_order[i : i + 5] for i in range(0, 100, 5)]
        task_map = {c: t for t, cls in enumerate(tasks) for c in cls}
        train_ds = ContinualDataset("uoft-cs/cifar100", "train", task_map)
        test_ds = ContinualDataset("uoft-cs/cifar100", "test", task_map)
        return train_ds, test_ds, 100, 32, True, task_map
    else:
        raise NotImplementedError(name)
