"""src/preprocess.py
Data-loading utilities, YAML helpers and small misc functions.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset

__all__ = [
    "save_yaml",
    "print_json",
    "build_dataloaders",
]


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------

def save_yaml(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(obj, f)


def print_json(obj: Dict[str, Any]) -> None:
    print(json.dumps(obj, indent=2, sort_keys=False))


# -----------------------------------------------------------------------------
# Dataset helpers
# -----------------------------------------------------------------------------


class HFDataset(torch.utils.data.Dataset):
    """Wraps a 🤗 *Dataset* to behave like a *torchvision* dataset."""

    def __init__(self, ds, transforms):
        self.ds = ds
        self.t = transforms

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        img = item["image"]
        label = item.get("label", item.get("labels"))
        return self.t(img), torch.tensor(label, dtype=torch.long)


def _count_classes(ds) -> int:
    """Return the number of distinct class labels in *ds* without materialising the whole dataset."""
    try:
        # Most datasets expose their "features" info
        features = ds.features
        if "label" in features and hasattr(features["label"], "num_classes"):
            return features["label"].num_classes
    except Exception:
        pass
    # Fallback (may be slow for large datasets, but only used when metadata missing)
    labels = set()
    for sample in ds:
        labels.add(sample.get("label", sample.get("labels")))
        if len(labels) > 10000:  # sanity-cap to avoid endless loops on corrupted datasets
            break
    return len(labels)


def build_dataloaders(dataset_cfg: Dict[str, Any], batch_size: int, num_workers: int):
    """Construct train/val dataloaders according to *dataset_cfg*.  Strictly raises
    on any download / access error – mandated by the spec.
    """
    try:
        if dataset_cfg["name"] == "imagenet1k":
            # The public HF mirror exposes "train" and "validation" splits directly.
            ds_train = load_dataset(dataset_cfg["hf_id"], split="train", streaming=False)
            ds_val = load_dataset(dataset_cfg["hf_id"], split="validation", streaming=False)
        else:
            ds_train = load_dataset(dataset_cfg["hf_id"], split=dataset_cfg["split"][0])
            ds_val = (
                load_dataset(dataset_cfg["hf_id"], split=dataset_cfg["split"][1])
                if len(dataset_cfg.get("split", [])) > 1
                else None
            )
    except Exception as e:  # pragma: no cover – fail hard according to spec
        raise RuntimeError(
            f"Dataset '{dataset_cfg['name']}' unavailable – NO-FALLBACK policy enforced. Details: {e}"
        )

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    t_train = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomResizedCrop(224),
        transforms.RandAugment(2, 9),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    t_val = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    dl_train = DataLoader(
        HFDataset(ds_train, t_train),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    dl_val = None
    if ds_val is not None:
        dl_val = DataLoader(
            HFDataset(ds_val, t_val),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    # Prefer metadata when possible – fall back to manual counting otherwise.
    num_classes = _count_classes(ds_train)
    return dl_train, dl_val, num_classes
