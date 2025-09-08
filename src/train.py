# src/train.py
"""Training-related utilities: configuration loader, backbone factory,  
continual-learning methods (CLIPON), JSON logger and high-level Trainer."""
from __future__ import annotations

import json, random, time
from pathlib import Path
from typing import Dict, Any, Tuple

import yaml
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models as tv_models
import timm

from src.preprocess import (get_train_transform, get_test_transform,
                            build_continual_dataset)
from src.evaluate import evaluate_model

# -----------------------------------------------------------------------------
#  CONFIG HELPERS
# -----------------------------------------------------------------------------
class CfgNode(dict):
    """Very light-weight Dict→object view so we can write `cfg.foo`."""
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e
    __setattr__ = dict.__setitem__


def load_cfg(path: str | Path = "config/config.yaml") -> CfgNode:
    """Load YAML configuration file."""
    with open(path, "r") as f:
        return CfgNode(yaml.safe_load(f))

# -----------------------------------------------------------------------------
#  LOGGING
# -----------------------------------------------------------------------------
class StdJSONLogger:
    """Write every logged event as JSON – both to stdout and a .jsonl file."""
    def __init__(self, out_path: Path):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(out_path, "w")

    def log(self, d: Dict[str, Any]):
        s = json.dumps(d, default=str)
        print(s, flush=True)
        self._f.write(s + "\n"); self._f.flush()

    def close(self):
        self._f.close()

# -----------------------------------------------------------------------------
#  BACKBONE FACTORY
# -----------------------------------------------------------------------------

def build_backbone(name: str) -> Tuple[nn.Module, int]:
    """Return (`torch.nn.Module`, feature_dim)."""
    if name == "resnet18":
        m = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)
        feat_dim = m.fc.in_features
        m.fc = nn.Identity()
    elif name == "vit_t16":
        m = timm.create_model("vit_tiny_patch16_224.augreg_in21k_ft_in1k", pretrained=True, num_classes=0)
        feat_dim = m.num_features
    elif name == "resnet18_frozen":
        m = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)
        for p in m.parameters():
            p.requires_grad_(False)
        m.fc = nn.Identity(); feat_dim = m.fc.in_features
    else:
        raise ValueError(name)
    return m, feat_dim

# -----------------------------------------------------------------------------
#  CLIP-ON – simplified implementation
# -----------------------------------------------------------------------------
class BPQ(nn.Module):
    """Binary Product Quantisation module (feature-level)."""
    def __init__(self, dim: int = 512, M: int = 8, bits: int = 6):
        super().__init__()
        assert dim % M == 0, "dim must be divisible by M"
        self.M, self.bits, self.sub = M, bits, dim // M
        self.register_parameter("codebooks", nn.Parameter(torch.randn(M, 2 ** bits, self.sub)))

    # no-grad helpers ----------------------------------------------------------
    @torch.no_grad()
    def encode(self, z: torch.Tensor) -> torch.Tensor:
        b = z.size(0)
        z_view = z.view(b, self.M, 1, self.sub)           # (B,M,1,sub)
        d2 = ((z_view - self.codebooks) ** 2).sum(-1)     # (B,M,K)
        return d2.argmin(-1)                              # (B,M)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        emb = self.codebooks[torch.arange(self.M).unsqueeze(0), codes]  # (B,M,sub)
        return emb.view(codes.size(0), -1)


class CLIPON(nn.Module):
    """Hybrid rehearsal/compression learner (greatly simplified)."""
    def __init__(self, backbone_name: str, num_classes: int, feat_dim: int, cfg: CfgNode):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.backbone, _ = build_backbone(backbone_name)
        self.head = nn.Linear(feat_dim, num_classes)
        self.bpq = BPQ(dim=feat_dim, bits=cfg.get("bits", 6))

        # small FIFO buffer ---------------------------------------------------
        self.buffer_size = int(cfg.get("fifo", 128))
        self.register_buffer("buf_codes", torch.empty(0, dtype=torch.long))
        self.register_buffer("buf_logits", torch.empty(0, num_classes))

        self.to(self.device)
        self.opt = torch.optim.SGD(self.parameters(), lr=cfg.optimiser.lr,
                                   momentum=cfg.optimiser.momentum,
                                   weight_decay=cfg.optimiser.weight_decay)

    # ---------------------------------------------------------------------
    def _add_to_buffer(self, feats: torch.Tensor, logits: torch.Tensor):
        codes = self.bpq.encode(feats.detach().cpu())
        self.buf_codes = torch.cat([self.buf_codes, codes])[-self.buffer_size:]
        self.buf_logits = torch.cat([self.buf_logits, logits.detach().cpu()])[-self.buffer_size:]

    # ---------------------------------------------------------------------
    def forward(self, x: torch.Tensor):
        z = self.backbone(x)
        return self.head(z)

    # ---------------------------------------------------------------------
    def observe(self, x: torch.Tensor, y: torch.Tensor, task_id: torch.Tensor):
        self.train()
        x, y = x.to(self.device), y.to(self.device)
        z = self.backbone(x)
        logits = self.head(z)
        loss_cls = F.cross_entropy(logits, y)

        # replay ----------------------------------------------------------------
        if len(self.buf_codes):
            idx = torch.randperm(len(self.buf_codes))[: x.size(0)]
            replay_z = self.bpq.decode(self.buf_codes[idx].to(self.device)).detach()
            replay_logits = self.head(replay_z)
            target = self.buf_logits[idx].to(self.device).argmax(-1)
            loss_replay = F.cross_entropy(replay_logits, target)
            loss = loss_cls + 0.7 * loss_replay
        else:
            loss = loss_cls

        self.opt.zero_grad(); loss.backward(); self.opt.step()

        self._add_to_buffer(z, F.one_hot(y, num_classes=self.head.out_features).float())
        return loss.item()

# -----------------------------------------------------------------------------
#  TRAINER
# -----------------------------------------------------------------------------
class Trainer:
    def __init__(self, cfg: CfgNode, logger: StdJSONLogger):
        self.cfg, self.logger = cfg, logger

    # ---------------------------------------------------------------------
    @staticmethod
    def _seed_all(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # ---------------------------------------------------------------------
    def run_task_sequence(self, method_name: str, backbone: str,
                          dataset_name: str, seed: int):
        self._seed_all(seed)

        # 1) build dataset --------------------------------------------------
        train_ds, test_ds, num_classes, img_res, is_cifar, task_map = build_continual_dataset(dataset_name)
        train_tf = get_train_transform(img_res, is_cifar)
        test_tf = get_test_transform(img_res, is_cifar)

        def _collate(batch):
            tids, imgs, ys = zip(*batch)
            imgs = torch.stack([train_tf(i) for i in imgs])
            return torch.tensor(tids), imgs, torch.tensor(ys)

        train_loader = DataLoader(train_ds, batch_size=self.cfg.batch_size, shuffle=True,
                                  num_workers=self.cfg.num_workers, collate_fn=_collate)

        def _test_collate(batch):
            tids, imgs, ys = zip(*batch)
            return (torch.tensor(tids),
                    torch.stack([test_tf(i) for i in imgs]),
                    torch.tensor(ys))

        test_loader = DataLoader(test_ds, batch_size=512, shuffle=False,
                                 num_workers=self.cfg.num_workers, collate_fn=_test_collate)

        # 2) instantiate method -------------------------------------------
        feat_dim = 512 if backbone.startswith("resnet") else 192
        if method_name == "clipon":
            model = CLIPON(backbone, num_classes=num_classes, feat_dim=feat_dim, cfg=self.cfg)
        else:
            raise NotImplementedError(method_name)

        # 3) train loop ----------------------------------------------------
        t0 = time.time()
        for tids, imgs, ys in train_loader:
            model.observe(imgs, ys, tids)

        # 4) evaluate ------------------------------------------------------
        acc, cm = evaluate_model(model, test_loader)
        runtime = time.time() - t0

        # 5) save + log ----------------------------------------------------
        res = dict(dataset=dataset_name, method=method_name, seed=seed,
                   avg_accuracy=acc, runtime_s=runtime)
        out_dir = Path(".research/iteration1")
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / f"{dataset_name}_{method_name}_{seed}.json"
        with open(json_path, "w") as f:
            json.dump(res, f, indent=2)

        self.logger.log({"phase": "done", **res})

        # confusion matrix figure (saved for later analysis) ---------------
        from matplotlib import pyplot as plt
        plt.figure(figsize=(6, 5))
        plt.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.colorbar()
        plt.tight_layout()
        img_path = out_dir / f"confusion_{dataset_name}_{method_name}.pdf"
        plt.savefig(img_path, bbox_inches="tight")
        plt.close()
