# src/train.py
"""Training-related utilities: configuration loader, backbone factory,
continual-learning methods (CLIPON + baselines), JSON logger and high-level
Trainer.

Key fixes (iteration 5):
1. **Numeric-string safety** – all optimiser hyper-parameters (lr, momentum,
   weight_decay) are explicitly cast to *float* before being passed to
   ``torch.optim.SGD``.  This prevents the previously observed
   ``TypeError: '<' not supported between instances of 'str' and 'float'`` that
   occurred when YAML parsed scientific-notation scalars as strings.
2. **Mandatory research paths** – every artefact is now written to the required
      • JSON results → ``.research/iteration5/``
      • Figures       → ``.research/iteration5/images``
"""
from __future__ import annotations

import json, random, re, time
from pathlib import Path
from typing import Dict, Any, Tuple, List

import yaml
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models as tv_models
import timm

from src.preprocess import (get_train_transform, get_test_transform,
                            build_continual_dataset)
from src.evaluate import evaluate_model

# =============================================================================
#  CONFIG HELPERS
# =============================================================================
class CfgNode(dict):
    """Light-weight *recursive* dict → object wrapper so that nested keys can be
    accessed via attribute notation (``cfg.foo.bar``)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for k, v in list(self.items()):
            if isinstance(v, dict):
                self[k] = CfgNode(v)              # recursion
            elif isinstance(v, list):             # also recurse into lists
                self[k] = [CfgNode(x) if isinstance(x, dict) else x for x in v]

    # attribute access ---------------------------------------------------------
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def _to_cfg(obj):
    """Utility used by ``load_cfg`` to recursively convert standard Python data
    structures (nested ``dict``/``list``) into ``CfgNode`` so that attribute
    access works everywhere."""
    if isinstance(obj, dict):
        return CfgNode({k: _to_cfg(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_cfg(x) for x in obj]
    return obj


def load_cfg(path: str | Path = "config/config.yaml") -> CfgNode:
    """Load YAML configuration file and return a *recursive* ``CfgNode``."""
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return _to_cfg(raw)


# =============================================================================
#  LOGGING
# =============================================================================
class StdJSONLogger:
    """Write every logged event as JSON – both to *stdout* and to ``.jsonl``."""

    def __init__(self, out_path: Path):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(out_path, "w")

    def log(self, d: Dict[str, Any]):
        s = json.dumps(d, default=str)
        print(s, flush=True)
        self._f.write(s + "\n")
        self._f.flush()

    def close(self):
        self._f.close()


# =============================================================================
#  BACKBONE FACTORY
# =============================================================================

def build_backbone(name: str) -> Tuple[nn.Module, int]:
    """Return ``(torch.nn.Module, feature_dim)`` given a *backbone* identifier."""
    if name == "resnet18":
        m = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)
        feat_dim = m.fc.in_features
        m.fc = nn.Identity()
    elif name == "vit_t16":
        m = timm.create_model("vit_tiny_patch16_224.augreg_in21k_ft_in1k",
                              pretrained=True, num_classes=0)
        feat_dim = m.num_features
    elif name == "resnet18_frozen":
        m = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)
        for p in m.parameters():
            p.requires_grad_(False)
        feat_dim = m.fc.in_features
        m.fc = nn.Identity()
    else:
        raise ValueError(name)
    return m, feat_dim


# =============================================================================
#  CONTINUAL-LEARNING METHODS
# =============================================================================
class BPQ(nn.Module):
    """Binary Product Quantisation (feature-space compression) – simplified."""

    def __init__(self, dim: int = 512, M: int = 8, bits: int = 6):
        super().__init__()
        assert dim % M == 0, "dim must be divisible by M"
        self.M, self.bits, self.sub = M, bits, dim // M
        self.register_parameter("codebooks",
                                nn.Parameter(torch.randn(M, 2 ** bits, self.sub)))

    # ------------------------------------------------------------------
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
    """Hybrid rehearsal/compression learner (drastically simplified version)."""

    def __init__(self, backbone_name: str, num_classes: int,
                 feat_dim: int, cfg: CfgNode):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.backbone, _ = build_backbone(backbone_name)
        self.head = nn.Linear(feat_dim, num_classes)
        self.bpq = BPQ(dim=feat_dim, bits=cfg.get("bits", 6))

        # FIFO buffer --------------------------------------------------------
        self.buffer_size = int(cfg.get("fifo", 128))
        self.register_buffer("buf_codes", torch.empty(0, self.bpq.M, dtype=torch.long))
        self.register_buffer("buf_logits", torch.empty(0, num_classes))

        self.to(self.device)
        # ---- optimiser (numeric-string safety) ----------------------------
        opt_cfg = cfg.optimiser
        lr           = float(opt_cfg.lr)
        momentum     = float(opt_cfg.momentum)
        weight_decay = float(opt_cfg.weight_decay)
        self.opt = torch.optim.SGD(self.parameters(), lr=lr,
                                   momentum=momentum,
                                   weight_decay=weight_decay)

    # ------------------------------------------------------------------
    def _add_to_buffer(self, feats: torch.Tensor, logits: torch.Tensor):
        codes = self.bpq.encode(feats.detach().cpu())            # (B,M)
        if self.buf_codes.numel() == 0:
            self.buf_codes = codes
            self.buf_logits = logits.detach().cpu()
        else:
            self.buf_codes = torch.cat([self.buf_codes, codes], dim=0)[-self.buffer_size:]
            self.buf_logits = torch.cat([self.buf_logits, logits.detach().cpu()], dim=0)[-self.buffer_size:]

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor):
        z = self.backbone(x)
        return self.head(z)

    # ------------------------------------------------------------------
    def observe(self, x: torch.Tensor, y: torch.Tensor, task_id: torch.Tensor):
        self.train()
        x, y = x.to(self.device), y.to(self.device)
        z = self.backbone(x)
        logits = self.head(z)
        loss_cls = F.cross_entropy(logits, y)

        # replay --------------------------------------------------------------
        if self.buf_codes.numel():
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
#  Experience Replay – minimal baseline (handles "erXX" patterns)
# -----------------------------------------------------------------------------
class ExperienceReplay(nn.Module):
    """Simple experience-replay baseline with a fixed FIFO buffer.

    The buffer size is parsed from the *method name* (e.g. ``er20`` ⇒ 20).
    """

    _regex = re.compile(r"er(\d+)")

    def __init__(self, method_name: str, backbone_name: str, num_classes: int,
                 feat_dim: int, cfg: CfgNode):
        super().__init__()
        m = self._regex.fullmatch(method_name)
        if m is None:
            raise ValueError(f"ExperienceReplay received unsupported method_name={method_name}")
        self.buffer_size: int = int(m.group(1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone, _ = build_backbone(backbone_name)
        self.head = nn.Linear(feat_dim, num_classes)
        self.to(self.device)

        # ---- optimiser (numeric-string safety) ----------------------------
        opt_cfg = cfg.optimiser
        lr           = float(opt_cfg.lr)
        momentum     = float(opt_cfg.momentum)
        weight_decay = float(opt_cfg.weight_decay)
        self.opt = torch.optim.SGD(self.parameters(), lr=lr,
                                   momentum=momentum,
                                   weight_decay=weight_decay)
        # FIFO buffer (CPU tensors) ------------------------------------------
        self.register_buffer("buf_x", torch.empty(0))   # flattened imgs
        self.register_buffer("buf_y", torch.empty(0, dtype=torch.long))
        self.img_shape: Tuple[int, ...] | None = None

    # ------------------------------------------------------------------
    def _add_to_buffer(self, x: torch.Tensor, y: torch.Tensor):
        x_cpu = x.detach().cpu()
        if self.img_shape is None:
            self.img_shape = tuple(x_cpu.shape[1:])
        x_flat = x_cpu.view(x_cpu.size(0), -1)

        if self.buf_x.numel() == 0:
            self.buf_x = x_flat
            self.buf_y = y.detach().cpu()
        else:
            self.buf_x = torch.cat([self.buf_x, x_flat], dim=0)[-self.buffer_size:]
            self.buf_y = torch.cat([self.buf_y, y.detach().cpu()], dim=0)[-self.buffer_size:]

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor):
        z = self.backbone(x)
        return self.head(z)

    # ------------------------------------------------------------------
    def observe(self, x: torch.Tensor, y: torch.Tensor, task_id: torch.Tensor):
        self.train()
        x_d, y_d = x.to(self.device), y.to(self.device)

        # current batch ---------------------------------------------------
        logits = self(x_d)
        loss = F.cross_entropy(logits, y_d)

        # replay ----------------------------------------------------------
        if self.buf_y.numel():
            idx = torch.randperm(len(self.buf_y))[: x.size(0)]
            x_rep = self.buf_x[idx].view(-1, *self.img_shape).to(self.device)
            y_rep = self.buf_y[idx].to(self.device)
            rep_logits = self(x_rep)
            loss_rep = F.cross_entropy(rep_logits, y_rep)
            loss = 0.5 * loss + 0.5 * loss_rep

        # optimisation ----------------------------------------------------
        self.opt.zero_grad(); loss.backward(); self.opt.step()

        # update buffer ---------------------------------------------------
        self._add_to_buffer(x, y)
        return float(loss.item())


# =============================================================================
#  TRAINER
# =============================================================================
class Trainer:
    def __init__(self, cfg: CfgNode, logger: StdJSONLogger):
        self.cfg, self.logger = cfg, logger

    # ------------------------------------------------------------------
    @staticmethod
    def _seed_all(seed: int):
        random.seed(seed); np.random.seed(seed)
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    # ------------------------------------------------------------------
    def _instantiate_method(self, method_name: str, backbone: str,
                             feat_dim: int, num_classes: int):
        if method_name == "clipon":
            return CLIPON(backbone, num_classes=num_classes, feat_dim=feat_dim, cfg=self.cfg)
        if ExperienceReplay._regex.fullmatch(method_name):
            return ExperienceReplay(method_name, backbone, num_classes, feat_dim, self.cfg)
        raise NotImplementedError(method_name)

    # ------------------------------------------------------------------
    def run_task_sequence(self, method_name: str, backbone: str,
                          dataset_name: str, seed: int):
        # ------------------------------------------------------------------
        # 1. Data -----------------------------------------------------------
        # ------------------------------------------------------------------
        try:
            train_ds, test_ds, num_classes, img_res, is_cifar, task_map = build_continual_dataset(dataset_name)
        except NotImplementedError as e:
            self.logger.log({"warning": str(e), "dataset": dataset_name, "skipped": True})
            return  # gracefully skip unsupported datasets

        self._seed_all(seed)
        train_tf = get_train_transform(img_res, is_cifar)
        test_tf = get_test_transform(img_res, is_cifar)

        # custom collate ----------------------------------------------------
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

        # ------------------------------------------------------------------
        # 2. Method ---------------------------------------------------------
        # ------------------------------------------------------------------
        feat_dim = 512 if backbone.startswith("resnet") else 192
        try:
            model = self._instantiate_method(method_name, backbone, feat_dim, num_classes)
        except NotImplementedError as e:
            self.logger.log({"warning": str(e), "method": method_name, "skipped": True})
            return  # skip unsupported methods

        # ------------------------------------------------------------------
        # 3. Training -------------------------------------------------------
        # ------------------------------------------------------------------
        t0 = time.time()
        for tids, imgs, ys in train_loader:
            model.observe(imgs, ys, tids)

        # ------------------------------------------------------------------
        # 4. Evaluation -----------------------------------------------------
        # ------------------------------------------------------------------
        acc, cm = evaluate_model(model, test_loader)
        runtime = time.time() - t0

        # ------------------------------------------------------------------
        # 5. Save & log -----------------------------------------------------
        # ------------------------------------------------------------------
        res = dict(dataset=dataset_name, method=method_name, seed=seed,
                   avg_accuracy=acc, runtime_s=runtime)
        out_dir = Path(".research/iteration5")
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / f"{dataset_name}_{method_name}_{seed}.json"
        with open(json_path, "w") as f:
            json.dump(res, f, indent=2)

        # mandatory verification print ------------------------------------
        print(json.dumps(res, indent=2), flush=True)
        self.logger.log({"phase": "done", **res})

        # confusion-matrix figure -----------------------------------------
        from matplotlib import pyplot as plt

        img_dir = Path(".research/iteration5/images"); img_dir.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(6, 5))
        plt.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.title("Confusion Matrix"); plt.colorbar(); plt.tight_layout()
        img_path = img_dir / f"confusion_{dataset_name}_{method_name}.pdf"
        plt.savefig(img_path, bbox_inches="tight"); plt.close()
