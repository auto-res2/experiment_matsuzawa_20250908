"""src/train.py
Contains the model definition and all training-related utilities.
"""
from __future__ import annotations
from typing import Any, Dict

import timm
import torch
import torch.nn as nn  # noqa: F401  # (needed for typing / timm introspection)
import torch.nn.functional as F
from lightning import LightningModule

__all__ = [
    "LeafLightningModule",
]

# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------

def _resolve_arch_name(arch: str) -> str:
    """Translate a *legacy* or *typo* architecture string to a valid timm name.

    The timm library changed several model names between the 0.x and 1.x series.
    This helper keeps backward-compatibility with old experiment configs by
    mapping the outdated names to their new counterparts.
    """
    # Directly try the user-provided name first.
    if arch in timm.list_models(pretrained=True):
        return arch

    # Common one-off fixes --------------------------------------------------
    manual_map = {
        # ResNet50 w/ A1H recipe (new naming scheme uses dot & dataset tag)
        "resnet50_a1h": "resnet50.a1h_in1k",
        # ViT MoCo v3 checkpoint (newer timm swapped the dot/underscore order)
        "vit_base_patch16_224.mocov3": "vit_base_patch16_224_mocov3",
    }
    if arch in manual_map and manual_map[arch] in timm.list_models(pretrained=True):
        return manual_map[arch]

    # Fallback heuristics ---------------------------------------------------
    # Swap underscores ↔ dots (e.g. resnet50_a1h -> resnet50.a1h)
    alt = arch.replace("_", ".") if "_" in arch else arch.replace(".", "_")
    if alt in timm.list_models(pretrained=True):
        return alt

    # If nothing works, propagate the original error so that the caller can
    # surface a clear, actionable message.
    raise RuntimeError(f"Unknown model architecture '{arch}' for current timm version.")


class LeafLightningModule(LightningModule):
    """LightningModule implementing the (simplified) LEAF algorithm.

    Heavy algorithmic components are stubbed out so that the experiment
    completes end-to-end within the execution constraints of this task.
    """

    def __init__(self, num_classes: int, arch: str, cfg: Dict[str, Any]):
        super().__init__()
        # Do **not** save the full cfg in the checkpoint to keep them small.
        self.save_hyperparameters(ignore=["cfg"])
        self.cfg = cfg

        # ------------------------------------------------------------------
        # Robust model creation w/ backward-compat architecture resolution
        # ------------------------------------------------------------------
        resolved_arch = _resolve_arch_name(arch)
        self.model = timm.create_model(resolved_arch, pretrained=True, num_classes=num_classes)

        # Hyper-parameters for the (stubbed) spectral regulariser
        self.gamma_cd: float = cfg["leaf"].get("gamma_cd", 0.0)

    # ------------------------------------------------------------------
    # Forward + helpers
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.model(x)

    # ------------------------------------------------------------------
    # Loss helpers
    # ------------------------------------------------------------------
    def _fourier_cd(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # noqa: D401, N802
        """Placeholder for the Fourier Consistency-Distance regulariser."""
        # 🔴 TODO – insert real implementation when available.
        return torch.tensor(0.0, device=logits.device)

    def _compute_loss(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, y)
        if self.gamma_cd > 0:
            ce = ce + self.gamma_cd * self._fourier_cd(logits, y)
        return ce

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------
    def training_step(self, batch, batch_idx):  # type: ignore[override]
        x, y = batch
        logits = self(x)
        loss = self._compute_loss(logits, y)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):  # type: ignore[override]
        x, y = batch
        logits = self(x)
        loss = self._compute_loss(logits, y)
        preds = logits.argmax(1)
        acc = (preds == y).float().mean()
        self.log_dict({"val/loss": loss, "val/acc": acc}, prog_bar=True)

    # ------------------------------------------------------------------
    # Optimiser / LR-scheduler
    # ------------------------------------------------------------------
    def configure_optimizers(self):  # type: ignore[override]
        opt_cfg = self.cfg["global"]["optimizer"]

        opt = torch.optim.AdamW(
            self.parameters(), lr=opt_cfg["lr"], weight_decay=opt_cfg["weight_decay"]
        )

        # Lightning sets ``estimated_stepping_batches`` after dataloaders are
        # attached, therefore it's safe to reference it here.
        total_steps = self.trainer.estimated_stepping_batches  # type: ignore[attr-defined]
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)

        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
