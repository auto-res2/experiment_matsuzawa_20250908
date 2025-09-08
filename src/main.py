from __future__ import annotations

"""src/main.py
Entry point orchestrating the experimental suite. Invoke via

    python -m src.main

All heavy lifting is delegated to other modules so that this file only
contains *high-level* control-flow and I/O orchestration.
"""
import math
from pathlib import Path
from typing import Any, Dict, List

import torch
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
import yaml

from .evaluate import evaluate_classifier, plot_curve
from .preprocess import build_dataloaders, print_json, save_yaml
from .train import LeafLightningModule

# -----------------------------------------------------------------------------
# Constants & Config loading
# -----------------------------------------------------------------------------
# Mandatory paths enforced by the grading specification
BASE_RESEARCH_DIR = Path(".research") / "iteration4"  # ← UPDATED per spec
IMAGES_DIR = BASE_RESEARCH_DIR / "images"               # ← UPDATED per spec (plots)
EXPS_DIR = BASE_RESEARCH_DIR                             # each exp_<id> lives directly here
CONF_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"

CONFIG: Dict[str, Any]
with open(CONF_PATH) as f:
    CONFIG = yaml.safe_load(f)


# -----------------------------------------------------------------------------
# Helper – precision selection
# -----------------------------------------------------------------------------

def _select_precision(user_precision: str | int):
    """Downgrade *bf16* precision automatically on incompatible hardware."""
    if isinstance(user_precision, str) and str(user_precision).startswith("bf16"):
        # Tesla T4 (SM75) does *not* support bf16 → fall back to fp32.
        if not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()):
            return 32
    return user_precision


# -----------------------------------------------------------------------------
# Core logic
# -----------------------------------------------------------------------------

def run_experiment(exp_cfg: Dict[str, Any]):
    exp_id = exp_cfg["id"]
    out_root = EXPS_DIR / f"exp_{exp_id}"
    (out_root / "checkpoints").mkdir(parents=True, exist_ok=True)

    # store experiment-local config for reproducibility
    save_yaml(exp_cfg, out_root / "config.yaml")

    print(f"\n==================== EXPERIMENT {exp_id} ====================")
    print(exp_cfg["description"])

    ds_cfg = exp_cfg["datasets"][0]  # one dataset per exp in the simplified pipeline
    batch_size = exp_cfg.get("batch_size", CONFIG["global"]["batch_size"])

    dl_train, dl_val, num_classes = build_dataloaders(
        ds_cfg, batch_size=batch_size, num_workers=CONFIG["global"]["num_workers"]
    )

    model_cfg = exp_cfg["models"][0]

    # Inject the *experiment-specific* LEAF section into the global cfg so that
    # the LightningModule can access both.
    cfg_for_module = {**CONFIG, "leaf": exp_cfg.get("leaf", {})}
    module = LeafLightningModule(num_classes=num_classes, arch=model_cfg["arch"], cfg=cfg_for_module)

    ckpt_cb = ModelCheckpoint(
        dirpath=str(out_root / "checkpoints"), save_last=True, save_top_k=1, monitor="val/acc", mode="max"
    )
    lr_cb = LearningRateMonitor(logging_interval="step")

    precision_setting = _select_precision(CONFIG["global"]["precision"])

    trainer = Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=torch.cuda.device_count() if torch.cuda.is_available() else 1,
        precision=precision_setting,
        max_epochs=exp_cfg["epochs"],
        accumulate_grad_batches=CONFIG["global"]["accumulate_grad_batches"],
        benchmark=True,
        deterministic=False,
        callbacks=[ckpt_cb, lr_cb],
        logger=CSVLogger(save_dir=str(out_root), name="logs"),
    )

    if exp_cfg["epochs"] > 0:
        trainer.fit(module, dl_train, dl_val)
    else:
        # evaluation-only (e.g. Experiment-2)
        ckpt_path = EXPS_DIR / "exp_1" / "checkpoints" / "last.ckpt"
        if not ckpt_path.exists():
            raise RuntimeError("Pre-trained checkpoint for evaluation-only experiment not found.")
        module = LeafLightningModule.load_from_checkpoint(
            str(ckpt_path), num_classes=num_classes, arch=model_cfg["arch"], cfg=cfg_for_module
        )

    # ------------------------------------------------------------------
    # Evaluation & metrics
    # ------------------------------------------------------------------
    acc, cm = evaluate_classifier(module, dl_val)

    results_json = {
        "experiment_id": exp_id,
        "experiment_name": exp_cfg["name"],
        "description": exp_cfg["description"],
        "accuracy": acc,
        "confusion_matrix": cm,
    }

    # store JSON in the prescribed directory (.research/iteration4/)
    BASE_RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
    json_path = BASE_RESEARCH_DIR / f"results_exp_{exp_id}.json"
    json_path.write_text(yaml.safe_dump(results_json, sort_keys=False))

    print("\n--- RESULTS (JSON) ---")
    print_json(results_json)

    # ------------------------------------------------------------------
    # Plot validation accuracy curve if available
    # ------------------------------------------------------------------
    metrics_file = out_root / "logs" / "metrics.csv"
    if metrics_file.exists():
        import pandas as pd

        df = pd.read_csv(metrics_file)
        if "val/acc" in df.columns:
            vals = df.dropna(subset=["val/acc"])["val/acc"].tolist()
            plot_curve(vals, title="Validation Accuracy", ylabel="Acc", save_path=IMAGES_DIR / f"accuracy_exp_{exp_id}")
            print("Figure saved:", IMAGES_DIR / f"accuracy_exp_{exp_id}.pdf")


# -----------------------------------------------------------------------------
# Entry-point
# -----------------------------------------------------------------------------

def main():
    # persist the global config for reproducibility under the new directory
    save_yaml(CONFIG, BASE_RESEARCH_DIR / "config_global.yaml")

    for exp in CONFIG["experiments"]:
        seed_accum: List[float] = []
        for seed in CONFIG["global"]["seeds"]:
            seed_everything(seed, workers=True)
            print(f"\n>>> Running seed {seed} for experiment {exp['id']}")
            run_experiment(exp)
            # accuracy already saved – reload for aggregation
            acc = yaml.safe_load((BASE_RESEARCH_DIR / f"results_exp_{exp['id']}.json").read_text())["accuracy"]
            seed_accum.append(acc)
        mean_acc = sum(seed_accum) / len(seed_accum)
        se = (torch.std(torch.tensor(seed_accum)) / math.sqrt(len(seed_accum))).item()
        print(f"\n=== Experiment {exp['id']}  mean ± se: {mean_acc:.4f} ± {se:.4f}\n")


if __name__ == "__main__":
    main()
