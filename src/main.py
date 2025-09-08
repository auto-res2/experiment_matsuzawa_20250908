"""src/main.py
Entry point for every experiment run.  Execute via

    python -m src.main

This script is intentionally minimal – all heavy lifting happens inside the
other modules.  It is *only* responsible for
1. reading `config/config.yaml`,
2. creating the required directory structure, and
3. orchestrating the experiment / result dumping loop.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml

from .train import GNNExperiment

# ---------------------------------------------------------------------------
#  Paths & directories
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / ".research" / "iteration5"
IMAGES_DIR = RESULTS_DIR / "images"
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

for p in (DATA_DIR, RESULTS_DIR, IMAGES_DIR):
    p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError as exc:
        sys.exit(f"[config] File not found: {path}\n{exc}")


def _run_experiment(spec: Dict[str, Any]):
    exp_name = spec["name"]
    common = spec["common"]
    runs: List[Dict[str, Any]] = spec["runs"]

    print("\n============================= EXPERIMENT:", exp_name, "=============================")
    print("Running", len(runs), "configuration(s)")

    for seed in (0, 1):  # demo seeds – extend to 20 for full paper
        for run_cfg in runs:
            merged_cfg = {
                **run_cfg,
                "exp_name": exp_name,
                "seed": seed,
                "optim": common["optim"],
                "lambda_cons": common["lambda_cons"],
            }
            print(
                f"\n--- Dataset {merged_cfg['dataset']} | {merged_cfg['model']['type']} | depth {merged_cfg['model']['depth']} | seed {seed} ---"
            )

            experiment = GNNExperiment(merged_cfg, RESULTS_DIR, IMAGES_DIR)
            result_dict = experiment.run()

            # ------------------------- persistence -------------------------
            json_name = (
                f"{exp_name}_{merged_cfg['dataset']}_{merged_cfg['model']['type']}"  # base
                f"_depth{merged_cfg['model']['depth']}_seed{seed}.json"
            )
            json_path = RESULTS_DIR / json_name
            with open(json_path, "w") as fp:
                json.dump(result_dict, fp, indent=2)

            print("(JSON result)")
            print(json.dumps(result_dict, indent=2))
            print("Figures:", ", ".join(result_dict["figures"]))


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    full_cfg = _load_yaml(CONFIG_PATH)
    for exp in full_cfg["experiments"]:
        _run_experiment(exp)
