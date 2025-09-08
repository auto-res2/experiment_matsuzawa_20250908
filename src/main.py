# src/main.py
"""Top-level script executed via `python -m src.main`."""
from __future__ import annotations

from pathlib import Path
from typing import List, Any

from src.train import load_cfg, Trainer, StdJSONLogger


def _resolve_seeds(exp: dict, cfg: dict) -> List[int]:
    """Handle `${seed_list}` indirection in YAML."""
    seeds_val: Any = exp.get("seeds", [])
    if isinstance(seeds_val, str) and "seed_list" in seeds_val:
        return cfg.get("seed_list", [])
    return list(seeds_val)


def main():
    cfg = load_cfg()

    # create logfile ---------------------------------------------------------
    log_dir = Path("logs"); log_dir.mkdir(exist_ok=True)
    logger = StdJSONLogger(log_dir / "stdout.jsonl")

    trainer = Trainer(cfg, logger)

    for exp in cfg["experiments"]:
        logger.log({"experiment": exp["id"], "description": exp["description"]})
        seeds = _resolve_seeds(exp, cfg)
        for ds in exp["datasets"]:
            for bb in exp["backbones"]:
                for m in exp["methods"]:
                    for sd in seeds:
                        try:
                            trainer.run_task_sequence(m, bb, ds, int(sd))
                        except Exception as e:
                            logger.log({"error": str(e), "dataset": ds, "method": m, "seed": sd})
                            raise  # fail fast – no silent fallback

    logger.close()


if __name__ == "__main__":
    main()
