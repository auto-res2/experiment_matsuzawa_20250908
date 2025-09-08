# src/main.py
"""Top-level script executed via `python -m src.main`."""
from __future__ import annotations

from pathlib import Path

from src.train import load_cfg, Trainer, StdJSONLogger


def main():
    cfg = load_cfg()

    # create logfile ---------------------------------------------------------
    log_dir = Path("logs"); log_dir.mkdir(exist_ok=True)
    logger = StdJSONLogger(log_dir / "stdout.jsonl")

    trainer = Trainer(cfg, logger)

    for exp in cfg["experiments"]:
        logger.log({"experiment": exp["id"], "description": exp["description"]})
        for ds in exp["datasets"]:
            for bb in exp["backbones"]:
                for m in exp["methods"]:
                    for sd in exp["seeds"]:
                        try:
                            trainer.run_task_sequence(m, bb, ds, sd)
                        except Exception as e:
                            logger.log({"error": str(e), "dataset": ds, "method": m, "seed": sd})
                            raise  # fail fast – no silent fallback

    logger.close()


if __name__ == "__main__":
    main()
