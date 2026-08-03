"""Run the deterministic one-factor group-weight ablation."""

import argparse
import os
from pathlib import Path

from mahjong_ai.training.ablation import run_ablation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=Path("configs/training/group-weight-ablation.yaml"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("artifacts/training/group-weight-ablation-v1"),
    )
    parser.add_argument("--workers", type=int, default=max(1, min(7, (os.cpu_count() or 2) - 1)))
    parser.add_argument("--allow-dirty", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = run_ablation(
        config_path=args.config,
        output_dir=args.output_dir,
        workers=args.workers,
        allow_dirty=args.allow_dirty,
    )
    print(f"best={result['best']['axis']}:{result['best']['level']}")
    print(f"average_rank_improvement={result['best']['objective']:+.4f}")
