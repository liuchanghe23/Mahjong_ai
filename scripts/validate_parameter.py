"""Validate one parameter candidate on an independent seed set."""

import argparse
from pathlib import Path

from mahjong_ai.training.validation import run_parameter_validation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=Path("configs/training/stage1-search.yaml"))
    parser.add_argument("--parameter", default="value_group")
    parser.add_argument("--value", type=float, default=0.5)
    parser.add_argument("--seed-set", default="selection")
    parser.add_argument("--matches", type=int, default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260811)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("artifacts/training/value-0.5-selection"),
    )
    parser.add_argument("--allow-dirty", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = run_parameter_validation(
        spec_path=args.spec, output_dir=args.output_dir,
        parameter=args.parameter, value=args.value, seed_set=args.seed_set,
        matches=args.matches, bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed, allow_dirty=args.allow_dirty,
    )
    rank = result["paired_deltas"]["平均顺位改善"]
    print(f"average_rank_improvement={rank['estimate']:+.4f}")
    print(f"ci=[{rank['ci_low']:+.4f}, {rank['ci_high']:+.4f}]")
