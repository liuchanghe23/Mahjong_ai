"""Run deterministic parallel random search with successive halving."""

import argparse
import os
from pathlib import Path

from mahjong_ai.training.search import run_search


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=Path("configs/training/stage1-search.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/training/stage1"))
    parser.add_argument("--candidates", type=int, default=24)
    parser.add_argument("--search-seed", type=int, default=20260803)
    parser.add_argument("--workers", type=int, default=max(1, min(7, (os.cpu_count() or 2) - 1)))
    parser.add_argument("--budgets", type=int, nargs="+", default=None)
    parser.add_argument("--keep-ratio", type=float, default=0.25)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--allow-dirty", action="store_true",
                        help="allow an uncommitted worktree for disposable smoke runs")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    summary = run_search(
        spec_path=args.spec,
        output_dir=args.output_dir,
        candidate_count=args.candidates,
        search_seed=args.search_seed,
        workers=args.workers,
        budgets=tuple(args.budgets) if args.budgets else None,
        keep_ratio=args.keep_ratio,
        bootstrap_samples=args.bootstrap_samples,
        allow_dirty=args.allow_dirty,
    )
    best = summary["best"]
    print(f"best_candidate={best['candidate_id']}")
    print(f"average_rank_improvement={best['objective']:+.4f}")
    print(f"result={best['result_path']}")
