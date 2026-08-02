"""Run the full baseline against every single-feature ablation."""

import argparse
from pathlib import Path

from mahjong_ai.baseline import BaselineEngine
from mahjong_ai.evaluation import run_comparison


ABLATIONS = (
    "no-risk", "no-yaku", "no-dora", "no-shape", "no-lookahead",
    "no-all-yaku", "legacy-overlap-shape",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matches", type=int, default=600)
    parser.add_argument("--base-seed", type=int, default=30000)
    parser.add_argument("--game-mode", default="4p-red-east")
    parser.add_argument("--ablations", nargs="+", choices=ABLATIONS, default=list(ABLATIONS))
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260802)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/ablations"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.matches % 6:
        raise SystemExit("--matches must be divisible by 6 for complete seat-pair cycles")
    full_path = Path("configs/baseline.yaml")
    for index, ablation in enumerate(args.ablations):
        config_path = Path("configs/ablations") / f"{ablation}.yaml"
        result = run_comparison(
            engine_a_name="full",
            engine_a_factory=lambda _pid, _seed, path=full_path: BaselineEngine(path),
            engine_b_name=ablation,
            engine_b_factory=lambda _pid, _seed, path=config_path: BaselineEngine(path),
            match_count=args.matches,
            base_seed=args.base_seed,
            game_mode=args.game_mode,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed + index,
        )
        stem = f"full-vs-{ablation}-{args.game_mode}-n{args.matches}-seed{args.base_seed}"
        result.write_json(args.output_dir / f"{stem}.json")
        result.write_markdown(args.output_dir / f"{stem}.md")
        rank = result.paired_deltas["平均顺位改善"]
        print(f"{ablation}: rank={rank['estimate']:.4f} CI=[{rank['ci_low']:.4f}, {rank['ci_high']:.4f}]")
