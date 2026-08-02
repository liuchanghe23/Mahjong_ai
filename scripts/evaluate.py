"""Run a reproducible benchmark and emit JSON plus Markdown reports."""

import argparse
from pathlib import Path

from mahjong_ai.baseline import BaselineEngine
from mahjong_ai.controls import EfficiencyEngine
from mahjong_ai.engines import RandomEngine
from mahjong_ai.evaluation import run_benchmark, run_comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    engine_choices = ("baseline", "efficiency", "random")
    parser.add_argument("--engine", choices=engine_choices, default="baseline")
    parser.add_argument("--opponent", choices=("none", *engine_choices), default="efficiency")
    parser.add_argument("--matches", type=int, default=1000)
    parser.add_argument("--base-seed", type=int, default=10000)
    parser.add_argument("--game-mode", default="4p-red-single")
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument("--opponent-config", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260802)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/evaluation"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    factories = {
        "baseline": lambda _pid, _seed: BaselineEngine(args.config),
        "efficiency": lambda _pid, _seed: EfficiencyEngine(),
        "random": lambda pid, seed: RandomEngine(seed * 10 + pid),
    }
    opponent_factories = dict(factories)
    opponent_factories["baseline"] = lambda _pid, _seed: BaselineEngine(args.opponent_config)
    if args.opponent == "none":
        result = run_benchmark(
            engine_name=args.engine,
            engine_factory=factories[args.engine],
            match_count=args.matches,
            base_seed=args.base_seed,
            game_mode=args.game_mode,
        )
        stats = result.stats
        stem = f"{args.engine}-{args.game_mode}-n{args.matches}-seed{args.base_seed}"
    else:
        if args.opponent == args.engine:
            raise SystemExit("--engine and --opponent must differ; use --opponent none for self-play")
        result = run_comparison(
            engine_a_name=args.engine,
            engine_a_factory=factories[args.engine],
            engine_b_name=args.opponent,
            engine_b_factory=opponent_factories[args.opponent],
            match_count=args.matches,
            base_seed=args.base_seed,
            game_mode=args.game_mode,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
        )
        stats = result.stats[args.engine]
        stem = f"{args.engine}-vs-{args.opponent}-{args.game_mode}-n{args.matches}-seed{args.base_seed}"
    json_path = result.write_json(args.output_dir / f"{stem}.json")
    markdown_path = result.write_markdown(args.output_dir / f"{stem}.md")
    print(f"json={json_path}")
    print(f"markdown={markdown_path}")
    print(f"average_rank={stats.average_rank:.4f}")
    print(f"win_rate={stats.win_rate_per_round:.2%}")
    print(f"deal_in_rate={stats.deal_in_rate_per_round:.2%}")
