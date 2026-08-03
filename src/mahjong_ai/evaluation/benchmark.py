"""Deterministic benchmark runner for one engine configuration."""

import json
import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter_ns
from typing import Any

from riichienv import GameRule, RiichiEnv

from mahjong_ai.engine import DecisionEngine, require_legal_action
from mahjong_ai.evaluation.metrics import AggregateStats, DecisionTiming, aggregate, extract_player_stats


EngineFactory = Callable[[int, int], DecisionEngine]


@dataclass(frozen=True)
class BenchmarkResult:
    engine: str
    game_mode: str
    base_seed: int
    match_count: int
    seeds: tuple[int, ...]
    stats: AggregateStats

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "game_mode": self.game_mode,
            "base_seed": self.base_seed,
            "match_count": self.match_count,
            "seeds": list(self.seeds),
            "stats": self.stats.to_dict(),
        }

    def write_json(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def write_markdown(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        s = self.stats
        rows = [
            ("对局数", str(self.match_count)),
            ("玩家样本", str(s.player_samples)),
            ("平均顺位", f"{s.average_rank:.4f}"),
            ("一位率", f"{s.rank_counts[1] / s.player_samples:.2%}"),
            ("四位率", f"{s.rank_counts[4] / s.player_samples:.2%}"),
            ("每局和牌率", f"{s.win_rate_per_round:.2%}"),
            ("每局放铳率", f"{s.deal_in_rate_per_round:.2%}"),
            ("每局立直率", f"{s.riichi_rate_per_round:.2%}"),
            ("平均和牌收入", f"{s.average_win_points:.1f}"),
            ("平均放铳损失", f"{s.average_deal_in_points:.1f}"),
            ("平均决策耗时", f"{s.mean_decision_ms:.3f} ms"),
            ("最大决策耗时", f"{s.max_decision_ms:.3f} ms"),
            ("流局听牌未知样本", str(s.draw_tenpai_unknown)),
        ]
        lines = [
            f"# {self.engine} 固定种子评测",
            "",
            f"- 模式：`{self.game_mode}`",
            f"- 种子：`{self.base_seed}` 至 `{self.base_seed + self.match_count - 1}`",
            "",
            "| 指标 | 数值 |",
            "|---|---:|",
            *(f"| {name} | {value} |" for name, value in rows),
            "",
            "> 流局事件在四家全听或全不听时点差均为0，MJAI日志无法区分，因此相关样本标记为未知。",
            "",
        ]
        path.write_text("\n".join(lines), encoding="utf-8")
        return path


@dataclass(frozen=True)
class ComparisonResult:
    engines: tuple[str, str]
    game_mode: str
    base_seed: int
    match_count: int
    seats_per_engine: int
    stats: dict[str, AggregateStats]
    paired_matches: tuple[dict[str, Any], ...] = ()
    paired_deltas: dict[str, dict[str, float]] | None = None
    bootstrap_seed: int = 20260802
    bootstrap_samples: int = 10_000

    def to_dict(self) -> dict[str, Any]:
        return {
            "engines": list(self.engines),
            "game_mode": self.game_mode,
            "base_seed": self.base_seed,
            "match_count": self.match_count,
            "physical_match_count": self.match_count * 2,
            "seats_per_engine": self.seats_per_engine,
            "stats": {name: stats.to_dict() for name, stats in self.stats.items()},
            "paired_matches": list(self.paired_matches),
            "paired_deltas": self.paired_deltas or {},
            "bootstrap": {"seed": self.bootstrap_seed, "samples": self.bootstrap_samples},
        }

    def write_json(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def write_markdown(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        headers = self.engines
        metric_rows = [
            ("玩家样本", lambda s: str(s.player_samples)),
            ("平均顺位", lambda s: f"{s.average_rank:.4f}"),
            ("一位率", lambda s: f"{s.rank_counts[1] / s.player_samples:.2%}"),
            ("四位率", lambda s: f"{s.rank_counts[4] / s.player_samples:.2%}"),
            ("每局和牌率", lambda s: f"{s.win_rate_per_round:.2%}"),
            ("每局放铳率", lambda s: f"{s.deal_in_rate_per_round:.2%}"),
            ("每局立直率", lambda s: f"{s.riichi_rate_per_round:.2%}"),
            ("平均样本收支", lambda s: f"{s.average_point_delta:.1f}"),
            ("平均和牌收入", lambda s: f"{s.average_win_points:.1f}"),
            ("平均放铳损失", lambda s: f"{s.average_deal_in_points:.1f}"),
            ("平均决策耗时", lambda s: f"{s.mean_decision_ms:.3f} ms"),
        ]
        lines = [
            f"# {headers[0]} vs {headers[1]} 固定种子对比",
            "",
            f"- 模式：`{self.game_mode}`",
            f"- 配对种子数：{self.match_count}",
            f"- 实际对局数：{self.match_count * 2}",
            f"- 每局席位：每个引擎{self.seats_per_engine}席；每个种子进行A/B席位镜像互换",
            f"- 种子：`{self.base_seed}` 至 `{self.base_seed + self.match_count - 1}`",
            "",
            f"| 指标 | {headers[0]} | {headers[1]} |",
            "|---|---:|---:|",
            *(
                f"| {label} | {formatter(self.stats[headers[0]])} | {formatter(self.stats[headers[1]])} |"
                for label, formatter in metric_rows
            ),
            "",
            "## 配对差值（正数表示前者更优）",
            "",
            "| 指标 | 点估计 | 95% Bootstrap CI |",
            "|---|---:|---:|",
            *(f"| {name} | {value['estimate']:.4f} | [{value['ci_low']:.4f}, {value['ci_high']:.4f}] |"
              for name, value in (self.paired_deltas or {}).items()),
            "",
            "> 每个种子运行两局：第二局完全互换A/B席位，镜像对作为一个统计样本。",
            f"> 置信区间按种子镜像对重采样 {self.bootstrap_samples} 次（随机种子 {self.bootstrap_seed}）。",
            "",
        ]
        path.write_text("\n".join(lines), encoding="utf-8")
        return path

def run_benchmark(
    *,
    engine_name: str,
    engine_factory: EngineFactory,
    match_count: int,
    base_seed: int,
    game_mode: str = "4p-red-single",
) -> BenchmarkResult:
    if match_count < 1:
        raise ValueError("match_count must be at least 1")
    player_count = 3 if game_mode.startswith("3p-") else 4
    all_match_stats = []
    seeds = tuple(base_seed + offset for offset in range(match_count))

    for seed in seeds:
        env = RiichiEnv(game_mode=game_mode, seed=seed, rule=GameRule.default_mjsoul())
        engines = {pid: engine_factory(pid, seed) for pid in range(player_count)}
        timings = {pid: DecisionTiming() for pid in range(player_count)}
        observations = env.reset()
        while not env.done():
            actions = {}
            for pid, observation in observations.items():
                started = perf_counter_ns()
                action = engines[pid].act(observation)
                timings[pid].add(perf_counter_ns() - started)
                actions[pid] = require_legal_action(observation, action)
            observations = env.step(actions)
        scores, ranks = tuple(env.scores()), tuple(env.ranks())
        all_match_stats.append(extract_player_stats(list(env.mjai_log), scores, ranks, timings))

    return BenchmarkResult(
        engine=engine_name,
        game_mode=game_mode,
        base_seed=base_seed,
        match_count=match_count,
        seeds=seeds,
        stats=aggregate(engine_name, all_match_stats),
    )


def run_comparison(
    *,
    engine_a_name: str,
    engine_a_factory: EngineFactory,
    engine_b_name: str,
    engine_b_factory: EngineFactory,
    match_count: int,
    base_seed: int,
    game_mode: str = "4p-red-single",
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 20260802,
) -> ComparisonResult:
    if match_count < 1:
        raise ValueError("match_count must be at least 1")
    if game_mode.startswith("3p-"):
        raise ValueError("Balanced 2-vs-2 comparison currently requires a four-player mode")
    if engine_a_name == engine_b_name:
        raise ValueError("Comparison engine names must be distinct")

    names = (engine_a_name, engine_b_name)
    factories = {engine_a_name: engine_a_factory, engine_b_name: engine_b_factory}
    grouped: dict[str, list[list[Any]]] = {name: [] for name in names}
    paired_matches: list[dict[str, Any]] = []

    for offset in range(match_count):
        seed = base_seed + offset
        # Cycle through all C(4, 2)=6 seat pairs so A appears both adjacent
        # and opposite, and occupies every seat equally over a full cycle.
        seat_pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
        original_a_seats = set(seat_pairs[offset % len(seat_pairs)])
        paired_players: dict[str, list[Any]] = {name: [] for name in names}
        for a_seats in (original_a_seats, set(range(4)) - original_a_seats):
            labels = [engine_a_name if pid in a_seats else engine_b_name for pid in range(4)]
            env = RiichiEnv(game_mode=game_mode, seed=seed, rule=GameRule.default_mjsoul())
            engines = {pid: factories[labels[pid]](pid, seed) for pid in range(4)}
            timings = {pid: DecisionTiming() for pid in range(4)}
            observations = env.reset()
            while not env.done():
                actions = {}
                for pid, observation in observations.items():
                    started = perf_counter_ns()
                    action = engines[pid].act(observation)
                    timings[pid].add(perf_counter_ns() - started)
                    actions[pid] = require_legal_action(observation, action)
                observations = env.step(actions)

            players = extract_player_stats(
                list(env.mjai_log), tuple(env.scores()), tuple(env.ranks()), timings
            )
            for name in names:
                team = [player for player in players if labels[player.player_id] == name]
                grouped[name].append(team)
                paired_players[name].extend(team)
        paired_matches.append({
            "seed": seed,
            "a_seats": sorted(original_a_seats),
            "mirrored_a_seats": sorted(set(range(4)) - original_a_seats),
            engine_a_name: _team_metrics(paired_players[engine_a_name]),
            engine_b_name: _team_metrics(paired_players[engine_b_name]),
        })

    deltas = _paired_bootstrap(
        paired_matches, engine_a_name, engine_b_name, bootstrap_samples, bootstrap_seed
    )

    return ComparisonResult(
        engines=names,
        game_mode=game_mode,
        base_seed=base_seed,
        match_count=match_count,
        seats_per_engine=2,
        stats={name: aggregate(name, grouped[name]) for name in names},
        paired_matches=tuple(paired_matches),
        paired_deltas=deltas,
        bootstrap_seed=bootstrap_seed,
        bootstrap_samples=bootstrap_samples,
    )


def _team_metrics(players: list[Any]) -> dict[str, float]:
    rounds = sum(player.rounds for player in players)
    count = len(players)
    return {
        "average_rank": sum(player.rank for player in players) / count,
        "fourth_rate": sum(player.rank == 4 for player in players) / count,
        "average_point_delta": sum(player.point_delta for player in players) / count,
        "win_rate": sum(player.wins for player in players) / rounds if rounds else 0.0,
        "deal_in_rate": sum(player.deal_ins for player in players) / rounds if rounds else 0.0,
    }


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _paired_bootstrap(
    matches: list[dict[str, Any]], a: str, b: str, samples: int, seed: int
) -> dict[str, dict[str, float]]:
    if samples < 1:
        raise ValueError("bootstrap_samples must be at least 1")
    # Every definition is oriented so a positive value favours engine A.
    definitions = {
        "平均顺位改善": ("average_rank", -1.0),
        "四位率改善": ("fourth_rate", -1.0),
        "平均点差改善": ("average_point_delta", 1.0),
        "和牌率改善": ("win_rate", 1.0),
        "放铳率改善": ("deal_in_rate", -1.0),
    }
    observations = {
        label: [direction * (match[a][metric] - match[b][metric]) for match in matches]
        for label, (metric, direction) in definitions.items()
    }
    rng = random.Random(seed)
    bootstrapped = {label: [] for label in definitions}
    for _ in range(samples):
        indices = [rng.randrange(len(matches)) for _ in matches]
        for label, values in observations.items():
            bootstrapped[label].append(sum(values[index] for index in indices) / len(indices))
    return {
        label: {
            "estimate": sum(values) / len(values),
            "ci_low": _percentile(bootstrapped[label], 0.025),
            "ci_high": _percentile(bootstrapped[label], 0.975),
        }
        for label, values in observations.items()
    }
