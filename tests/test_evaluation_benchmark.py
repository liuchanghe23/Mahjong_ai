from pathlib import Path

from mahjong_ai.baseline import BaselineEngine
from mahjong_ai.engines import RandomEngine
from mahjong_ai.evaluation import run_benchmark, run_comparison


def test_fixed_seed_benchmark_is_reproducible_except_timing() -> None:
    first = run_benchmark(
        engine_name="random",
        engine_factory=lambda pid, seed: RandomEngine(seed * 10 + pid),
        match_count=5,
        base_seed=100,
    )
    second = run_benchmark(
        engine_name="random",
        engine_factory=lambda pid, seed: RandomEngine(seed * 10 + pid),
        match_count=5,
        base_seed=100,
    )

    assert first.seeds == second.seeds
    assert first.stats.rank_counts == second.stats.rank_counts
    assert first.stats.wins == second.stats.wins
    assert first.stats.deal_ins == second.stats.deal_ins


def test_comparison_rotates_and_groups_two_seats_per_engine(tmp_path: Path) -> None:
    result = run_comparison(
        engine_a_name="baseline",
        engine_a_factory=lambda _pid, _seed: BaselineEngine(),
        engine_b_name="random",
        engine_b_factory=lambda pid, seed: RandomEngine(seed * 10 + pid),
        match_count=4,
        base_seed=200,
    )

    assert result.stats["baseline"].player_samples == 8
    assert result.stats["random"].player_samples == 8
    assert sum(result.stats["baseline"].rank_counts.values()) == 8
    assert len(result.paired_matches) == 4
    assert set(result.paired_deltas) == {
        "平均顺位改善", "四位率改善", "平均点差改善", "和牌率改善", "放铳率改善"
    }
    assert all(
        value["ci_low"] <= value["estimate"] <= value["ci_high"]
        for value in result.paired_deltas.values()
    )
    assert result.write_json(tmp_path / "report.json").exists()
    assert result.write_markdown(tmp_path / "report.md").exists()
