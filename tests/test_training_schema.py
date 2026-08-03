from pathlib import Path

import pytest

from mahjong_ai.baseline import BaselineEngine
from mahjong_ai.baseline.config import load_config
from mahjong_ai.training import apply_parameters, config_hash, load_training_spec
from mahjong_ai.training.schema import write_config_snapshot


def spec_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "training" / "stage1-search.yaml"


def test_training_baseline_disables_unreliable_yaku_features() -> None:
    spec = load_training_spec(spec_path())

    assert spec.base_config.name == "baseline-v4-training"
    assert spec.base_config.weights["legacy_value_honor_pair"] == 0.0
    assert all(
        spec.base_config.weights[name] == 0.0
        for name in (
            "yaku_yakuhai_delta", "yaku_tanyao_delta",
            "yaku_chiitoitsu_delta", "yaku_flush_delta",
        )
    )


def test_candidate_constraints_and_hash_are_deterministic() -> None:
    spec = load_training_spec(spec_path())
    first = apply_parameters(spec, spec.initial_values)
    second = apply_parameters(spec, spec.initial_values)

    assert config_hash(first) == config_hash(second)
    assert first.group_weights["efficiency"] == 1.0
    with pytest.raises(ValueError, match="outside"):
        apply_parameters(spec, {**spec.initial_values, "risk_group": 3.1})


def test_snapshot_round_trips_as_a_baseline_config(tmp_path: Path) -> None:
    spec = load_training_spec(spec_path())
    candidate = apply_parameters(spec, spec.initial_values)
    path = write_config_snapshot(candidate, tmp_path / "candidate.yaml")

    loaded = load_config(path)

    assert config_hash(loaded) == config_hash(candidate)
    assert BaselineEngine(config=loaded).config is loaded


def test_seed_sets_are_disjoint_and_seat_balanced() -> None:
    sets = load_training_spec(spec_path()).seed_sets
    ordered = sorted(sets.values(), key=lambda item: item.base_seed)

    assert all(item.matches % 6 == 0 for item in ordered)
    assert all(left.stop_seed <= right.base_seed for left, right in zip(ordered, ordered[1:]))
