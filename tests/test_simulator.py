from pathlib import Path

import pytest

from mahjong_ai.engine import IllegalEngineActionError, require_legal_action
from mahjong_ai.replay import normalized_events, validate_replay
from mahjong_ai.simulator import run_batch, run_match


def test_seeded_match_is_reproducible(tmp_path: Path) -> None:
    first_path = tmp_path / "first.jsonl"
    second_path = tmp_path / "second.jsonl"

    first = run_match(seed=42, replay_path=first_path)
    second = run_match(seed=42, replay_path=second_path)

    assert first.scores == second.scores
    assert first.ranks == second.ranks
    assert first.event_count == second.event_count
    assert normalized_events(first_path) == normalized_events(second_path)
    assert validate_replay(first_path) >= 1


def test_batch_uses_consecutive_seeds() -> None:
    results = run_batch(count=3, base_seed=100)

    assert [result.seed for result in results] == [100, 101, 102]
    assert all(result.event_count > 0 for result in results)


def test_random_match_finishes() -> None:
    result = run_match(seed=7)

    assert len(result.scores) == 4
    assert sorted(result.ranks) == [1, 2, 3, 4]


def test_legal_action_guard_rejects_unavailable_action() -> None:
    class FakeAction:
        def __init__(self, value: str) -> None:
            self.value = value

        def to_mjai(self) -> str:
            return self.value

    class FakeObservation:
        def legal_actions(self) -> list[FakeAction]:
            return [FakeAction('{"type":"none"}')]

    with pytest.raises(IllegalEngineActionError):
        require_legal_action(
            FakeObservation(),  # type: ignore[arg-type]
            FakeAction('{"type":"dahai","pai":"1m"}'),  # type: ignore[arg-type]
        )
