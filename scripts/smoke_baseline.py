"""Run one match with four baseline engines and print the first decision summary."""

from pathlib import Path

from mahjong_ai.baseline import BaselineEngine
from mahjong_ai.simulator import run_match


if __name__ == "__main__":
    engines: dict[int, BaselineEngine] = {}

    def factory(player_id: int, _seed: int) -> BaselineEngine:
        engine = BaselineEngine()
        engines[player_id] = engine
        return engine

    result = run_match(
        seed=20260802,
        game_mode="4p-red-single",
        engine_factory=factory,
        replay_path=Path("artifacts/replays/baseline-smoke.jsonl"),
    )
    print(f"scores={result.scores}")
    print(f"ranks={result.ranks}")
    print(f"events={result.event_count}")
    for player_id, engine in engines.items():
        decision = engine.last_decision
        print(f"player={player_id} last_policy={decision.policy if decision else 'none'}")

