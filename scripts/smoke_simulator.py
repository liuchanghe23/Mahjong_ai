"""Run one reproducible local match and validate its MJAI replay."""

from pathlib import Path

from mahjong_ai.simulator import run_match


if __name__ == "__main__":
    result = run_match(seed=20260802, replay_path=Path("artifacts/replays/smoke.jsonl"))
    print(f"seed={result.seed}")
    print(f"scores={result.scores}")
    print(f"ranks={result.ranks}")
    print(f"events={result.event_count}")
    print(f"replay={result.replay_path}")
