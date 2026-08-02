"""MJAI JSONL persistence and replay validation."""

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from riichienv import MjaiReplay


MjaiEvent = dict[str, Any]


def write_jsonl(events: Iterable[MjaiEvent], destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as output:
        for event in events:
            output.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")))
            output.write("\n")
    return destination


def read_jsonl(source: Path) -> list[MjaiEvent]:
    with source.open(encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file if line.strip()]


def validate_replay(source: Path, rule: str = "mjsoul") -> int:
    """Parse an MJAI log with RiichiEnv and return its number of rounds."""
    replay = MjaiReplay.from_jsonl(str(source), rule=rule)
    rounds = replay.num_rounds()
    if rounds < 1:
        raise ValueError(f"Replay contains no completed rounds: {source}")
    # Materialization exercises per-round parsing rather than only the header.
    if len(list(replay.take_kyokus())) != rounds:
        raise ValueError(f"Replay round count is inconsistent: {source}")
    return rounds


def normalized_events(source: Path) -> list[MjaiEvent]:
    """Load events for deterministic replay comparisons."""
    return read_jsonl(source)

