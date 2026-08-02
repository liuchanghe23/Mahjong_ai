"""Reproducible match runner around RiichiEnv."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from riichienv import GameRule, RiichiEnv

from mahjong_ai.engine import DecisionEngine, require_legal_action
from mahjong_ai.engines import RandomEngine
from mahjong_ai.replay import validate_replay, write_jsonl


EngineFactory = Callable[[int, int], DecisionEngine]


@dataclass(frozen=True)
class MatchResult:
    seed: int
    scores: tuple[int, ...]
    ranks: tuple[int, ...]
    event_count: int
    replay_path: Path | None = None


def random_engine_factory(player_id: int, match_seed: int) -> DecisionEngine:
    return RandomEngine(seed=match_seed * 10 + player_id)


def run_match(
    *,
    seed: int,
    game_mode: str = "4p-red-single",
    engine_factory: EngineFactory = random_engine_factory,
    replay_path: Path | None = None,
) -> MatchResult:
    """Run a seeded match, guarding every engine action for legality."""
    env = RiichiEnv(game_mode=game_mode, seed=seed, rule=GameRule.default_mjsoul())
    player_count = 3 if game_mode.startswith("3p-") else 4
    engines = {player_id: engine_factory(player_id, seed) for player_id in range(player_count)}
    observations = env.reset()

    while not env.done():
        actions = {
            player_id: require_legal_action(observation, engines[player_id].act(observation))
            for player_id, observation in observations.items()
        }
        observations = env.step(actions)

    events = list(env.mjai_log)
    if replay_path is not None:
        write_jsonl(events, replay_path)
        validate_replay(replay_path)

    scores, ranks = env.scores(), env.ranks()
    return MatchResult(
        seed=seed,
        scores=tuple(scores),
        ranks=tuple(ranks),
        event_count=len(events),
        replay_path=replay_path,
    )


def run_random_match(game_mode: str = "4p-red-single", seed: int = 0) -> MatchResult:
    """Backward-compatible convenience wrapper for one random match."""
    return run_match(seed=seed, game_mode=game_mode)


def run_batch(count: int, base_seed: int = 0, game_mode: str = "4p-red-single") -> list[MatchResult]:
    if count < 1:
        raise ValueError("count must be at least 1")
    return [run_match(seed=base_seed + offset, game_mode=game_mode) for offset in range(count)]

