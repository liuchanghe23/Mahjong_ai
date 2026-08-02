from riichienv import GameRule, RiichiEnv

from mahjong_ai.controls import EfficiencyEngine
from mahjong_ai.simulator import run_match


def test_efficiency_engine_uses_lexicographic_tile_efficiency() -> None:
    env = RiichiEnv(seed=301, rule=GameRule.default_mjsoul())
    observation = next(iter(env.reset().values()))
    engine = EfficiencyEngine()

    action = engine.act(observation)
    decision = engine.last_decision

    assert decision is not None
    assert decision.policy == "efficiency_discard"
    best = min(
        decision.candidates,
        key=lambda item: (item.shanten, -item.ukeire_count, -item.ukeire_types, str(item.action)),
    )
    assert decision.selected == best.action
    assert action.to_mjai() in {candidate.to_mjai() for candidate in observation.legal_actions()}


def test_efficiency_engine_completes_match() -> None:
    result = run_match(seed=302, engine_factory=lambda _pid, _seed: EfficiencyEngine())

    assert result.event_count > 0
    assert sorted(result.ranks) == [1, 2, 3, 4]

