"""Closed-hand control engine using only shanten and ukeire."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from riichienv import Action, ActionType, Observation

from mahjong_ai.baseline.efficiency_features import extract_efficiency
from mahjong_ai.baseline.features import remove_one_tile
from mahjong_ai.baseline.state import PublicState


@dataclass(frozen=True)
class EfficiencyCandidate:
    action: dict[str, Any]
    shanten: int
    ukeire_count: int
    ukeire_types: int


@dataclass(frozen=True)
class EfficiencyDecision:
    policy: str
    selected: dict[str, Any]
    candidates: tuple[EfficiencyCandidate, ...] = ()


class EfficiencyEngine:
    """Deterministic control: win, riichi, never call, maximize tile efficiency."""

    def __init__(self, config_path: Path | None = None) -> None:
        # Retain the optional argument for API compatibility; this control
        # intentionally has no trainable configuration.
        _ = config_path
        self._pending_riichi = False
        self.last_decision: EfficiencyDecision | None = None

    def act(self, observation: Observation) -> Action:
        legal = observation.legal_actions()
        if not legal:
            raise RuntimeError("EfficiencyEngine received an observation without legal actions")

        winning = [action for action in legal if action.action_type in {ActionType.TSUMO, ActionType.RON}]
        if winning:
            return self._hard_choice("win", winning[0])

        state = PublicState.from_observation(observation)
        if state.riichi_declared[state.player_id]:
            ankan = [action for action in legal if action.action_type == ActionType.ANKAN]
            if ankan:
                return self._hard_choice("riichi_wait_preserving_ankan", ankan[0])
            discards = [action for action in legal if action.action_type == ActionType.DISCARD]
            if discards:
                drawn = state.hand[-1]
                exact = [action for action in discards if action.tile == drawn]
                if not exact:
                    raise RuntimeError("Cannot identify physical drawn tile after riichi")
                return self._hard_choice("riichi_tsumogiri", exact[0])

        riichi = [action for action in legal if action.action_type == ActionType.RIICHI]
        if riichi:
            self._pending_riichi = True
            return self._hard_choice("riichi", riichi[0])

        discards = [action for action in legal if action.action_type == ActionType.DISCARD]
        if discards:
            return self._choose_discard(state, discards)

        passes = [action for action in legal if action.action_type == ActionType.PASS]
        if passes:
            return self._hard_choice("conservative_pass", passes[0])
        return self._hard_choice("legal_fallback", legal[0])

    def _hard_choice(self, policy: str, action: Action) -> Action:
        self.last_decision = EfficiencyDecision(policy, json.loads(action.to_mjai()))
        return action

    def _choose_discard(self, state: PublicState, actions: list[Action]) -> Action:
        evaluated: list[tuple[Action, EfficiencyCandidate]] = []
        for action in actions:
            remaining = remove_one_tile(state.hand, action.tile)
            features = extract_efficiency(remaining, state.visible_counts)
            evaluated.append(
                (
                    action,
                    EfficiencyCandidate(
                        action=json.loads(action.to_mjai()),
                        shanten=features.shanten,
                        ukeire_count=features.ukeire_count,
                        ukeire_types=len(features.ukeire_kinds),
                    ),
                )
            )

        selectable = evaluated
        policy = "efficiency_discard"
        if self._pending_riichi:
            selectable = [item for item in evaluated if item[1].shanten == 0]
            if not selectable:
                raise RuntimeError("Riichi declaration has no tenpai-preserving discard")
            policy = "riichi_tenpai_discard"

        key = lambda item: (
            item[1].shanten,
            -item[1].ukeire_count,
            -item[1].ukeire_types,
            item[0].to_mjai(),
        )
        evaluated.sort(key=key)
        selectable.sort(key=key)
        selected = selectable[0][0]
        self._pending_riichi = False
        self.last_decision = EfficiencyDecision(
            policy=policy,
            selected=json.loads(selected.to_mjai()),
            candidates=tuple(candidate for _, candidate in evaluated),
        )
        return selected
