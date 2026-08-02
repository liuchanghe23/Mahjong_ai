"""Explainable baseline decision engine."""

import json
from pathlib import Path

from riichienv import Action, ActionType, Observation
import riichienv.convert as convert

from mahjong_ai.baseline.config import BaselineConfig, default_config_path, load_config
from mahjong_ai.baseline.decision import CandidateEvaluation, DecisionRecord
from mahjong_ai.baseline.features import FeaturePipeline
from mahjong_ai.baseline.scoring import aggregate_group_contributions, select_best_variant
from mahjong_ai.baseline.state import PublicState


class BaselineEngine:
    """Hard-rule action selection followed by explainable discard scoring."""

    def __init__(self, config_path: Path | None = None) -> None:
        self.config: BaselineConfig = load_config(config_path or default_config_path())
        self._features = FeaturePipeline.from_config(self.config)
        self.last_decision: DecisionRecord | None = None
        self._pending_riichi = False

    def act(self, observation: Observation) -> Action:
        legal = observation.legal_actions()
        if not legal:
            raise RuntimeError("BaselineEngine received an observation without legal actions")

        winning = [action for action in legal if action.action_type in {ActionType.TSUMO, ActionType.RON}]
        if winning and self.config.policy.always_win:
            return self._hard_choice("win", winning[0])

        state = PublicState.from_observation(observation)
        if state.riichi_declared[state.player_id]:
            valid_ankan = [action for action in legal if action.action_type == ActionType.ANKAN]
            if valid_ankan:
                # RiichiEnv only exposes this action after comparing the exact
                # wait set before and after the closed kan.
                return self._hard_choice("riichi_wait_preserving_ankan", valid_ankan[0])
            discards = [action for action in legal if action.action_type == ActionType.DISCARD]
            if discards:
                return self._choose_riichi_tsumogiri(state, discards)

        riichi = [action for action in legal if action.action_type == ActionType.RIICHI]
        if riichi and self.config.policy.always_riichi:
            self._pending_riichi = True
            return self._hard_choice("riichi", riichi[0])

        discards = [action for action in legal if action.action_type == ActionType.DISCARD]
        if discards:
            action = self._choose_discard(observation, discards, state=state)
            self._pending_riichi = False
            return action

        passes = [action for action in legal if action.action_type == ActionType.PASS]
        if passes and self.config.policy.conservative_calls:
            return self._hard_choice("conservative_pass", passes[0])

        return self._hard_choice("legal_fallback", legal[0])

    def _hard_choice(self, policy: str, action: Action) -> Action:
        self.last_decision = DecisionRecord(policy, json.loads(action.to_mjai()), ())
        return action

    def _choose_riichi_tsumogiri(self, state: PublicState, discards: list[Action]) -> Action:
        """Discard the physical drawn tile after riichi, never choose by score."""
        drawn_tile = state.hand[-1]
        exact = [action for action in discards if action.tile == drawn_tile]
        if exact:
            return self._hard_choice("riichi_tsumogiri", exact[0])

        # Defensive fallback for externally constructed observations that sort
        # the hand and lose the physical drawn-tile position.
        latest_draw_kind = None
        for action in discards:
            candidate = json.loads(action.to_mjai())
            if candidate.get("tsumogiri"):
                latest_draw_kind = candidate.get("pai")
                return self._hard_choice("riichi_tsumogiri", action)
        raise RuntimeError(
            "Riichi observation contains discards but the drawn tile cannot be identified; "
            f"last hand tile={drawn_tile}, actions={[a.to_mjai() for a in discards]}, hint={latest_draw_kind}"
        )

    def _choose_discard(
        self,
        observation: Observation,
        actions: list[Action],
        state: PublicState | None = None,
    ) -> Action:
        state = state or PublicState.from_observation(observation)
        evaluated: list[tuple[Action, CandidateEvaluation]] = []
        for action in actions:
            score, contributions, features = select_best_variant(
                self._features.extract_candidates(action.tile, state), self.config
            )
            tile_name = convert.tid_to_mjai(action.tile)
            yaku_improvements = []
            if features.yaku is not None:
                before = {item.yaku: item for item in features.yaku.before}
                for after in features.yaku.after:
                    delta = after.potential - before[after.yaku].potential
                    if abs(delta) >= 0.01:
                        yaku_improvements.append(f"{after.yaku}役潜力{delta:+.2f}")
            reasons = (
                f"切{tile_name}后为{features.shanten}向听",
                f"有效牌{len(features.ukeire_kinds)}种、预计{features.ukeire_count}枚",
                f"放铳风险系数{features.danger:.2f}",
                *yaku_improvements,
            )
            evaluated.append(
                (
                    action,
                    CandidateEvaluation(
                        action=json.loads(action.to_mjai()),
                        score=score,
                        features=features.values,
                        normalized_features=features.normalized_values,
                        contributions=contributions,
                        group_contributions=aggregate_group_contributions(contributions),
                        shanten=features.shanten,
                        ukeire_kinds=features.ukeire_kinds,
                        ukeire_count=features.ukeire_count,
                        danger=features.danger,
                        reasons=reasons,
                        yaku_before=features.yaku.before if features.yaku else (),
                        yaku_after=features.yaku.after if features.yaku else (),
                    ),
                )
            )

        # After the RIICHI declaration action, only choose a discard that
        # actually leaves the hand in tenpai. RiichiEnv intentionally exposes
        # all discards at this intermediate stage.
        selectable = evaluated
        policy = "linear_discard"
        if self._pending_riichi:
            tenpai = [item for item in evaluated if item[1].shanten == 0]
            if not tenpai:
                raise RuntimeError("Riichi declaration has no tenpai-preserving discard")
            selectable = tenpai
            policy = "riichi_tenpai_discard"

        # Shanten is a hard lexicographic priority, not a trainable weight.
        # MJAI string provides a deterministic final tie-break.
        evaluated.sort(key=lambda item: (item[1].shanten, -item[1].score, item[0].to_mjai()))
        selectable.sort(key=lambda item: (item[1].shanten, -item[1].score, item[0].to_mjai()))
        selected_action = selectable[0][0]
        self.last_decision = DecisionRecord(
            policy=policy,
            selected=json.loads(selected_action.to_mjai()),
            candidates=tuple(candidate for _, candidate in evaluated),
        )
        return selected_action
