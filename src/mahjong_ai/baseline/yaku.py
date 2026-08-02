"""Stage-aware yaku-potential estimates for incomplete hands."""

from dataclasses import dataclass
from enum import Enum

from mahjong_ai.baseline.config import YakuConfig
from mahjong_ai.baseline.state import PublicState
from mahjong_ai.baseline.tiles import counts34, is_honor, tile_rank


class YakuState(str, Enum):
    IMPOSSIBLE = "impossible"
    POSSIBLE = "possible"
    LIKELY = "likely"
    GUARANTEED = "guaranteed"
    COMPLETE = "complete"


@dataclass(frozen=True)
class YakuEvaluation:
    yaku: str
    state: YakuState
    probability: float
    expected_han: float
    potential: float
    evidence: tuple[str, ...]


@dataclass(frozen=True)
class YakuDelta:
    before: tuple[YakuEvaluation, ...]
    after: tuple[YakuEvaluation, ...]
    values: dict[str, float]


def _probability_for_shanten(shanten: int, config: YakuConfig) -> float:
    key = {0: "ready", 1: "one", 2: "two", 3: "three"}.get(shanten, "far")
    return config.shanten_probability[key]


def _state_for_probability(probability: float, guaranteed: bool = False) -> YakuState:
    if guaranteed:
        return YakuState.GUARANTEED
    if probability <= 0:
        return YakuState.IMPOSSIBLE
    if probability >= 0.7:
        return YakuState.LIKELY
    return YakuState.POSSIBLE


def _is_terminal_or_honor(kind: int) -> bool:
    return is_honor(kind) or tile_rank(kind) in {1, 9}


class YakuPotentialEvaluator:
    """Estimate yaku routes without claiming exact win probability."""

    def __init__(self, config: YakuConfig) -> None:
        self.config = config

    def evaluate(self, hand: list[int], state: PublicState) -> tuple[YakuEvaluation, ...]:
        return (
            self._yakuhai(hand, state),
            self._tanyao(hand, state),
            self._chiitoitsu(hand, state),
            self._flush(hand, state),
        )

    def compare(self, before_hand: list[int], after_hand: list[int], state: PublicState) -> YakuDelta:
        before = self.evaluate(before_hand, state)
        after = self.evaluate(after_hand, state)
        before_by_name = {item.yaku: item for item in before}
        values = {
            f"yaku_{item.yaku}_delta": item.potential - before_by_name[item.yaku].potential
            for item in after
        }
        return YakuDelta(before, after, values)

    def _yakuhai(self, hand: list[int], state: PublicState) -> YakuEvaluation:
        counts = counts34(hand)
        expected_han = 0.0
        evidence: list[str] = []
        guaranteed = False
        for kind in state.value_honor_kinds:
            han = state.value_honor_han(kind)
            count = counts[kind]
            if count >= 3:
                expected_han += han
                guaranteed = True
                evidence.append(f"役牌种类{kind}已有刻子，预计{han}番")
            elif count == 2:
                expected_han += han * self.config.yakuhai_pair_probability
                evidence.append(f"役牌种类{kind}为对子")
            elif count == 1:
                expected_han += han * self.config.yakuhai_single_probability
        probability = min(1.0, expected_han / max(1, sum(state.value_honor_han(k) for k in state.value_honor_kinds)))
        return YakuEvaluation(
            "yakuhai",
            _state_for_probability(probability, guaranteed),
            probability,
            expected_han,
            expected_han,
            tuple(evidence) or ("没有形成役牌对子或刻子",),
        )

    def _tanyao(self, hand: list[int], state: PublicState) -> YakuEvaluation:
        fixed_forbidden = sum(
            _is_terminal_or_honor(kind)
            for meld in state.self_melds
            for kind in set(meld)
        )
        if fixed_forbidden:
            return YakuEvaluation(
                "tanyao", YakuState.IMPOSSIBLE, 0.0, 0.0, 0.0, ("副露中已经固定幺九牌",)
            )
        forbidden = sum(_is_terminal_or_honor(kind) for kind, count in enumerate(counts34(hand)) for _ in range(count))
        probability = max(0.0, 1.0 - forbidden * self.config.tanyao_forbidden_tile_penalty)
        guaranteed = forbidden == 0 and bool(state.self_melds)
        return YakuEvaluation(
            "tanyao",
            _state_for_probability(probability, guaranteed),
            probability,
            probability,
            probability,
            (f"手牌中剩余{forbidden}张幺九牌",),
        )

    def _chiitoitsu(self, hand: list[int], state: PublicState) -> YakuEvaluation:
        if state.self_melds:
            return YakuEvaluation(
                "chiitoitsu", YakuState.IMPOSSIBLE, 0.0, 0.0, 0.0, ("已经副露，七对子不成立",)
            )
        counts = counts34(hand)
        pairs = sum(count >= 2 for count in counts)
        unique = sum(count > 0 for count in counts)
        shanten = 6 - pairs + max(0, 7 - unique)
        probability = _probability_for_shanten(shanten, self.config)
        expected_han = 2.0 * probability
        return YakuEvaluation(
            "chiitoitsu",
            _state_for_probability(probability),
            probability,
            expected_han,
            expected_han,
            (f"七对子{shanten}向听，已有{pairs}组对子",),
        )

    def _flush(self, hand: list[int], state: PublicState) -> YakuEvaluation:
        counts = counts34(hand)
        total = max(1, sum(counts))
        is_open = bool(state.self_melds)
        meld_kinds = [kind for meld in state.self_melds for kind in meld]
        best = (0.0, 0.0, "", YakuState.IMPOSSIBLE, ())
        for suit_index, suit_name in enumerate(("万", "筒", "索")):
            start = suit_index * 9
            main = sum(counts[start : start + 9])
            honors = sum(counts[27:])
            off_suit = total - main - honors
            incompatible_meld = any(kind < 27 and kind // 9 != suit_index for kind in meld_kinds)
            if incompatible_meld:
                continue

            honitsu_probability = max(0.0, (main + honors) / total - off_suit * self.config.flush_off_suit_penalty)
            chinitsu_probability = max(
                0.0,
                main / total
                - off_suit * self.config.flush_off_suit_penalty
                - honors * self.config.flush_honor_penalty_for_chinitsu,
            )
            honitsu_han = 2.0 if is_open else 3.0
            chinitsu_han = 5.0 if is_open else 6.0
            honitsu_value = honitsu_probability * honitsu_han
            chinitsu_value = chinitsu_probability * chinitsu_han
            if chinitsu_value > honitsu_value:
                value, probability, label = chinitsu_value, chinitsu_probability, f"清一色（{suit_name}）"
            else:
                value, probability, label = honitsu_value, honitsu_probability, f"混一色（{suit_name}）"
            if value > best[0]:
                best = (
                    value,
                    probability,
                    label,
                    _state_for_probability(probability, off_suit == 0 and (honors == 0 or label.startswith("混"))),
                    (f"{suit_name}子{main}张、字牌{honors}张、异色{off_suit}张",),
                )
        value, probability, label, yaku_state, evidence = best
        return YakuEvaluation("flush", yaku_state, probability, value, value, (label, *evidence))
