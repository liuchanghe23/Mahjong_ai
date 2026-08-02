"""Metric extraction from completed MJAI match logs."""

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class DecisionTiming:
    count: int = 0
    total_ns: int = 0
    max_ns: int = 0

    def add(self, elapsed_ns: int) -> None:
        self.count += 1
        self.total_ns += elapsed_ns
        self.max_ns = max(self.max_ns, elapsed_ns)

    @property
    def mean_ms(self) -> float:
        return self.total_ns / self.count / 1_000_000 if self.count else 0.0

    @property
    def max_ms(self) -> float:
        return self.max_ns / 1_000_000


@dataclass
class PlayerMatchStats:
    player_id: int
    initial_score: int
    final_score: int
    rank: int
    rounds: int = 0
    wins: int = 0
    tsumo_wins: int = 0
    ron_wins: int = 0
    deal_ins: int = 0
    riichi: int = 0
    exhaustive_draws: int = 0
    draw_tenpai_known: int = 0
    draw_tenpai_unknown: int = 0
    win_points: int = 0
    deal_in_points: int = 0
    timing: DecisionTiming = field(default_factory=DecisionTiming)

    @property
    def point_delta(self) -> int:
        return self.final_score - self.initial_score


@dataclass
class AggregateStats:
    engine: str
    matches: int
    player_samples: int
    rounds: int
    wins: int
    tsumo_wins: int
    ron_wins: int
    deal_ins: int
    riichi: int
    exhaustive_draws: int
    draw_tenpai_known: int
    draw_tenpai_unknown: int
    total_point_delta: int
    total_win_points: int
    total_deal_in_points: int
    rank_counts: dict[int, int]
    decision_count: int
    decision_total_ns: int
    decision_max_ns: int

    def rate(self, numerator: int, denominator: int) -> float:
        return numerator / denominator if denominator else 0.0

    @property
    def average_rank(self) -> float:
        return sum(rank * count for rank, count in self.rank_counts.items()) / self.player_samples

    @property
    def average_point_delta(self) -> float:
        return self.total_point_delta / self.player_samples

    @property
    def win_rate_per_round(self) -> float:
        return self.rate(self.wins, self.rounds)

    @property
    def deal_in_rate_per_round(self) -> float:
        return self.rate(self.deal_ins, self.rounds)

    @property
    def riichi_rate_per_round(self) -> float:
        return self.rate(self.riichi, self.rounds)

    @property
    def known_draw_tenpai_rate(self) -> float:
        known_samples = self.draw_tenpai_known + (self.exhaustive_draws - self.draw_tenpai_unknown - self.draw_tenpai_known)
        return self.rate(self.draw_tenpai_known, known_samples)

    @property
    def average_win_points(self) -> float:
        return self.rate(self.total_win_points, self.wins)

    @property
    def average_deal_in_points(self) -> float:
        return self.rate(self.total_deal_in_points, self.deal_ins)

    @property
    def mean_decision_ms(self) -> float:
        return self.rate(self.decision_total_ns, self.decision_count) / 1_000_000

    @property
    def max_decision_ms(self) -> float:
        return self.decision_max_ns / 1_000_000

    def to_dict(self) -> dict[str, Any]:
        raw = asdict(self)
        raw["derived"] = {
            "average_rank": self.average_rank,
            "average_point_delta": self.average_point_delta,
            "win_rate_per_round": self.win_rate_per_round,
            "deal_in_rate_per_round": self.deal_in_rate_per_round,
            "riichi_rate_per_round": self.riichi_rate_per_round,
            "known_draw_tenpai_rate": self.known_draw_tenpai_rate,
            "average_win_points": self.average_win_points,
            "average_deal_in_points": self.average_deal_in_points,
            "mean_decision_ms": self.mean_decision_ms,
            "max_decision_ms": self.max_decision_ms,
        }
        return raw


def extract_player_stats(
    events: list[dict[str, Any]],
    scores: tuple[int, ...],
    ranks: tuple[int, ...],
    timings: dict[int, DecisionTiming],
) -> list[PlayerMatchStats]:
    start = next(event for event in events if event.get("type") == "start_kyoku")
    initial_scores = [int(value) for value in start["scores"]]
    rounds = sum(event.get("type") == "start_kyoku" for event in events)
    stats = [
        PlayerMatchStats(pid, initial_scores[pid], scores[pid], ranks[pid], rounds=rounds, timing=timings[pid])
        for pid in range(len(scores))
    ]

    for event in events:
        event_type = event.get("type")
        if event_type == "reach":
            stats[int(event["actor"])].riichi += 1
        elif event_type == "hora":
            actor, target = int(event["actor"]), int(event["target"])
            deltas = [int(value) for value in event.get("deltas", [0] * len(scores))]
            stats[actor].wins += 1
            stats[actor].win_points += max(0, deltas[actor])
            if actor == target:
                stats[actor].tsumo_wins += 1
            else:
                stats[actor].ron_wins += 1
                stats[target].deal_ins += 1
                stats[target].deal_in_points += max(0, -deltas[target])
        elif event_type == "ryukyoku" and event.get("reason") == "exhaustive_draw":
            deltas = [int(value) for value in event.get("deltas", [0] * len(scores))]
            ambiguous = all(value == 0 for value in deltas)
            for pid, value in enumerate(deltas):
                stats[pid].exhaustive_draws += 1
                if ambiguous:
                    stats[pid].draw_tenpai_unknown += 1
                elif value > 0:
                    stats[pid].draw_tenpai_known += 1
    return stats


def aggregate(engine: str, matches: list[list[PlayerMatchStats]]) -> AggregateStats:
    players = [player for match in matches for player in match]
    timing = [player.timing for player in players]
    return AggregateStats(
        engine=engine,
        matches=len(matches),
        player_samples=len(players),
        rounds=sum(player.rounds for player in players),
        wins=sum(player.wins for player in players),
        tsumo_wins=sum(player.tsumo_wins for player in players),
        ron_wins=sum(player.ron_wins for player in players),
        deal_ins=sum(player.deal_ins for player in players),
        riichi=sum(player.riichi for player in players),
        exhaustive_draws=sum(player.exhaustive_draws for player in players),
        draw_tenpai_known=sum(player.draw_tenpai_known for player in players),
        draw_tenpai_unknown=sum(player.draw_tenpai_unknown for player in players),
        total_point_delta=sum(player.point_delta for player in players),
        total_win_points=sum(player.win_points for player in players),
        total_deal_in_points=sum(player.deal_in_points for player in players),
        rank_counts={rank: sum(player.rank == rank for player in players) for rank in range(1, 5)},
        decision_count=sum(item.count for item in timing),
        decision_total_ns=sum(item.total_ns for item in timing),
        decision_max_ns=max((item.max_ns for item in timing), default=0),
    )

