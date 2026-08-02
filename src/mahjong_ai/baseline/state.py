"""Stable state projection from RiichiEnv observations."""

from dataclasses import dataclass
from typing import Any

from riichienv import Observation

from mahjong_ai.baseline.tiles import counts34, tile_kind


@dataclass(frozen=True)
class PublicState:
    player_id: int
    hand: tuple[int, ...]
    visible_counts: tuple[int, ...]
    discards: tuple[tuple[int, ...], ...]
    dora_indicators: tuple[int, ...]
    riichi_declared: tuple[bool, ...]
    round_wind: int
    dealer: int
    self_melds: tuple[tuple[int, ...], ...] = ()

    @classmethod
    def from_observation(cls, observation: Observation) -> "PublicState":
        raw: dict[str, Any] = observation.to_dict()
        player_id = int(raw["player_id"])
        hand = tuple(int(tile) for tile in raw["hands"][player_id])
        discards = tuple(tuple(int(tile) for tile in river) for river in raw["discards"])
        dora_indicators = tuple(int(tile) for tile in raw["dora_indicators"])

        visible_tiles = list(hand) + list(dora_indicators)
        visible_tiles.extend(tile for river in discards for tile in river)
        parsed_self_melds: list[tuple[int, ...]] = []
        for meld_owner, player_melds in enumerate(raw["melds"]):
            for meld in player_melds:
                if isinstance(meld, dict):
                    meld_tiles = tuple(int(tile) for tile in meld.get("tiles", []))
                elif isinstance(meld, (list, tuple)):
                    meld_tiles = tuple(int(tile) for tile in meld)
                else:
                    meld_tiles = ()
                visible_tiles.extend(meld_tiles)
                if meld_owner == player_id:
                    parsed_self_melds.append(tuple(tile_kind(tile) for tile in meld_tiles))

        return cls(
            player_id=player_id,
            hand=hand,
            visible_counts=tuple(counts34(visible_tiles)),
            discards=discards,
            dora_indicators=dora_indicators,
            riichi_declared=tuple(bool(value) for value in raw["riichi_declared"]),
            round_wind=int(raw["round_wind"]),
            dealer=int(raw["oya"]),
            self_melds=tuple(parsed_self_melds),
        )

    @property
    def seat_wind(self) -> int:
        """Return 0=East, 1=South, 2=West, 3=North."""
        return (self.player_id - self.dealer) % 4

    @property
    def value_honor_kinds(self) -> frozenset[int]:
        """Dragons plus this player's round and seat winds in 34-tile form."""
        return frozenset({31, 32, 33, 27 + self.round_wind, 27 + self.seat_wind})

    def value_honor_han(self, kind: int) -> int:
        """Return actual yakuhai han, including double-wind overlap."""
        han = int(kind in {31, 32, 33})
        han += int(kind == 27 + self.round_wind)
        han += int(kind == 27 + self.seat_wind)
        return han

    def is_genbutsu(self, kind: int) -> bool:
        for opponent, declared in enumerate(self.riichi_declared):
            if opponent != self.player_id and declared:
                if not any(tile_kind(tile) == kind for tile in self.discards[opponent]):
                    return False
        return any(
            declared for opponent, declared in enumerate(self.riichi_declared) if opponent != self.player_id
        )
