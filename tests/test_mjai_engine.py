import pytest

from mahjong_ai.mjai_engine import MjaiBotEngine, MjaiEngineError


class FakeBot:
    def __init__(self, responses: list[dict[str, str] | None]) -> None:
        self.responses = iter(responses)
        self.events: list[str] = []

    def react(self, event: str) -> dict[str, str] | None:
        self.events.append(event)
        return next(self.responses)


class FakeObservation:
    def __init__(self, selected: object | None = None) -> None:
        self.selected = selected
        self.received: object | None = None

    def new_events(self) -> list[str]:
        return ['{"type":"start_game"}', '{"type":"tsumo","actor":0,"pai":"1m"}']

    def select_action_from_mjai(self, response: object) -> object | None:
        self.received = response
        return self.selected


def test_mjai_adapter_feeds_events_and_selects_last_response() -> None:
    expected_action = object()
    bot = FakeBot([None, {"type": "dahai", "pai": "1m"}])
    observation = FakeObservation(selected=expected_action)

    action = MjaiBotEngine(bot).act(observation)  # type: ignore[arg-type]

    assert action is expected_action
    assert len(bot.events) == 2
    assert observation.received == {"type": "dahai", "pai": "1m"}


def test_mjai_adapter_rejects_missing_response() -> None:
    bot = FakeBot([None, None])

    with pytest.raises(MjaiEngineError, match="produced no response"):
        MjaiBotEngine(bot).act(FakeObservation())  # type: ignore[arg-type]


def test_mjai_adapter_rejects_unselectable_response() -> None:
    bot = FakeBot([None, {"type": "dahai", "pai": "9m"}])

    with pytest.raises(MjaiEngineError, match="does not match a legal action"):
        MjaiBotEngine(bot).act(FakeObservation())  # type: ignore[arg-type]

