"""Adapter from an MJAI-speaking bot to the common decision interface."""

from typing import Any, Protocol

from riichienv import Action, Observation


MjaiResponse = str | dict[str, Any]


class MjaiBot(Protocol):
    """Minimal protocol implemented by Mortal's libriichi bot wrapper."""

    def react(self, event: str) -> MjaiResponse | None:
        """Consume one MJAI JSON event and optionally return an action."""


class MjaiEngineError(RuntimeError):
    """Raised when an MJAI bot fails to produce a selectable action."""


class MjaiBotEngine:
    """Feed unseen observation events to a stateful MJAI bot."""

    def __init__(self, bot: MjaiBot) -> None:
        self._bot = bot

    def act(self, observation: Observation) -> Action:
        response: MjaiResponse | None = None
        events = observation.new_events()
        for event in events:
            candidate = self._bot.react(event)
            if candidate is not None:
                response = candidate

        if response is None:
            raise MjaiEngineError(
                f"MJAI bot produced no response after consuming {len(events)} new event(s)"
            )

        action = observation.select_action_from_mjai(response)
        if action is None:
            raise MjaiEngineError(f"MJAI response does not match a legal action: {response!r}")
        return action

