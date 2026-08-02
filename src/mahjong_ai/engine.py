"""Common decision-engine contracts and action safety checks."""

from typing import Protocol

from riichienv import Action, Observation


class DecisionEngine(Protocol):
    """Interface shared by random, heuristic, and future Mortal engines."""

    def act(self, observation: Observation) -> Action:
        """Return one action for the supplied observation."""


class IllegalEngineActionError(RuntimeError):
    """Raised when an engine returns an action outside the legal action set."""


def action_signature(action: Action) -> str:
    """Create a stable MJAI representation suitable for comparisons and logs."""
    return action.to_mjai()


def require_legal_action(observation: Observation, action: Action) -> Action:
    """Reject an engine action unless it is present in the current legal set."""
    legal_signatures = {action_signature(candidate) for candidate in observation.legal_actions()}
    proposed = action_signature(action)
    if proposed not in legal_signatures:
        raise IllegalEngineActionError(
            f"Engine returned illegal action {proposed}; legal actions: {sorted(legal_signatures)}"
        )
    return action

