"""Built-in engines used before Mortal is connected."""

from riichienv import Action, Observation
from riichienv.agents import RandomAgent


class RandomEngine:
    """Deterministic baseline engine when constructed with a seed."""

    def __init__(self, seed: int) -> None:
        self._agent = RandomAgent(seed=seed)

    def act(self, observation: Observation) -> Action:
        return self._agent.act(observation)

