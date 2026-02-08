"""
Multi-factor attraction model.

Calculates attraction between two agents using configurable weights
for similarity, complementarity, universal attractiveness, social
proximity, age compatibility, and random chemistry.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from seldon.core.agent import Agent
    from seldon.core.config import ExperimentConfig


class AttractionModel:
    """Computes attraction scores between agents. All weights from config."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.ts = config.trait_system
        self.weights = config.attraction_weights

    def calculate(
        self, agent1: Agent, agent2: Agent,
        rng: np.random.Generator | None = None,
    ) -> float:
        """
        Compute attraction score in [0, 1].

        Components are weighted by config.attraction_weights.
        """
        rng = rng or np.random.default_rng()

        components = {
            "similarity": self._similarity(agent1, agent2),
            "complementarity": self._complementarity(agent1, agent2),
            "universal_attractiveness": self._universal_attractiveness(agent2),
            "social_proximity": self._social_proximity(agent1, agent2),
            "age_compatibility": self._age_compatibility(agent1, agent2),
            "random_chemistry": rng.uniform(0, 1),
        }

        score = sum(
            self.weights.get(k, 0.0) * v
            for k, v in components.items()
        )
        return float(np.clip(score, 0.0, 1.0))

    def _similarity(self, a1: Agent, a2: Agent) -> float:
        """Cosine similarity of trait vectors."""
        dot = np.dot(a1.traits, a2.traits)
        norm1 = np.linalg.norm(a1.traits)
        norm2 = np.linalg.norm(a2.traits)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(dot / (norm1 * norm2))

    def _complementarity(self, a1: Agent, a2: Agent) -> float:
        """
        How well traits complement each other.

        Agents are complementary when one is strong where the other is weak,
        particularly for neutral-desirability traits.
        """
        desirability = self.ts.desirability
        neutral_mask = desirability == 0
        if not neutral_mask.any():
            return 0.5

        # For neutral traits, measure how well they balance each other
        combined = a1.traits[neutral_mask] + a2.traits[neutral_mask]
        # Closest to 1.0 (balanced) = most complementary
        balance_score = 1.0 - np.abs(combined - 1.0).mean()
        return float(np.clip(balance_score, 0.0, 1.0))

    def _universal_attractiveness(self, agent: Agent) -> float:
        """
        Traits that are universally considered attractive.

        High positive-desirability traits = more universally attractive.
        """
        desirability = self.ts.desirability
        positive_mask = desirability > 0
        if not positive_mask.any():
            return 0.5
        return float(agent.traits[positive_mask].mean())

    def _social_proximity(self, a1: Agent, a2: Agent) -> float:
        """
        Social bond strength between agents.

        Returns the existing bond strength, or a base value for strangers.
        """
        bond = a1.social_bonds.get(a2.id, 0.0)
        # Agents in the same generation get a slight proximity bonus
        gen_bonus = 0.2 if a1.generation == a2.generation else 0.0
        return float(np.clip(bond + gen_bonus, 0.0, 1.0))

    def _age_compatibility(self, a1: Agent, a2: Agent) -> float:
        """
        Age compatibility — closer ages score higher.

        Uses a Gaussian-like decay centered on zero age difference.
        """
        diff = abs(a1.age - a2.age)
        # Score decays with age difference; sigma ~= 5 years
        return float(np.exp(-0.5 * (diff / 5.0) ** 2))
