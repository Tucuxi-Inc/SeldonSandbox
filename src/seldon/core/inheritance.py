"""
Birth-order-based trait inheritance.

Rules are configurable per birth order position:
  1st: "worst"   — less desirable parent value per trait
  2nd: "weirdest" — whichever parent is farther from population mean
  3rd: "best"    — more desirable parent value per trait
  4th+: "random_weighted" — weighted random mix

All rules use the TraitSystem desirability vector and are
configurable via ExperimentConfig.birth_order_rules.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from seldon.core.agent import Agent
    from seldon.core.config import ExperimentConfig


class InheritanceEngine:
    """Generates child traits from two parents based on birth order rules."""

    RULE_TYPES = ("worst", "best", "weirdest", "random_weighted", "average")

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.ts = config.trait_system

    def inherit(
        self,
        parent1: Agent,
        parent2: Agent,
        birth_order: int,
        population: list[Agent],
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """
        Generate child traits.

        1. Look up rule for this birth order position
        2. Apply rule to parents' traits
        3. Add Gaussian environmental noise
        4. Clamp to [0, 1]
        """
        rng = rng or np.random.default_rng()
        rule = self.config.birth_order_rules.get(birth_order, "random_weighted")

        if rule not in self.RULE_TYPES:
            raise ValueError(f"Unknown inheritance rule: '{rule}'")

        method = getattr(self, f"_inherit_{rule}")
        base = method(parent1.traits, parent2.traits, population, rng)

        noise = rng.normal(0, self.config.inheritance_noise_sigma, size=self.ts.count)
        return np.clip(base + noise, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Rule implementations
    # ------------------------------------------------------------------

    def _inherit_worst(
        self, p1: np.ndarray, p2: np.ndarray,
        population: list[Agent], rng: np.random.Generator,
    ) -> np.ndarray:
        """First-born: gets less desirable value per trait."""
        desirability = self.ts.desirability
        result = np.empty(self.ts.count)
        for i in range(self.ts.count):
            if desirability[i] > 0:      # Positive trait -> take minimum
                result[i] = min(p1[i], p2[i])
            elif desirability[i] < 0:    # Negative trait -> take maximum (worse)
                result[i] = max(p1[i], p2[i])
            else:                         # Neutral -> average
                result[i] = (p1[i] + p2[i]) / 2
        return result

    def _inherit_best(
        self, p1: np.ndarray, p2: np.ndarray,
        population: list[Agent], rng: np.random.Generator,
    ) -> np.ndarray:
        """Third-born: gets more desirable value per trait."""
        desirability = self.ts.desirability
        result = np.empty(self.ts.count)
        for i in range(self.ts.count):
            if desirability[i] > 0:      # Positive trait -> take maximum
                result[i] = max(p1[i], p2[i])
            elif desirability[i] < 0:    # Negative trait -> take minimum (better)
                result[i] = min(p1[i], p2[i])
            else:
                result[i] = (p1[i] + p2[i]) / 2
        return result

    def _inherit_weirdest(
        self, p1: np.ndarray, p2: np.ndarray,
        population: list[Agent], rng: np.random.Generator,
    ) -> np.ndarray:
        """Second-born: gets whichever parent value is farther from population mean."""
        if not population:
            return (p1 + p2) / 2

        pop_traits = np.array([a.traits for a in population if a.is_alive])
        if len(pop_traits) == 0:
            return (p1 + p2) / 2

        pop_mean = pop_traits.mean(axis=0)
        result = np.empty(self.ts.count)
        for i in range(self.ts.count):
            if abs(p1[i] - pop_mean[i]) >= abs(p2[i] - pop_mean[i]):
                result[i] = p1[i]
            else:
                result[i] = p2[i]
        return result

    def _inherit_random_weighted(
        self, p1: np.ndarray, p2: np.ndarray,
        population: list[Agent], rng: np.random.Generator,
    ) -> np.ndarray:
        """Fourth+: per-trait random weighted mix of both parents."""
        weights = rng.uniform(0, 1, size=self.ts.count)
        return p1 * weights + p2 * (1 - weights)

    def _inherit_average(
        self, p1: np.ndarray, p2: np.ndarray,
        population: list[Agent], rng: np.random.Generator,
    ) -> np.ndarray:
        """Simple midpoint average."""
        return (p1 + p2) / 2
