"""
Fertility Manager — reproduction constraints and mortality.

Controls who can reproduce, when, and handles maternal/child mortality.
All parameters from config.fertility_config.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from seldon.core.agent import Agent
from seldon.core.config import ExperimentConfig


class FertilityManager:
    """
    Manages reproduction eligibility, probability, and mortality.

    Parameters come from config.fertility_config:
    - Age windows for fertility
    - Birth spacing constraints
    - Maternal and child mortality
    - Societal pressure toward target family size
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        fc = config.fertility_config

        self.female_fertility_start: int = fc["female_fertility_start"]
        self.female_fertility_end: int = fc["female_fertility_end"]
        self.male_fertility_start: int = fc["male_fertility_start"]
        self.min_birth_spacing: int = fc["min_birth_spacing_generations"]
        self.max_children_per_gen: int = fc["max_children_per_generation"]
        self.maternal_mortality_rate: float = fc["maternal_mortality_rate"]
        self.child_mortality_rate: float = fc["child_mortality_rate"]
        self.societal_pressure: float = fc["societal_fertility_pressure"]
        self.target_children: float = fc["target_children_mean"]

    def can_reproduce(
        self, parent1: Agent, parent2: Agent, generation: int,
    ) -> bool:
        """
        Check if a pair meets the basic requirements for reproduction.

        Checks:
        - Both alive
        - Both within fertility age windows
        - Birth spacing constraint met
        - Max children per generation not exceeded
        """
        if not parent1.is_alive or not parent2.is_alive:
            return False

        # Age check — use the broader window (either parent can be
        # in their fertility window)
        if not self._in_fertility_window(parent1) and not self._in_fertility_window(parent2):
            return False

        # Birth spacing
        for p in (parent1, parent2):
            if p.last_birth_generation is not None:
                gap = generation - p.last_birth_generation
                if gap < self.min_birth_spacing:
                    return False

        # Max children this generation (check births this gen)
        for p in (parent1, parent2):
            if p.last_birth_generation == generation:
                return False  # Already had a child this generation

        return True

    def will_reproduce(
        self,
        parent1: Agent,
        parent2: Agent,
        generation: int,
        rng: np.random.Generator,
    ) -> bool:
        """
        Probabilistic reproduction decision.

        Base probability is modified by:
        - Societal fertility pressure
        - Current number of children vs target
        """
        if not self.can_reproduce(parent1, parent2, generation):
            return False

        # Count existing shared children
        shared_children = set(parent1.children_ids) & set(parent2.children_ids)
        num_children = len(shared_children)

        # Base probability influenced by societal pressure
        base_prob = 0.3 + self.societal_pressure * 0.2

        # Adjust based on distance from target
        if num_children >= self.target_children:
            # Already at or above target — reduce probability
            overshoot = num_children - self.target_children
            base_prob *= max(0.1, 1.0 - overshoot * 0.3)
        else:
            # Below target — increase probability slightly
            undershoot = self.target_children - num_children
            base_prob *= min(1.5, 1.0 + undershoot * 0.1)

        base_prob = min(base_prob, 0.9)
        return bool(rng.random() < base_prob)

    def check_child_mortality(self, rng: np.random.Generator) -> bool:
        """
        Check if a newborn survives.

        Returns True if child dies (mortality event).
        """
        return bool(rng.random() < self.child_mortality_rate)

    def check_maternal_mortality(self, rng: np.random.Generator) -> bool:
        """
        Check if the birthing parent dies during childbirth.

        Returns True if parent dies (mortality event).
        """
        return bool(rng.random() < self.maternal_mortality_rate)

    def _in_fertility_window(self, agent: Agent) -> bool:
        """Check if an agent is within the fertility age window."""
        # Use the broader window encompassing both male and female ranges
        min_age = min(self.female_fertility_start, self.male_fertility_start)
        max_age = self.female_fertility_end
        return min_age <= agent.age <= max_age
