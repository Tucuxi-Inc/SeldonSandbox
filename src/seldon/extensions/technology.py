"""
Technology Extension — breakthrough-driven advancement.

R3/R4 breakthroughs increment a global tech level. Higher tech reduces
mortality and increases settlement carrying capacity.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from seldon.extensions.base import SimulationExtension

if TYPE_CHECKING:
    from seldon.core.agent import Agent
    from seldon.core.config import ExperimentConfig


class TechnologyExtension(SimulationExtension):
    """Global tech level driven by deep/sacrificial processing breakthroughs."""

    @property
    def name(self) -> str:
        return "technology"

    @property
    def description(self) -> str:
        return "Technology advancement from R3/R4 breakthroughs"

    def get_default_config(self) -> dict[str, Any]:
        return {
            "tech_per_breakthrough": 0.1,
            "tech_capacity_multiplier": 1.5,
            "tech_mortality_reduction": 0.05,
            "tech_fertility_bonus": 0.02,
            "tech_decay_enabled": False,
            "tech_decay_rate": 0.01,
        }

    def __init__(self) -> None:
        self.tech_level: float = 0.0
        self._breakthroughs_this_gen: int = 0

    def _get_config(self, config: ExperimentConfig) -> dict[str, Any]:
        defaults = self.get_default_config()
        overrides = config.extensions.get("technology", {})
        defaults.update(overrides)
        return defaults

    def on_simulation_start(
        self, population: list[Agent], config: ExperimentConfig,
    ) -> None:
        self.tech_level = 0.0

    def on_generation_end(
        self, generation: int, population: list[Agent],
        config: ExperimentConfig,
    ) -> None:
        """Count breakthroughs and advance tech level."""
        tech = self._get_config(config)

        # Count breakthroughs from agents who had them this generation
        # (agents with high recent contribution in R3/R4)
        from seldon.core.processing import ProcessingRegion
        self._breakthroughs_this_gen = sum(
            1 for a in population
            if (a.processing_region in (
                ProcessingRegion.DEEP, ProcessingRegion.SACRIFICIAL,
            ) and a.contribution_history
            and a.contribution_history[-1] > 1.0)
        )

        # Advance tech
        self.tech_level += (
            self._breakthroughs_this_gen * tech["tech_per_breakthrough"]
        )

        # Optional decay
        if tech["tech_decay_enabled"]:
            self.tech_level = max(
                0.0, self.tech_level - tech["tech_decay_rate"],
            )

    def modify_mortality(
        self, agent: Agent, base_rate: float,
        config: ExperimentConfig,
    ) -> float:
        """Reduce mortality proportional to tech level."""
        tech = self._get_config(config)
        reduction = self.tech_level * tech["tech_mortality_reduction"]
        return float(np.clip(base_rate - reduction, 0.0, 1.0))

    def get_metrics(self, population: list[Agent]) -> dict[str, Any]:
        tech = self.get_default_config()
        return {
            "tech_level": self.tech_level,
            "breakthroughs_this_gen": self._breakthroughs_this_gen,
            "effective_capacity_multiplier": (
                1.0 + self.tech_level * (tech["tech_capacity_multiplier"] - 1.0)
            ),
        }
