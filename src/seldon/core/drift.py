"""
Experience-based trait drift.

Traits change over time based on:
- Random life experiences (Gaussian noise scaled by drift rate)
- Processing region effects (e.g., sacrificial deepens depth_drive)
- Age-based stability (older agents drift less)
- Per-trait stability values from TraitSystem
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from seldon.core.agent import Agent
    from seldon.core.config import ExperimentConfig


class TraitDriftEngine:
    """Applies trait drift based on experiences, age, and processing region."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.ts = config.trait_system

    def drift_traits(
        self, agent: Agent, rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """
        Apply random drift to agent traits.

        Drift magnitude is scaled by:
        - config.trait_drift_rate (global rate)
        - trait-specific stability (higher stability = less drift)
        - age factor: 1 / (1 + age * config.trait_drift_age_factor)
        """
        rng = rng or np.random.default_rng()

        age_dampening = 1.0 / (1.0 + agent.age * self.config.trait_drift_age_factor)
        # Per-trait drift scale: base_rate * (1 - stability) * age_dampening
        drift_scale = (
            self.config.trait_drift_rate
            * (1.0 - self.ts.stability)
            * age_dampening
        )

        noise = rng.normal(0, drift_scale)
        return np.clip(agent.traits + noise, 0.0, 1.0)

    def apply_region_effects(
        self, agent: Agent, rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """
        Apply processing-region-specific trait modifications.

        Region effects are defined in config.region_effects as
        {region_name: {trait_name: modifier}}.
        """
        region_name = agent.processing_region.value
        effects = self.config.region_effects.get(region_name, {})

        if not effects:
            return agent.traits

        modified = agent.traits.copy()
        for trait_name, modifier in effects.items():
            try:
                idx = self.ts.trait_index(trait_name)
                modified[idx] += modifier
            except KeyError:
                pass  # Trait not in current preset; skip silently

        return np.clip(modified, 0.0, 1.0)

    def update_suffering(self, agent: Agent) -> None:
        """Update agent suffering based on processing region."""
        from seldon.core.processing import ProcessingRegion

        suffering_rates = {
            ProcessingRegion.UNDER_PROCESSING: -0.05,
            ProcessingRegion.OPTIMAL: -0.03,
            ProcessingRegion.DEEP: 0.02,
            ProcessingRegion.SACRIFICIAL: 0.08,
            ProcessingRegion.PATHOLOGICAL: 0.12,
        }
        rate = suffering_rates.get(agent.processing_region, 0.0)
        agent.suffering = float(np.clip(agent.suffering + rate, 0.0, 1.0))

    def update_burnout(self, agent: Agent) -> None:
        """Update burnout level. Recovers in R1/R2, accumulates in R4/R5."""
        from seldon.core.processing import ProcessingRegion

        burnout_rates = {
            ProcessingRegion.UNDER_PROCESSING: -0.02,
            ProcessingRegion.OPTIMAL: -0.01,
            ProcessingRegion.DEEP: 0.01,
            ProcessingRegion.SACRIFICIAL: 0.05,
            ProcessingRegion.PATHOLOGICAL: 0.08,
        }
        rate = burnout_rates.get(agent.processing_region, 0.0)
        agent.burnout_level = float(np.clip(agent.burnout_level + rate, 0.0, 1.0))
