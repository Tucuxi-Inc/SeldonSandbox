"""
RSH Five Regions processing model.

Classifies agents into one of five cognitive processing regions based on
their depth_drive trait, burnout level, and productive potential.
All thresholds come from ExperimentConfig — never hardcoded.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from seldon.core.agent import Agent
    from seldon.core.config import ExperimentConfig


class ProcessingRegion(Enum):
    """RSH Five Regions of cognitive processing."""
    UNDER_PROCESSING = "under_processing"
    OPTIMAL = "optimal"
    DEEP = "deep"
    SACRIFICIAL = "sacrificial"
    PATHOLOGICAL = "pathological"


class ProcessingClassifier:
    """
    Classifies agents into RSH processing regions.

    Uses depth_drive trait + burnout + suffering to determine region.
    The R4 vs R5 distinction (productive vs unproductive suffering)
    is determined by the productive potential calculation.
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.ts = config.trait_system
        self._depth_idx = self.ts.trait_index("depth_drive")

    def classify(self, agent: Agent) -> ProcessingRegion:
        """Classify an agent into a processing region."""
        depth = agent.traits[self._depth_idx]
        thresholds = self.config.region_thresholds

        if depth < thresholds["under_to_optimal"]:
            return ProcessingRegion.UNDER_PROCESSING
        elif depth < thresholds["optimal_to_deep"]:
            return ProcessingRegion.OPTIMAL
        elif depth < thresholds["deep_to_extreme"]:
            return ProcessingRegion.DEEP
        else:
            potential = self._productive_potential(agent)
            if potential >= thresholds["productive_potential_threshold"]:
                return ProcessingRegion.SACRIFICIAL
            else:
                return ProcessingRegion.PATHOLOGICAL

    def _productive_potential(self, agent: Agent) -> float:
        """
        Determine whether extreme processing is productive (R4) or destructive (R5).

        Based on creativity, resilience, and burnout level with configurable weights.
        """
        weights = self.config.productive_weights

        creativity_idx = self.ts.trait_index("creativity")
        resilience_idx = self.ts.trait_index("resilience")

        potential = (
            agent.traits[creativity_idx] * weights["creativity"]
            + agent.traits[resilience_idx] * weights["resilience"]
            - agent.burnout_level * weights["burnout_penalty"]
        )
        return float(np.clip(potential, 0.0, 1.0))
