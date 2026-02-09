"""
Epigenetic model for the Seldon Sandbox.

Epigenetic markers are environmentally-triggered toggles that modify trait
values without changing the underlying genome. Markers can be activated by
sustained conditions (e.g., high suffering for multiple generations) and
can be inherited transgenerationally with configurable probability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from seldon.core.agent import Agent
    from seldon.core.config import ExperimentConfig
    from seldon.core.traits import TraitSystem


# ---------------------------------------------------------------------------
# Epigenetic marker definitions
# ---------------------------------------------------------------------------

@dataclass
class EpigeneticMarker:
    """Definition of a single epigenetic marker."""
    name: str
    target_trait: str
    modifier: float  # How much to shift the trait when active
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "target_trait": self.target_trait,
            "modifier": self.modifier,
            "description": self.description,
        }


# Default markers — these model real epigenetic phenomena
DEFAULT_MARKERS: list[EpigeneticMarker] = [
    EpigeneticMarker(
        name="stress_resilience",
        target_trait="resilience",
        modifier=0.05,
        description="Activates when suffering > 0.6 for sustained periods; boosts resilience",
    ),
    EpigeneticMarker(
        name="creative_amplification",
        target_trait="creativity",
        modifier=0.08,
        description="Activates in deep/sacrificial processing regions; boosts creativity",
    ),
    EpigeneticMarker(
        name="social_withdrawal",
        target_trait="extraversion",
        modifier=-0.05,
        description="Activates when trust is low and agent has been hurt; reduces extraversion",
    ),
    EpigeneticMarker(
        name="resource_conservation",
        target_trait="conscientiousness",
        modifier=0.05,
        description="Activates during resource scarcity; boosts conscientiousness",
    ),
    EpigeneticMarker(
        name="trauma_sensitivity",
        target_trait="neuroticism",
        modifier=0.03,
        description="Transgenerational: activates when parent had high suffering; increases neuroticism",
    ),
]


class EpigeneticModel:
    """Manages epigenetic state activation, deactivation, and inheritance."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self._ec = config.epigenetics_config
        self.markers = {m.name: m for m in DEFAULT_MARKERS}

    @property
    def enabled(self) -> bool:
        return self._ec.get("enabled", True)

    @property
    def transgenerational_rate(self) -> float:
        return self._ec.get("transgenerational_rate", 0.3)

    @property
    def activation_threshold_generations(self) -> int:
        return self._ec.get("activation_threshold_generations", 3)

    @property
    def max_active_markers(self) -> int:
        return self._ec.get("max_active_markers", 5)

    # ------------------------------------------------------------------
    # State initialization
    # ------------------------------------------------------------------
    def initialize_state(self) -> dict[str, bool]:
        """Return a fresh epigenetic state with all markers inactive."""
        if not self.enabled:
            return {}
        return {m.name: False for m in DEFAULT_MARKERS}

    # ------------------------------------------------------------------
    # Activation / deactivation
    # ------------------------------------------------------------------
    def update_epigenetic_state(
        self, agent: Agent, trait_system: TraitSystem,
    ) -> dict[str, bool]:
        """
        Check activation/deactivation conditions for each marker.

        Modifies agent.epigenetic_state in place and returns it.
        """
        if not self.enabled:
            return agent.epigenetic_state

        state = dict(agent.epigenetic_state) if agent.epigenetic_state else self.initialize_state()
        active_count = sum(1 for v in state.values() if v)
        threshold = self.activation_threshold_generations

        # --- stress_resilience ---
        if "stress_resilience" in state:
            if not state["stress_resilience"]:
                # Activate if suffering > 0.6 for threshold generations
                if (len(agent.suffering_history) >= threshold
                        and all(s > 0.6 for s in agent.suffering_history[-threshold:])):
                    if active_count < self.max_active_markers:
                        state["stress_resilience"] = True
                        active_count += 1
            else:
                # Deactivate if suffering drops below 0.3 for threshold generations
                if (len(agent.suffering_history) >= threshold
                        and all(s < 0.3 for s in agent.suffering_history[-threshold:])):
                    state["stress_resilience"] = False
                    active_count -= 1

        # --- creative_amplification ---
        if "creative_amplification" in state:
            from seldon.core.processing import ProcessingRegion
            deep_regions = {ProcessingRegion.DEEP, ProcessingRegion.SACRIFICIAL}
            if not state["creative_amplification"]:
                if (len(agent.region_history) >= threshold
                        and all(r in deep_regions for r in agent.region_history[-threshold:])):
                    if active_count < self.max_active_markers:
                        state["creative_amplification"] = True
                        active_count += 1
            else:
                from seldon.core.processing import ProcessingRegion as PR
                non_deep = {PR.UNDER_PROCESSING, PR.OPTIMAL}
                if (len(agent.region_history) >= threshold
                        and all(r in non_deep for r in agent.region_history[-threshold:])):
                    state["creative_amplification"] = False
                    active_count -= 1

        # --- social_withdrawal ---
        if "social_withdrawal" in state:
            try:
                trust_idx = trait_system.trait_index("trust")
                trust_val = float(agent.traits[trust_idx])
            except KeyError:
                trust_val = 0.5
            if not state["social_withdrawal"]:
                if trust_val < 0.3 and agent.suffering > 0.5:
                    if active_count < self.max_active_markers:
                        state["social_withdrawal"] = True
                        active_count += 1
            else:
                if trust_val > 0.6:
                    state["social_withdrawal"] = False
                    active_count -= 1

        # --- resource_conservation ---
        if "resource_conservation" in state:
            total_resources = sum(agent.resource_holdings.values()) if agent.resource_holdings else 0
            if not state["resource_conservation"]:
                if total_resources < 0.2 and len(agent.contribution_history) >= threshold:
                    if active_count < self.max_active_markers:
                        state["resource_conservation"] = True
                        active_count += 1
            else:
                if total_resources > 0.8:
                    state["resource_conservation"] = False
                    active_count -= 1

        # --- trauma_sensitivity ---
        # This marker is primarily inherited, not self-activated
        # It deactivates when agent has high resilience for extended period
        if "trauma_sensitivity" in state and state.get("trauma_sensitivity", False):
            try:
                resilience_idx = trait_system.trait_index("resilience")
                resilience_val = float(agent.traits[resilience_idx])
            except KeyError:
                resilience_val = 0.5
            if resilience_val > 0.8:
                state["trauma_sensitivity"] = False
                active_count -= 1

        agent.epigenetic_state = state
        return state

    # ------------------------------------------------------------------
    # Trait application
    # ------------------------------------------------------------------
    def apply_epigenetic_modifiers(
        self, agent: Agent, trait_system: TraitSystem,
    ) -> np.ndarray:
        """
        Apply active epigenetic marker modifiers to agent traits.

        Returns modified trait array (also modifies agent.traits in place).
        """
        if not self.enabled or not agent.epigenetic_state:
            return agent.traits

        for marker_name, active in agent.epigenetic_state.items():
            if not active:
                continue
            marker = self.markers.get(marker_name)
            if marker is None:
                continue
            try:
                idx = trait_system.trait_index(marker.target_trait)
            except KeyError:
                continue
            agent.traits[idx] = float(np.clip(
                agent.traits[idx] + marker.modifier, 0.0, 1.0,
            ))

        return agent.traits

    # ------------------------------------------------------------------
    # Transgenerational inheritance
    # ------------------------------------------------------------------
    def inherit_epigenetic_state(
        self, parent1: Agent, parent2: Agent,
        rng: np.random.Generator,
    ) -> dict[str, bool]:
        """
        Inherit epigenetic state from parents.

        Only markers that are active in at least one parent can be inherited,
        with ``transgenerational_rate`` probability.
        """
        if not self.enabled:
            return {}

        child_state = self.initialize_state()
        active_count = 0

        for marker_name in child_state:
            p1_active = parent1.epigenetic_state.get(marker_name, False)
            p2_active = parent2.epigenetic_state.get(marker_name, False)

            if not (p1_active or p2_active):
                continue

            # Higher probability if both parents have it
            rate = self.transgenerational_rate
            if p1_active and p2_active:
                rate = min(1.0, rate * 1.5)

            if rng.random() < rate and active_count < self.max_active_markers:
                child_state[marker_name] = True
                active_count += 1

        return child_state

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------
    def get_marker_prevalence(
        self, population: list[Agent],
    ) -> dict[str, dict[str, Any]]:
        """
        Compute prevalence of each epigenetic marker across population.

        Returns dict[marker_name, {"active_count": int, "prevalence": float}].
        """
        alive = [a for a in population if a.is_alive]
        total = len(alive)
        result: dict[str, dict[str, Any]] = {}
        for marker_name in self.markers:
            active_count = sum(
                1 for a in alive
                if a.epigenetic_state.get(marker_name, False)
            )
            result[marker_name] = {
                "active_count": active_count,
                "prevalence": active_count / total if total > 0 else 0.0,
                "description": self.markers[marker_name].description,
            }
        return result
