"""
Culture / Memes Extension — memetic propagation and effects.

Tracks cultural memes that spread through social contact, mutate, and
go extinct. The "tortured genius" meme tests whether cultural narratives
can maintain equilibrium in the sacrificial processing region.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from seldon.extensions.base import SimulationExtension

if TYPE_CHECKING:
    from seldon.core.agent import Agent
    from seldon.core.config import ExperimentConfig


@dataclass
class Meme:
    """A cultural meme with effects on agent behavior."""

    id: str
    name: str
    effects: dict[str, float] = field(default_factory=dict)
    prevalence: float = 0.0


# Default memes with their effects on decision utilities
_DEFAULT_MEMES = {
    "tortured_genius": {
        "name": "Tortured Genius",
        "effects": {"depth_drive_boost": 0.15, "contribution_boost": 0.1},
    },
    "efficiency_worship": {
        "name": "Efficiency Worship",
        "effects": {"contribution_boost": 0.2, "depth_drive_penalty": -0.1},
    },
    "family_first": {
        "name": "Family First",
        "effects": {"reproduction_boost": 0.2, "migration_penalty": -0.15},
    },
}


class CultureExtension(SimulationExtension):
    """Meme propagation and cultural effects on agent decisions."""

    @property
    def name(self) -> str:
        return "culture"

    @property
    def description(self) -> str:
        return "Cultural meme propagation with effects on behavior"

    def get_default_config(self) -> dict[str, Any]:
        return {
            "initial_memes": list(_DEFAULT_MEMES.keys()),
            "spread_rate": 0.12,
            "mutation_rate": 0.01,
            "extinction_threshold": 0.05,
            "max_memes_per_agent": 5,
            "initial_adoption_rate": 0.3,
        }

    def __init__(self) -> None:
        self.memes: dict[str, Meme] = {}
        self.rng: np.random.Generator | None = None

    def _get_config(self, config: ExperimentConfig) -> dict[str, Any]:
        defaults = self.get_default_config()
        overrides = config.extensions.get("culture", {})
        defaults.update(overrides)
        return defaults

    def on_simulation_start(
        self, population: list[Agent], config: ExperimentConfig,
    ) -> None:
        """Create initial memes and assign to random subset of agents."""
        cult = self._get_config(config)
        self.rng = np.random.default_rng(config.random_seed)

        # Create memes
        self.memes = {}
        for meme_id in cult["initial_memes"]:
            if meme_id in _DEFAULT_MEMES:
                defn = _DEFAULT_MEMES[meme_id]
                self.memes[meme_id] = Meme(
                    id=meme_id,
                    name=defn["name"],
                    effects=dict(defn["effects"]),
                )

        # Assign memes to random agents
        adoption_rate = cult["initial_adoption_rate"]
        for agent in population:
            for meme_id in self.memes:
                if self.rng.random() < adoption_rate:
                    if meme_id not in agent.cultural_memes:
                        agent.cultural_memes.append(meme_id)

        self._update_prevalence(population)

    def on_generation_end(
        self, generation: int, population: list[Agent],
        config: ExperimentConfig,
    ) -> None:
        """Spread, mutate, and extinct memes."""
        if not self.memes or not population:
            return

        cult = self._get_config(config)
        rng = self.rng or np.random.default_rng(config.random_seed)

        # Spread: agents adopt memes from population based on prevalence
        spread_rate = cult["spread_rate"]
        max_memes = cult["max_memes_per_agent"]
        for agent in population:
            for meme_id, meme in self.memes.items():
                if meme_id in agent.cultural_memes:
                    continue
                if len(agent.cultural_memes) >= max_memes:
                    break
                # Adoption probability scales with prevalence and spread rate
                prob = meme.prevalence * spread_rate
                if rng.random() < prob:
                    agent.cultural_memes.append(meme_id)

        # Mutation: small random changes to meme effects
        mutation_rate = cult["mutation_rate"]
        for meme in self.memes.values():
            if rng.random() < mutation_rate:
                for key in meme.effects:
                    meme.effects[key] += float(rng.normal(0, 0.02))
                    meme.effects[key] = float(
                        np.clip(meme.effects[key], -1.0, 1.0)
                    )

        # Update prevalence
        self._update_prevalence(population)

        # Extinction: remove memes below threshold
        extinct_threshold = cult["extinction_threshold"]
        extinct_ids = [
            mid for mid, m in self.memes.items()
            if m.prevalence < extinct_threshold
        ]
        for mid in extinct_ids:
            del self.memes[mid]
            for agent in population:
                if mid in agent.cultural_memes:
                    agent.cultural_memes.remove(mid)

    def modify_decision(
        self, agent: Agent, context: str, utilities: dict[str, float],
        config: ExperimentConfig,
    ) -> dict[str, float]:
        """Apply meme effects to decision utilities."""
        if not agent.cultural_memes:
            return utilities

        for meme_id in agent.cultural_memes:
            meme = self.memes.get(meme_id)
            if meme is None:
                continue
            for key, boost in meme.effects.items():
                # Map meme effects to action utilities
                for action in utilities:
                    if action in key or key.replace("_boost", "").replace("_penalty", "") in action:
                        utilities[action] = utilities.get(action, 0.0) + boost

        return utilities

    def get_metrics(self, population: list[Agent]) -> dict[str, Any]:
        self._update_prevalence(population)
        prevalence = {mid: m.prevalence for mid, m in self.memes.items()}
        dominant = max(prevalence, key=prevalence.get) if prevalence else None
        return {
            "meme_count": len(self.memes),
            "meme_prevalence": prevalence,
            "dominant_meme": dominant,
            "cultural_diversity": self._shannon_entropy(prevalence),
        }

    # --- Helpers ---

    def _update_prevalence(self, population: list[Agent]) -> None:
        """Recalculate meme prevalence as fraction of population holding it."""
        n = max(len(population), 1)
        for meme_id in self.memes:
            count = sum(
                1 for a in population if meme_id in a.cultural_memes
            )
            self.memes[meme_id].prevalence = count / n

    @staticmethod
    def _shannon_entropy(prevalence: dict[str, float]) -> float:
        """Compute Shannon entropy of meme distribution."""
        values = [v for v in prevalence.values() if v > 0]
        if not values:
            return 0.0
        total = sum(values)
        probs = [v / total for v in values]
        return float(-sum(p * np.log2(p) for p in probs if p > 0))
