"""
Conflict Extension — personality-based triggers and resolution.

Detects conflicts from dominance clashes, trust betrayals, and resource
scarcity. Resolves them using the decision model with trait-influenced
methods: submission, compromise, separation, escalation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from seldon.extensions.base import SimulationExtension

if TYPE_CHECKING:
    from seldon.core.agent import Agent
    from seldon.core.config import ExperimentConfig


# Resolution method definitions with trait influences
_RESOLUTION_METHODS = {
    "submission": {"dominance": -0.3, "agreeableness": 0.2},
    "compromise": {"agreeableness": 0.3, "empathy": 0.2},
    "separation": {"extraversion": -0.2, "adaptability": 0.3},
    "escalation": {"dominance": 0.3, "risk_taking": 0.2},
}


class ConflictExtension(SimulationExtension):
    """Personality-based conflict detection and resolution."""

    @property
    def name(self) -> str:
        return "conflict"

    @property
    def description(self) -> str:
        return "Personality-based conflict triggers and trait-influenced resolution"

    def get_default_config(self) -> dict[str, Any]:
        return {
            "dominance_clash_threshold": 0.8,
            "trust_betrayal_threshold": 0.3,
            "resource_scarcity_threshold": 0.7,
            "conflict_mortality_increase": 0.05,
            "max_conflicts_per_generation": 10,
        }

    def __init__(self) -> None:
        self._conflicts_this_gen: list[dict[str, Any]] = []
        self._resolution_counts: dict[str, int] = {
            method: 0 for method in _RESOLUTION_METHODS
        }
        self._casualties: int = 0
        self._agents_in_conflict: set[str] = set()

    def _get_config(self, config: ExperimentConfig) -> dict[str, Any]:
        defaults = self.get_default_config()
        overrides = config.extensions.get("conflict", {})
        defaults.update(overrides)
        return defaults

    def on_generation_start(
        self, generation: int, population: list[Agent],
        config: ExperimentConfig,
    ) -> None:
        """Reset per-generation conflict state."""
        self._conflicts_this_gen = []
        self._resolution_counts = {m: 0 for m in _RESOLUTION_METHODS}
        self._casualties = 0
        self._agents_in_conflict = set()

    def on_generation_end(
        self, generation: int, population: list[Agent],
        config: ExperimentConfig,
    ) -> None:
        """Detect and resolve conflicts."""
        conf = self._get_config(config)
        ts = config.trait_system
        rng = np.random.default_rng(config.random_seed + generation)

        max_conflicts = conf["max_conflicts_per_generation"]

        # Detect dominance clashes
        dom_idx = ts.trait_index("dominance") if "dominance" in [
            ts.trait_name(i) for i in range(ts.count)
        ] else None
        trust_idx = ts.trait_index("trust") if "trust" in [
            ts.trait_name(i) for i in range(ts.count)
        ] else None

        conflict_count = 0
        # Sample pairs to check for conflicts (avoid O(N^2))
        n = len(population)
        if n < 2:
            return

        max_checks = min(n * 2, 100)
        for _ in range(max_checks):
            if conflict_count >= max_conflicts:
                break

            i, j = rng.choice(n, size=2, replace=False)
            a1, a2 = population[i], population[j]

            if a1.id in self._agents_in_conflict or a2.id in self._agents_in_conflict:
                continue

            triggered = False
            trigger_type = ""

            # Dominance clash
            if dom_idx is not None:
                if (a1.traits[dom_idx] > conf["dominance_clash_threshold"]
                        and a2.traits[dom_idx] > conf["dominance_clash_threshold"]):
                    triggered = True
                    trigger_type = "dominance_clash"

            # Trust betrayal
            if not triggered and trust_idx is not None:
                bond = a1.social_bonds.get(a2.id, 0.5)
                if (a1.traits[trust_idx] < conf["trust_betrayal_threshold"]
                        or a2.traits[trust_idx] < conf["trust_betrayal_threshold"]):
                    if bond > 0.3:  # Need some relationship to betray
                        triggered = True
                        trigger_type = "trust_betrayal"

            if triggered:
                resolution = self._resolve_conflict(a1, a2, ts, rng)
                self._conflicts_this_gen.append({
                    "agents": [a1.id, a2.id],
                    "trigger": trigger_type,
                    "resolution": resolution,
                })
                self._resolution_counts[resolution] += 1
                self._agents_in_conflict.add(a1.id)
                self._agents_in_conflict.add(a2.id)

                # Apply consequences
                self._apply_consequences(a1, a2, resolution, ts)
                conflict_count += 1

    def modify_mortality(
        self, agent: Agent, base_rate: float,
        config: ExperimentConfig,
    ) -> float:
        """Increase mortality for agents involved in conflicts."""
        if agent.id in self._agents_in_conflict:
            conf = self._get_config(config)
            return base_rate + conf["conflict_mortality_increase"]
        return base_rate

    def get_metrics(self, population: list[Agent]) -> dict[str, Any]:
        return {
            "conflict_count": len(self._conflicts_this_gen),
            "resolution_distribution": dict(self._resolution_counts),
            "casualties": self._casualties,
            "agents_in_conflict": len(self._agents_in_conflict),
        }

    # --- Helpers ---

    def _resolve_conflict(
        self, a1: Agent, a2: Agent, ts: Any, rng: np.random.Generator,
    ) -> str:
        """Choose resolution method based on agents' traits."""
        methods = list(_RESOLUTION_METHODS.keys())
        scores = []

        for method in methods:
            influences = _RESOLUTION_METHODS[method]
            score = 0.0
            for trait_name, weight in influences.items():
                try:
                    idx = ts.trait_index(trait_name)
                    # Average both agents' traits
                    avg = (a1.traits[idx] + a2.traits[idx]) / 2.0
                    score += avg * weight
                except (KeyError, IndexError):
                    pass
            scores.append(score)

        # Softmax selection
        scores_arr = np.array(scores)
        scores_arr = scores_arr - scores_arr.max()
        exp_scores = np.exp(scores_arr)
        probs = exp_scores / exp_scores.sum()

        chosen_idx = rng.choice(len(methods), p=probs)
        return methods[chosen_idx]

    def _apply_consequences(
        self, a1: Agent, a2: Agent, resolution: str, ts: Any,
    ) -> None:
        """Apply relationship and trait consequences of conflict resolution."""
        if resolution == "submission":
            # Submitter loses relationship strength
            a1.social_bonds[a2.id] = max(
                0.0, a1.social_bonds.get(a2.id, 0.5) - 0.2,
            )
        elif resolution == "compromise":
            # Both gain slight bond
            for a, b in [(a1, a2), (a2, a1)]:
                a.social_bonds[b.id] = min(
                    1.0, a.social_bonds.get(b.id, 0.3) + 0.1,
                )
        elif resolution == "separation":
            # Bond broken
            a1.social_bonds.pop(a2.id, None)
            a2.social_bonds.pop(a1.id, None)
        elif resolution == "escalation":
            # Both lose bond, suffer
            a1.social_bonds[a2.id] = 0.0
            a2.social_bonds[a1.id] = 0.0
            a1.suffering = min(1.0, a1.suffering + 0.1)
            a2.suffering = min(1.0, a2.suffering + 0.1)
