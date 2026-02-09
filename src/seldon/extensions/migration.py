"""
Migration Extension — population movement and settlement founding.

The core "agent orchestration" experiment: settlement viability depends
on group personality composition. The ``evaluate_settlement_viability``
function answers "can this group build something together?"

Requires: geography extension.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from seldon.core.decision import DecisionContext
from seldon.core.processing import ProcessingRegion
from seldon.extensions.base import SimulationExtension
from seldon.extensions.geography import GeographyExtension, Location

if TYPE_CHECKING:
    from seldon.core.agent import Agent
    from seldon.core.config import ExperimentConfig


class MigrationExtension(SimulationExtension):
    """Settlement founding with personality-composition viability analysis."""

    def __init__(self, geography: GeographyExtension) -> None:
        self.geography = geography
        self._events_this_gen: list[dict[str, Any]] = []
        self._new_settlements: int = 0
        self._viability_scores: list[float] = []

    @property
    def name(self) -> str:
        return "migration"

    @property
    def description(self) -> str:
        return "Population movement and settlement founding with viability analysis"

    def get_default_config(self) -> dict[str, Any]:
        return {
            "requires": ["geography"],
            "migration_min_age": 16,
            "overcrowding_threshold": 0.9,
            "min_founding_group_size": 5,
            "push_overcrowding_weight": 0.4,
            "pull_resource_weight": 0.3,
            "pull_space_weight": 0.3,
            # Settlement viability thresholds
            "min_conscientiousness": 0.4,
            "max_neuroticism": 0.7,
            "requires_leader": True,
            "leader_extraversion_threshold": 0.7,
            "min_region_diversity": 2,
            "optimal_r2_proportion": 0.4,
            "min_r3r4_proportion": 0.1,
        }

    def _get_config(self, config: ExperimentConfig) -> dict[str, Any]:
        defaults = self.get_default_config()
        overrides = config.extensions.get("migration", {})
        defaults.update(overrides)
        return defaults

    def evaluate_settlement_viability(
        self,
        founding_group: list[Agent],
        config: ExperimentConfig,
    ) -> tuple[float, list[str]]:
        """
        Evaluate whether a founding group can build a viable settlement.

        Returns (success_probability, risk_factors). Each check contributes
        to probability; each failure adds a risk factor string.
        """
        mig = self._get_config(config)
        ts = config.trait_system
        risk_factors: list[str] = []
        score = 0.0
        n_checks = 7

        if not founding_group:
            return 0.0, ["empty_group"]

        traits = np.array([a.traits for a in founding_group])

        # 1. Mean conscientiousness
        consc_idx = ts.trait_index("conscientiousness")
        mean_consc = float(traits[:, consc_idx].mean())
        if mean_consc >= mig["min_conscientiousness"]:
            score += 1.0
        else:
            risk_factors.append("low_conscientiousness")

        # 2. Mean neuroticism
        neur_idx = ts.trait_index("neuroticism")
        mean_neur = float(traits[:, neur_idx].mean())
        if mean_neur <= mig["max_neuroticism"]:
            score += 1.0
        else:
            risk_factors.append("high_neuroticism")

        # 3. Leadership presence (high extraversion)
        extra_idx = ts.trait_index("extraversion")
        leader_threshold = mig["leader_extraversion_threshold"]
        has_leader = any(
            a.traits[extra_idx] > leader_threshold for a in founding_group
        )
        if has_leader or not mig["requires_leader"]:
            score += 1.0
        else:
            risk_factors.append("no_leader")

        # 4. Extraversion variance (need mix of introverts + extraverts)
        extra_var = float(traits[:, extra_idx].var())
        if extra_var > 0.02:
            score += 1.0
        else:
            risk_factors.append("low_extraversion_variance")

        # 5. Region diversity
        regions = set(a.processing_region.value for a in founding_group)
        if len(regions) >= mig["min_region_diversity"]:
            score += 1.0
        else:
            risk_factors.append("low_region_diversity")

        # 6. R2 (optimal) proportion
        n = len(founding_group)
        r2_count = sum(
            1 for a in founding_group
            if a.processing_region == ProcessingRegion.OPTIMAL
        )
        if r2_count / n >= mig["optimal_r2_proportion"]:
            score += 1.0
        else:
            risk_factors.append("insufficient_optimal_processors")

        # 7. R3/R4 (deep/sacrificial) proportion
        r3r4_count = sum(
            1 for a in founding_group
            if a.processing_region in (
                ProcessingRegion.DEEP, ProcessingRegion.SACRIFICIAL,
            )
        )
        if r3r4_count / n >= mig["min_r3r4_proportion"]:
            score += 1.0
        else:
            risk_factors.append("insufficient_deep_thinkers")

        success_probability = score / n_checks
        return success_probability, risk_factors

    def on_generation_end(
        self, generation: int, population: list[Agent],
        config: ExperimentConfig,
    ) -> None:
        """Evaluate migration and settlement founding decisions."""
        mig = self._get_config(config)
        self._events_this_gen = []
        self._new_settlements = 0
        self._viability_scores = []

        if not self.geography.locations:
            return

        # Find overcrowded locations
        overcrowded: dict[str, list[Agent]] = {}
        for loc_id, loc in self.geography.locations.items():
            ratio = loc.current_population / max(loc.carrying_capacity, 1)
            if ratio > mig["overcrowding_threshold"]:
                agents_here = [
                    a for a in population
                    if a.location_id == loc_id
                    and a.age >= mig["migration_min_age"]
                ]
                if agents_here:
                    overcrowded[loc_id] = agents_here

        if not overcrowded:
            return

        rng = self.geography.rng or np.random.default_rng(config.random_seed)

        for loc_id, agents in overcrowded.items():
            # Select a random subset of agents willing to migrate
            n_willing = max(1, len(agents) // 4)
            if len(agents) <= n_willing:
                migrants = agents
            else:
                indices = rng.choice(len(agents), size=n_willing, replace=False)
                migrants = [agents[i] for i in indices]

            if len(migrants) < mig["min_founding_group_size"]:
                # Not enough migrants — they move to least-crowded location
                self._relocate_to_best(migrants, loc_id, config)
                continue

            # Evaluate settlement viability
            viability, risks = self.evaluate_settlement_viability(
                migrants, config,
            )
            self._viability_scores.append(viability)

            if viability >= 0.5:
                # Found a new settlement
                new_loc = self._create_settlement(config, rng)
                for agent in migrants:
                    agent.location_id = new_loc.id
                self._new_settlements += 1
                self._events_this_gen.append({
                    "type": "settlement_founded",
                    "location_id": new_loc.id,
                    "founders": len(migrants),
                    "viability": viability,
                    "risks": risks,
                })
            else:
                # Failed viability — relocate to existing location
                self._relocate_to_best(migrants, loc_id, config)
                self._events_this_gen.append({
                    "type": "migration_failed",
                    "from": loc_id,
                    "group_size": len(migrants),
                    "viability": viability,
                    "risks": risks,
                })

    def get_metrics(self, population: list[Agent]) -> dict[str, Any]:
        return {
            "migration_events": len(self._events_this_gen),
            "new_settlements": self._new_settlements,
            "viability_scores": self._viability_scores,
            "events": self._events_this_gen,
        }

    # --- Helpers ---

    def _relocate_to_best(
        self,
        migrants: list[Agent],
        from_loc: str,
        config: ExperimentConfig,
    ) -> None:
        """Move migrants to the least-crowded available location."""
        best_loc = None
        best_ratio = float("inf")
        for loc_id, loc in self.geography.locations.items():
            if loc_id == from_loc:
                continue
            ratio = loc.current_population / max(loc.carrying_capacity, 1)
            if ratio < best_ratio:
                best_ratio = ratio
                best_loc = loc_id

        if best_loc:
            for agent in migrants:
                agent.location_id = best_loc
            self._events_this_gen.append({
                "type": "migration_relocation",
                "from": from_loc,
                "to": best_loc,
                "count": len(migrants),
            })

    def _create_settlement(
        self,
        config: ExperimentConfig,
        rng: np.random.Generator,
    ) -> Location:
        """Create a new settlement at a random position."""
        geo = self.geography._get_config(config)
        map_size = geo["map_size"]
        base_cap = geo["base_carrying_capacity"]

        loc_id = f"loc_{len(self.geography.locations):03d}"
        new_loc = Location(
            id=loc_id,
            name=f"New Settlement {len(self.geography.locations) + 1}",
            coordinates=(
                int(rng.integers(0, map_size[0])),
                int(rng.integers(0, map_size[1])),
            ),
            carrying_capacity=int(base_cap * 0.6),  # New settlements start smaller
            resource_richness=float(rng.uniform(0.4, 1.2)),
        )
        self.geography.add_location(new_loc)
        return new_loc
