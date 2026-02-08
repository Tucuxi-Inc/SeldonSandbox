"""
Metrics Collector — enhanced per-generation statistics.

Extends GenerationSnapshot with richer analytics including trait entropy,
region transitions, suffering by region, lore metrics, and outsider tracking.
Provides time series extraction and export for visualization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from seldon.core.agent import Agent
from seldon.core.config import ExperimentConfig
from seldon.core.engine import GenerationSnapshot
from seldon.core.processing import ProcessingRegion


@dataclass
class GenerationMetrics:
    """Extended metrics for a single generation."""

    # Base snapshot data
    generation: int
    population_size: int
    births: int
    deaths: int
    breakthroughs: int
    pairs_formed: int

    # Trait statistics
    trait_means: np.ndarray
    trait_stds: np.ndarray
    trait_entropy: float  # Shannon entropy of trait distributions

    # Region analysis
    region_counts: dict[str, int]
    region_fractions: dict[str, float]
    region_transitions: dict[str, int]  # "R2->R3": count

    # Contribution
    total_contribution: float
    mean_contribution: float
    max_contribution: float

    # Suffering
    mean_suffering: float
    suffering_by_region: dict[str, float]

    # Demographics
    mean_age: float
    age_distribution: dict[str, int]  # age buckets

    # Birth order
    birth_order_counts: dict[int, int]

    # Lore metrics
    total_memories: int
    societal_memories: int
    myths_count: int

    # Outsider tracking
    outsider_count: int
    outsider_descendant_count: int

    # Events
    dissolutions: int
    infidelity_events: int
    outsiders_injected: int

    # Council voice distribution (if enabled)
    dominant_voice_counts: dict[str, int] = field(default_factory=dict)


class MetricsCollector:
    """
    Collects and aggregates metrics across generations.

    Works alongside the simulation engine to provide richer analytics
    than the base GenerationSnapshot.
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.ts = config.trait_system
        self.metrics_history: list[GenerationMetrics] = []
        self._previous_regions: dict[str, str] = {}  # agent_id -> region name

    def collect(
        self,
        population: list[Agent],
        snapshot: GenerationSnapshot,
    ) -> GenerationMetrics:
        """Collect enhanced metrics for a generation."""
        # Trait entropy
        trait_entropy = 0.0
        if population:
            trait_matrix = np.array([a.traits for a in population])
            trait_entropy = self._compute_trait_entropy(trait_matrix)

        # Region fractions
        total = max(snapshot.population_size, 1)
        region_fractions = {
            name: count / total
            for name, count in snapshot.region_counts.items()
        }

        # Region transitions
        transitions = self._compute_region_transitions(population)

        # Max contribution
        max_contrib = 0.0
        if population:
            contributions = [
                a.contribution_history[-1] if a.contribution_history else 0.0
                for a in population
            ]
            max_contrib = max(contributions) if contributions else 0.0

        # Suffering by region
        suffering_by_region = self._compute_suffering_by_region(population)

        # Age distribution (buckets: 0-15, 16-30, 31-50, 51+)
        age_dist = self._compute_age_distribution(population)

        # Lore metrics
        total_memories = sum(
            len(a.personal_memories) + len(a.inherited_lore)
            for a in population
        )
        myths = sum(
            1 for a in population
            for m in a.personal_memories + a.inherited_lore
            if m.get("memory_type") == "myth"
        )

        # Outsider tracking
        outsiders = sum(1 for a in population if a.is_outsider)
        descendants = sum(
            1 for a in population
            if not a.is_outsider and a.is_descendant_of_outsider
        )

        # Council voice distribution
        voice_counts: dict[str, int] = {}
        for a in population:
            if a.dominant_voice:
                voice_counts[a.dominant_voice] = voice_counts.get(a.dominant_voice, 0) + 1

        events = snapshot.events

        metrics = GenerationMetrics(
            generation=snapshot.generation,
            population_size=snapshot.population_size,
            births=snapshot.births,
            deaths=snapshot.deaths,
            breakthroughs=snapshot.breakthroughs,
            pairs_formed=snapshot.pairs_formed,
            trait_means=snapshot.trait_means,
            trait_stds=snapshot.trait_stds,
            trait_entropy=trait_entropy,
            region_counts=snapshot.region_counts,
            region_fractions=region_fractions,
            region_transitions=transitions,
            total_contribution=snapshot.total_contribution,
            mean_contribution=snapshot.mean_contribution,
            max_contribution=max_contrib,
            mean_suffering=snapshot.mean_suffering,
            suffering_by_region=suffering_by_region,
            mean_age=snapshot.mean_age,
            age_distribution=age_dist,
            birth_order_counts=snapshot.birth_order_counts,
            total_memories=total_memories,
            societal_memories=events.get("memories_created", 0),
            myths_count=myths,
            outsider_count=outsiders,
            outsider_descendant_count=descendants,
            dissolutions=events.get("dissolutions", 0),
            infidelity_events=events.get("infidelity_events", 0),
            outsiders_injected=events.get("outsiders_injected", 0),
            dominant_voice_counts=voice_counts,
        )

        # Update tracking state
        self._previous_regions = {
            a.id: a.processing_region.value for a in population
        }

        self.metrics_history.append(metrics)
        return metrics

    def get_time_series(self, field_name: str) -> list[Any]:
        """Extract a time series for a specific metric field."""
        return [getattr(m, field_name) for m in self.metrics_history]

    def export_for_visualization(self) -> list[dict[str, Any]]:
        """Export all metrics as a list of JSON-serializable dicts."""
        result = []
        for m in self.metrics_history:
            d: dict[str, Any] = {
                "generation": m.generation,
                "population_size": m.population_size,
                "births": m.births,
                "deaths": m.deaths,
                "breakthroughs": m.breakthroughs,
                "pairs_formed": m.pairs_formed,
                "trait_means": m.trait_means.tolist(),
                "trait_stds": m.trait_stds.tolist(),
                "trait_entropy": m.trait_entropy,
                "region_counts": m.region_counts,
                "region_fractions": m.region_fractions,
                "region_transitions": m.region_transitions,
                "total_contribution": m.total_contribution,
                "mean_contribution": m.mean_contribution,
                "max_contribution": m.max_contribution,
                "mean_suffering": m.mean_suffering,
                "suffering_by_region": m.suffering_by_region,
                "mean_age": m.mean_age,
                "age_distribution": m.age_distribution,
                "birth_order_counts": {str(k): v for k, v in m.birth_order_counts.items()},
                "total_memories": m.total_memories,
                "societal_memories": m.societal_memories,
                "myths_count": m.myths_count,
                "outsider_count": m.outsider_count,
                "outsider_descendant_count": m.outsider_descendant_count,
                "dissolutions": m.dissolutions,
                "infidelity_events": m.infidelity_events,
                "outsiders_injected": m.outsiders_injected,
                "dominant_voice_counts": m.dominant_voice_counts,
            }
            result.append(d)
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _compute_trait_entropy(self, trait_matrix: np.ndarray) -> float:
        """Compute Shannon entropy of trait distributions (binned)."""
        if trait_matrix.size == 0:
            return 0.0

        total_entropy = 0.0
        n_traits = trait_matrix.shape[1]
        for i in range(n_traits):
            values = trait_matrix[:, i]
            # Bin into 10 buckets
            hist, _ = np.histogram(values, bins=10, range=(0, 1))
            probs = hist / hist.sum()
            probs = probs[probs > 0]
            total_entropy += -np.sum(probs * np.log2(probs))

        return float(total_entropy / n_traits)

    def _compute_region_transitions(
        self, population: list[Agent],
    ) -> dict[str, int]:
        """Count how many agents transitioned between regions."""
        transitions: dict[str, int] = {}
        for a in population:
            prev = self._previous_regions.get(a.id)
            current = a.processing_region.value
            if prev is not None and prev != current:
                key = f"{prev}->{current}"
                transitions[key] = transitions.get(key, 0) + 1
        return transitions

    def _compute_suffering_by_region(
        self, population: list[Agent],
    ) -> dict[str, float]:
        """Compute mean suffering per region."""
        region_suffering: dict[str, list[float]] = {
            r.value: [] for r in ProcessingRegion
        }
        for a in population:
            region_suffering[a.processing_region.value].append(a.suffering)

        return {
            region: (float(np.mean(vals)) if vals else 0.0)
            for region, vals in region_suffering.items()
        }

    def _compute_age_distribution(
        self, population: list[Agent],
    ) -> dict[str, int]:
        """Compute age distribution in buckets."""
        buckets = {"0-15": 0, "16-30": 0, "31-50": 0, "51+": 0}
        for a in population:
            if a.age <= 15:
                buckets["0-15"] += 1
            elif a.age <= 30:
                buckets["16-30"] += 1
            elif a.age <= 50:
                buckets["31-50"] += 1
            else:
                buckets["51+"] += 1
        return buckets
