"""Tests for MetricsCollector."""

import numpy as np
import pytest

from seldon.core.agent import Agent
from seldon.core.config import ExperimentConfig
from seldon.core.engine import SimulationEngine, GenerationSnapshot
from seldon.core.processing import ProcessingRegion
from seldon.metrics.collector import MetricsCollector, GenerationMetrics


def _make_agent(agent_id: str, traits: np.ndarray, age: int = 25, **kwargs) -> Agent:
    defaults = dict(
        id=agent_id, name=f"A-{agent_id}", age=age, generation=0,
        birth_order=1, traits=traits, traits_at_birth=traits.copy(),
    )
    defaults.update(kwargs)
    return Agent(**defaults)


class TestCollectMetrics:
    def test_collect_from_simulation(self):
        config = ExperimentConfig(
            initial_population=30, random_seed=42,
        )
        engine = SimulationEngine(config)
        history = engine.run(generations=3)

        collector = MetricsCollector(config)
        for snap in history:
            metrics = collector.collect(engine.population, snap)
            assert isinstance(metrics, GenerationMetrics)

        assert len(collector.metrics_history) == 3

    def test_metrics_have_expected_fields(self):
        config = ExperimentConfig(initial_population=30, random_seed=42)
        engine = SimulationEngine(config)
        history = engine.run(generations=1)

        collector = MetricsCollector(config)
        metrics = collector.collect(engine.population, history[0])

        assert metrics.generation == 0
        assert metrics.population_size > 0
        assert isinstance(metrics.trait_entropy, float)
        assert metrics.trait_entropy >= 0
        assert len(metrics.region_fractions) == 5
        assert sum(metrics.region_fractions.values()) == pytest.approx(1.0, abs=0.01)

    def test_trait_entropy_positive(self):
        config = ExperimentConfig(initial_population=50, random_seed=42)
        engine = SimulationEngine(config)
        history = engine.run(generations=1)

        collector = MetricsCollector(config)
        metrics = collector.collect(engine.population, history[0])
        assert metrics.trait_entropy > 0

    def test_suffering_by_region(self):
        config = ExperimentConfig(initial_population=50, random_seed=42)
        engine = SimulationEngine(config)
        history = engine.run(generations=3)

        collector = MetricsCollector(config)
        metrics = collector.collect(engine.population, history[-1])

        # Should have all 5 regions
        assert len(metrics.suffering_by_region) == 5
        for region, value in metrics.suffering_by_region.items():
            assert value >= 0

    def test_age_distribution(self):
        config = ExperimentConfig(initial_population=50, random_seed=42)
        engine = SimulationEngine(config)
        history = engine.run(generations=5)

        collector = MetricsCollector(config)
        metrics = collector.collect(engine.population, history[-1])

        assert "0-15" in metrics.age_distribution
        assert "16-30" in metrics.age_distribution
        assert "31-50" in metrics.age_distribution
        assert "51+" in metrics.age_distribution
        total = sum(metrics.age_distribution.values())
        assert total == metrics.population_size


class TestRegionTransitions:
    def test_transitions_tracked_across_generations(self):
        config = ExperimentConfig(
            initial_population=50, random_seed=42,
            trait_drift_rate=0.1,  # High drift to cause transitions
        )
        engine = SimulationEngine(config)
        history = engine.run(generations=5)

        collector = MetricsCollector(config)
        for snap in history:
            collector.collect(engine.population, snap)

        # After first generation, transitions should be empty (no previous state)
        assert collector.metrics_history[0].region_transitions == {}
        # Later generations may have transitions (with high drift)


class TestTimeSeries:
    def test_get_time_series(self):
        config = ExperimentConfig(initial_population=30, random_seed=42)
        engine = SimulationEngine(config)
        history = engine.run(generations=5)

        collector = MetricsCollector(config)
        for snap in history:
            collector.collect(engine.population, snap)

        pop_series = collector.get_time_series("population_size")
        assert len(pop_series) == 5

        entropy_series = collector.get_time_series("trait_entropy")
        assert len(entropy_series) == 5
        assert all(isinstance(v, float) for v in entropy_series)


class TestExportForVisualization:
    def test_export_returns_dicts(self):
        config = ExperimentConfig(initial_population=20, random_seed=42)
        engine = SimulationEngine(config)
        history = engine.run(generations=3)

        collector = MetricsCollector(config)
        for snap in history:
            collector.collect(engine.population, snap)

        exported = collector.export_for_visualization()
        assert len(exported) == 3
        for d in exported:
            assert isinstance(d, dict)
            assert "generation" in d
            assert "population_size" in d
            assert "trait_entropy" in d
            assert isinstance(d["trait_means"], list)
            assert isinstance(d["region_fractions"], dict)

    def test_export_json_serializable(self):
        import json
        config = ExperimentConfig(initial_population=20, random_seed=42)
        engine = SimulationEngine(config)
        history = engine.run(generations=2)

        collector = MetricsCollector(config)
        for snap in history:
            collector.collect(engine.population, snap)

        exported = collector.export_for_visualization()
        # Should not raise
        json_str = json.dumps(exported)
        assert len(json_str) > 0


class TestEmptyPopulation:
    def test_collect_empty_population(self):
        config = ExperimentConfig(initial_population=0, random_seed=42)
        # Create a minimal snapshot for empty population
        snap = GenerationSnapshot(
            generation=0, population_size=0, births=0, deaths=0,
            breakthroughs=0, pairs_formed=0,
            trait_means=np.zeros(config.trait_system.count),
            trait_stds=np.zeros(config.trait_system.count),
            region_counts={r.value: 0 for r in ProcessingRegion},
            total_contribution=0, mean_contribution=0,
            mean_suffering=0, mean_age=0,
            birth_order_counts={},
            events={"dissolutions": 0, "infidelity_events": 0, "outsiders_injected": 0},
        )

        collector = MetricsCollector(config)
        metrics = collector.collect([], snap)
        assert metrics.population_size == 0
        assert metrics.trait_entropy == 0.0


class TestCouncilVoiceDistribution:
    def test_voice_counts_when_council_enabled(self):
        config = ExperimentConfig(
            initial_population=30,
            random_seed=42,
            cognitive_council_enabled=True,
        )
        engine = SimulationEngine(config)
        history = engine.run(generations=2)

        collector = MetricsCollector(config)
        # Collect on generation 2 so all agents (including gen-0 babies)
        # have been through a council voice update
        metrics = collector.collect(engine.population, history[1])

        # With council enabled, agents should have dominant voices
        assert len(metrics.dominant_voice_counts) > 0
        # Most agents should have voices (newborns from this gen won't yet)
        total_with_voices = sum(metrics.dominant_voice_counts.values())
        assert total_with_voices > 0
