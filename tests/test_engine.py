"""Integration tests for SimulationEngine."""

import numpy as np
import pytest

from seldon.core.config import ExperimentConfig
from seldon.core.engine import SimulationEngine, GenerationSnapshot


class TestBasicSimulation:
    def test_single_generation_runs(self):
        config = ExperimentConfig(initial_population=20, random_seed=42)
        engine = SimulationEngine(config)
        history = engine.run(generations=1)
        assert len(history) == 1
        assert history[0].population_size > 0

    def test_multi_generation_runs(self):
        config = ExperimentConfig(
            initial_population=50, generations_to_run=10, random_seed=42,
        )
        engine = SimulationEngine(config)
        history = engine.run()
        assert len(history) == 10

    def test_population_changes_over_time(self):
        config = ExperimentConfig(
            initial_population=50, generations_to_run=5, random_seed=42,
        )
        engine = SimulationEngine(config)
        history = engine.run()
        sizes = [s.population_size for s in history]
        # Population shouldn't stay exactly the same every generation
        assert len(set(sizes)) > 1

    def test_snapshot_has_all_fields(self):
        config = ExperimentConfig(initial_population=20, random_seed=42)
        engine = SimulationEngine(config)
        history = engine.run(generations=1)
        snap = history[0]

        assert isinstance(snap, GenerationSnapshot)
        assert snap.generation == 0
        assert snap.trait_means.shape == (config.trait_system.count,)
        assert snap.trait_stds.shape == (config.trait_system.count,)
        assert len(snap.region_counts) == 5


class TestTraitDrift:
    def test_traits_change_over_generations(self):
        config = ExperimentConfig(
            initial_population=50, generations_to_run=20,
            trait_drift_rate=0.1, random_seed=42,
        )
        engine = SimulationEngine(config)
        history = engine.run()

        gen0_means = history[0].trait_means
        gen19_means = history[19].trait_means
        # With high drift rate, means should change noticeably
        assert not np.allclose(gen0_means, gen19_means, atol=0.02)


class TestReproduction:
    def test_births_occur(self):
        config = ExperimentConfig(
            initial_population=50, generations_to_run=5, random_seed=42,
        )
        engine = SimulationEngine(config)
        history = engine.run()
        total_births = sum(s.births for s in history)
        assert total_births > 0

    def test_deaths_occur(self):
        config = ExperimentConfig(
            initial_population=50, generations_to_run=10, random_seed=42,
        )
        engine = SimulationEngine(config)
        history = engine.run()
        total_deaths = sum(s.deaths for s in history)
        assert total_deaths > 0


class TestRegionDistribution:
    def test_all_regions_have_counts(self):
        config = ExperimentConfig(initial_population=200, random_seed=42)
        engine = SimulationEngine(config)
        history = engine.run(generations=1)
        # With 200 agents, we should see most regions populated
        assert sum(history[0].region_counts.values()) == history[0].population_size


class TestFullPreset:
    def test_full_trait_preset_runs(self):
        config = ExperimentConfig(
            trait_preset="full", initial_population=30,
            generations_to_run=3, random_seed=42,
        )
        engine = SimulationEngine(config)
        history = engine.run()
        assert len(history) == 3
        assert history[0].trait_means.shape == (50,)


class TestDeterminism:
    def test_same_seed_produces_same_results(self):
        config = ExperimentConfig(
            initial_population=50, generations_to_run=5, random_seed=99,
        )
        engine1 = SimulationEngine(config)
        h1 = engine1.run()

        engine2 = SimulationEngine(config)
        h2 = engine2.run()

        for s1, s2 in zip(h1, h2):
            assert s1.population_size == s2.population_size
            assert s1.births == s2.births
            assert s1.deaths == s2.deaths
            np.testing.assert_array_equal(s1.trait_means, s2.trait_means)
