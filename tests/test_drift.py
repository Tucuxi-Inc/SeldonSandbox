"""Tests for TraitDriftEngine."""

import numpy as np
import pytest

from seldon.core.agent import Agent
from seldon.core.config import ExperimentConfig
from seldon.core.drift import TraitDriftEngine
from seldon.core.processing import ProcessingRegion


def _make_agent(traits, age=25, region=ProcessingRegion.OPTIMAL):
    return Agent(
        id="test", name="Test", age=age, generation=0, birth_order=1,
        traits=traits, traits_at_birth=traits.copy(),
        processing_region=region,
    )


class TestDriftBasics:
    def test_drift_changes_traits(self):
        config = ExperimentConfig(trait_drift_rate=0.1)
        de = TraitDriftEngine(config)
        rng = np.random.default_rng(42)

        traits = np.full(config.trait_system.count, 0.5)
        agent = _make_agent(traits)
        new_traits = de.drift_traits(agent, rng)
        assert not np.array_equal(new_traits, traits)

    def test_drift_stays_bounded(self):
        config = ExperimentConfig(trait_drift_rate=0.5)
        de = TraitDriftEngine(config)
        rng = np.random.default_rng(42)

        for _ in range(100):
            traits = rng.uniform(0, 1, size=config.trait_system.count)
            agent = _make_agent(traits)
            new_traits = de.drift_traits(agent, rng)
            assert np.all(new_traits >= 0.0)
            assert np.all(new_traits <= 1.0)

    def test_old_agents_drift_less(self):
        config = ExperimentConfig(trait_drift_rate=0.1)
        de = TraitDriftEngine(config)

        traits = np.full(config.trait_system.count, 0.5)

        # Run many drift iterations to get average magnitude
        rng_young = np.random.default_rng(42)
        rng_old = np.random.default_rng(42)

        young_drifts = []
        old_drifts = []
        for _ in range(200):
            young = _make_agent(traits.copy(), age=5)
            old = _make_agent(traits.copy(), age=80)
            young_new = de.drift_traits(young, rng_young)
            old_new = de.drift_traits(old, rng_old)
            young_drifts.append(np.abs(young_new - traits).mean())
            old_drifts.append(np.abs(old_new - traits).mean())

        assert np.mean(young_drifts) > np.mean(old_drifts)


class TestRegionEffects:
    def test_sacrificial_deepens_depth_drive(self):
        config = ExperimentConfig()
        de = TraitDriftEngine(config)
        ts = config.trait_system

        traits = np.full(ts.count, 0.5)
        agent = _make_agent(traits, region=ProcessingRegion.SACRIFICIAL)
        new_traits = de.apply_region_effects(agent)

        # Sacrificial region should push depth_drive up
        assert new_traits[ts.DEPTH_DRIVE] > traits[ts.DEPTH_DRIVE]

    def test_optimal_no_region_effect(self):
        config = ExperimentConfig()
        de = TraitDriftEngine(config)

        traits = np.full(config.trait_system.count, 0.5)
        agent = _make_agent(traits, region=ProcessingRegion.OPTIMAL)
        new_traits = de.apply_region_effects(agent)

        np.testing.assert_array_equal(new_traits, traits)


class TestSufferingAndBurnout:
    def test_suffering_increases_in_sacrificial(self):
        config = ExperimentConfig()
        de = TraitDriftEngine(config)

        traits = np.full(config.trait_system.count, 0.5)
        agent = _make_agent(traits, region=ProcessingRegion.SACRIFICIAL)
        agent.suffering = 0.0
        de.update_suffering(agent)
        assert agent.suffering > 0.0

    def test_suffering_decreases_in_optimal(self):
        config = ExperimentConfig()
        de = TraitDriftEngine(config)

        traits = np.full(config.trait_system.count, 0.5)
        agent = _make_agent(traits, region=ProcessingRegion.OPTIMAL)
        agent.suffering = 0.5
        de.update_suffering(agent)
        assert agent.suffering < 0.5

    def test_suffering_stays_bounded(self):
        config = ExperimentConfig()
        de = TraitDriftEngine(config)
        traits = np.full(config.trait_system.count, 0.5)

        agent = _make_agent(traits, region=ProcessingRegion.PATHOLOGICAL)
        agent.suffering = 0.99
        de.update_suffering(agent)
        assert agent.suffering <= 1.0

        agent2 = _make_agent(traits, region=ProcessingRegion.UNDER_PROCESSING)
        agent2.suffering = 0.01
        de.update_suffering(agent2)
        assert agent2.suffering >= 0.0
