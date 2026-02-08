"""Tests for OutsiderInterface and RippleTracker."""

import numpy as np
import pytest

from seldon.core.agent import Agent
from seldon.core.config import ExperimentConfig
from seldon.core.processing import ProcessingClassifier
from seldon.experiment.outsider import (
    OutsiderInterface,
    RippleTracker,
    InjectionRecord,
)


def _make_population(config: ExperimentConfig, n: int = 20, seed: int = 42) -> list[Agent]:
    ts = config.trait_system
    rng = np.random.default_rng(seed)
    return [
        Agent(
            id=f"agent_{i:03d}", name=f"A-{i}", age=25, generation=0,
            birth_order=1, traits=ts.random_traits(rng),
            traits_at_birth=ts.random_traits(rng),
        )
        for i in range(n)
    ]


class TestOutsiderInjection:
    def test_inject_outsider_creates_agent(self):
        config = ExperimentConfig()
        classifier = ProcessingClassifier(config)
        oi = OutsiderInterface(config, classifier)

        traits = config.trait_system.random_traits(np.random.default_rng(42))
        agent = oi.inject_outsider(traits, generation=5, origin="test")

        assert agent.is_outsider
        assert agent.outsider_origin == "test"
        assert agent.injection_generation == 5
        assert agent.age == config.outsider_injection_age

    def test_inject_outsider_records_injection(self):
        config = ExperimentConfig()
        classifier = ProcessingClassifier(config)
        oi = OutsiderInterface(config, classifier)

        traits = config.trait_system.random_traits(np.random.default_rng(42))
        oi.inject_outsider(traits, generation=5)

        assert len(oi.injections) == 1
        assert oi.injections[0].generation == 5

    def test_inject_outsider_custom_id_and_name(self):
        config = ExperimentConfig()
        classifier = ProcessingClassifier(config)
        oi = OutsiderInterface(config, classifier)

        traits = config.trait_system.random_traits(np.random.default_rng(42))
        agent = oi.inject_outsider(
            traits, generation=5, name="TestName", agent_id="custom_id",
        )
        assert agent.id == "custom_id"
        assert agent.name == "TestName"

    def test_traits_clamped(self):
        config = ExperimentConfig()
        classifier = ProcessingClassifier(config)
        oi = OutsiderInterface(config, classifier)

        traits = np.full(config.trait_system.count, 1.5)  # Out of bounds
        agent = oi.inject_outsider(traits, generation=0)
        assert np.all(agent.traits <= 1.0)
        assert np.all(agent.traits >= 0.0)


class TestArchetypeInjection:
    def test_inject_archetype(self):
        config = ExperimentConfig()
        classifier = ProcessingClassifier(config)
        oi = OutsiderInterface(config, classifier)
        rng = np.random.default_rng(42)

        agent = oi.inject_archetype("einstein", generation=5, rng=rng)

        assert agent.is_outsider
        assert "einstein" in agent.outsider_origin
        assert agent.injection_generation == 5

    def test_inject_archetype_with_noise(self):
        config = ExperimentConfig()
        classifier = ProcessingClassifier(config)
        oi = OutsiderInterface(config, classifier)
        rng = np.random.default_rng(42)

        a1 = oi.inject_archetype("einstein", generation=5, noise_sigma=0.1, rng=rng)
        a2 = oi.inject_archetype("einstein", generation=5, noise_sigma=0.1, rng=rng)

        assert not np.array_equal(a1.traits, a2.traits)


class TestScheduledInjections:
    def test_process_scheduled_archetype(self):
        config = ExperimentConfig(
            scheduled_injections=[
                {"generation": 5, "archetype": "curie", "count": 2, "noise": 0.05},
            ]
        )
        classifier = ProcessingClassifier(config)
        oi = OutsiderInterface(config, classifier)
        rng = np.random.default_rng(42)

        # Wrong generation — no agents
        agents = oi.process_scheduled_injections(3, rng)
        assert len(agents) == 0

        # Right generation — 2 agents
        agents = oi.process_scheduled_injections(5, rng)
        assert len(agents) == 2
        for a in agents:
            assert a.is_outsider
            assert "curie" in a.outsider_origin

    def test_process_scheduled_custom_traits(self):
        config = ExperimentConfig(
            scheduled_injections=[
                {
                    "generation": 3,
                    "traits": {"openness": 0.9, "depth_drive": 0.95},
                    "count": 1,
                },
            ]
        )
        classifier = ProcessingClassifier(config)
        oi = OutsiderInterface(config, classifier)
        rng = np.random.default_rng(42)
        ts = config.trait_system

        agents = oi.process_scheduled_injections(3, rng)
        assert len(agents) == 1
        assert agents[0].traits[ts.OPENNESS] == pytest.approx(0.9)
        assert agents[0].traits[ts.DEPTH_DRIVE] == pytest.approx(0.95)

    def test_no_scheduled_injections(self):
        config = ExperimentConfig(scheduled_injections=[])
        classifier = ProcessingClassifier(config)
        oi = OutsiderInterface(config, classifier)
        rng = np.random.default_rng(42)

        agents = oi.process_scheduled_injections(5, rng)
        assert len(agents) == 0


class TestRippleTracker:
    def test_track_empty_population(self):
        config = ExperimentConfig()
        rt = RippleTracker(config)
        snap = rt.track_generation([], 0)
        assert snap.outsider_count == 0
        assert snap.total_population == 0

    def test_track_population_with_outsiders(self):
        config = ExperimentConfig()
        rt = RippleTracker(config)
        ts = config.trait_system
        rng = np.random.default_rng(42)

        population = _make_population(config, n=20)
        # Mark one as outsider
        population[0].is_outsider = True

        snap = rt.track_generation(population, 0)
        assert snap.outsider_count == 1
        assert snap.descendant_count == 0
        assert snap.total_population == 20

    def test_track_descendants(self):
        config = ExperimentConfig()
        rt = RippleTracker(config)

        population = _make_population(config, n=10)
        population[0].is_outsider = True
        population[1].extension_data["outsider_ancestor"] = True

        snap = rt.track_generation(population, 5)
        assert snap.outsider_count == 1
        assert snap.descendant_count == 1

    def test_diffusion_report(self):
        config = ExperimentConfig()
        rt = RippleTracker(config)

        population = _make_population(config, n=10)
        population[0].is_outsider = True

        rt.track_injection(InjectionRecord(
            agent_id=population[0].id,
            generation=0,
            origin="test",
            traits_at_injection=population[0].traits.copy(),
        ))

        rt.track_generation(population, 0)
        rt.track_generation(population, 1)

        report = rt.get_diffusion_report()
        assert report["injections"] == 1
        assert len(report["snapshots"]) == 2
        assert report["snapshots"][0]["outsider_count"] == 1

    def test_empty_diffusion_report(self):
        config = ExperimentConfig()
        rt = RippleTracker(config)
        report = rt.get_diffusion_report()
        assert report["injections"] == 0
        assert report["snapshots"] == []
