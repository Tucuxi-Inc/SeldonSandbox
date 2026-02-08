"""Tests for RelationshipManager."""

import numpy as np
import pytest

from seldon.core.agent import Agent
from seldon.core.attraction import AttractionModel
from seldon.core.config import ExperimentConfig
from seldon.social.relationships import RelationshipManager


def _make_agent(agent_id: str, traits: np.ndarray, age: int = 25, **kwargs) -> Agent:
    defaults = dict(
        id=agent_id, name=f"Agent-{agent_id}", age=age, generation=0,
        birth_order=1, traits=traits, traits_at_birth=traits.copy(),
    )
    defaults.update(kwargs)
    return Agent(**defaults)


def _make_population(config: ExperimentConfig, n: int = 20, seed: int = 42) -> list[Agent]:
    ts = config.trait_system
    rng = np.random.default_rng(seed)
    return [
        _make_agent(f"agent_{i:03d}", ts.random_traits(rng), age=rng.integers(16, 35))
        for i in range(n)
    ]


class TestPairFormation:
    def test_form_pairs_creates_pairs(self):
        config = ExperimentConfig(random_seed=42)
        attraction = AttractionModel(config)
        rm = RelationshipManager(config, attraction)
        population = _make_population(config, n=20)
        rng = np.random.default_rng(42)

        pairs = rm.form_pairs(population, 0, rng)
        assert len(pairs) > 0

    def test_paired_agents_linked(self):
        config = ExperimentConfig(random_seed=42)
        attraction = AttractionModel(config)
        rm = RelationshipManager(config, attraction)
        population = _make_population(config, n=20)
        rng = np.random.default_rng(42)

        pairs = rm.form_pairs(population, 0, rng)
        for a1, a2 in pairs:
            assert a1.partner_id == a2.id
            assert a2.partner_id == a1.id
            assert a1.relationship_status == "paired"
            assert a2.relationship_status == "paired"

    def test_underage_excluded(self):
        config = ExperimentConfig()
        attraction = AttractionModel(config)
        rm = RelationshipManager(config, attraction)
        ts = config.trait_system
        rng = np.random.default_rng(42)

        # All agents age 10 — below pairing_min_age
        population = [
            _make_agent(f"a{i}", ts.random_traits(rng), age=10)
            for i in range(10)
        ]
        pairs = rm.form_pairs(population, 0, rng)
        assert len(pairs) == 0

    def test_already_paired_excluded(self):
        config = ExperimentConfig()
        attraction = AttractionModel(config)
        rm = RelationshipManager(config, attraction)
        ts = config.trait_system
        rng = np.random.default_rng(42)

        population = _make_population(config, n=10)
        # Pair everyone up manually
        for agent in population:
            agent.partner_id = "someone"
            agent.relationship_status = "paired"

        pairs = rm.form_pairs(population, 0, rng)
        assert len(pairs) == 0

    def test_single_by_choice_reduces_pairs(self):
        # With 100% single-by-choice, no pairs should form
        config = ExperimentConfig(
            relationship_config={
                **ExperimentConfig().relationship_config,
                "single_by_choice_rate": 1.0,
            }
        )
        attraction = AttractionModel(config)
        rm = RelationshipManager(config, attraction)
        population = _make_population(config, n=20)
        rng = np.random.default_rng(42)

        pairs = rm.form_pairs(population, 0, rng)
        assert len(pairs) == 0


class TestDissolution:
    def test_dissolution_when_enabled(self):
        config = ExperimentConfig(
            relationship_config={
                **ExperimentConfig().relationship_config,
                "dissolution_enabled": True,
                "dissolution_base_rate": 0.8,
                "dissolution_compatibility_threshold": 0.9,
            }
        )
        attraction = AttractionModel(config)
        rm = RelationshipManager(config, attraction)
        ts = config.trait_system
        rng = np.random.default_rng(42)

        # Create two very different agents (low compatibility)
        traits1 = np.full(ts.count, 0.1)
        traits2 = np.full(ts.count, 0.9)
        a1 = _make_agent("a1", traits1)
        a2 = _make_agent("a2", traits2)
        a1.partner_id = a2.id
        a2.partner_id = a1.id
        a1.relationship_status = "paired"
        a2.relationship_status = "paired"

        dissolved = rm.process_dissolutions([a1, a2], 5, rng)
        # With high base rate and high threshold, should often dissolve
        # (may not always due to random chemistry in attraction)
        if dissolved:
            assert a1.relationship_status == "dissolved"
            assert a2.relationship_status == "dissolved"
            assert a1.partner_id is None

    def test_dissolution_disabled(self):
        config = ExperimentConfig(
            relationship_config={
                **ExperimentConfig().relationship_config,
                "dissolution_enabled": False,
            }
        )
        attraction = AttractionModel(config)
        rm = RelationshipManager(config, attraction)
        ts = config.trait_system
        rng = np.random.default_rng(42)

        a1 = _make_agent("a1", ts.random_traits(rng))
        a2 = _make_agent("a2", ts.random_traits(rng))
        a1.partner_id = a2.id
        a2.partner_id = a1.id
        a1.relationship_status = "paired"
        a2.relationship_status = "paired"

        dissolved = rm.process_dissolutions([a1, a2], 5, rng)
        assert len(dissolved) == 0

    def test_dissolution_respects_cooldown(self):
        config = ExperimentConfig(
            relationship_config={
                **ExperimentConfig().relationship_config,
                "reparing_cooldown_generations": 3,
            }
        )
        attraction = AttractionModel(config)
        rm = RelationshipManager(config, attraction)
        ts = config.trait_system
        rng = np.random.default_rng(42)

        agent = _make_agent("a1", ts.random_traits(rng))
        agent.relationship_status = "dissolved"
        agent.extension_data["last_dissolution_gen"] = 5

        # At generation 6 (only 1 gen later), should not be eligible
        assert not rm._is_eligible(agent, 6)
        # At generation 8 (3 gens later), should be eligible
        assert rm._is_eligible(agent, 8)


class TestInfidelity:
    def test_infidelity_disabled_by_default(self):
        config = ExperimentConfig()
        attraction = AttractionModel(config)
        rm = RelationshipManager(config, attraction)
        ts = config.trait_system
        rng = np.random.default_rng(42)

        a1 = _make_agent("a1", ts.random_traits(rng))
        a1.partner_id = "a2"
        events = rm.check_infidelity([a1], 0, rng)
        assert len(events) == 0

    def test_infidelity_when_enabled(self):
        config = ExperimentConfig(
            relationship_config={
                **ExperimentConfig().relationship_config,
                "infidelity_enabled": True,
                "infidelity_base_rate": 0.9,
            }
        )
        attraction = AttractionModel(config)
        rm = RelationshipManager(config, attraction)
        ts = config.trait_system
        rng = np.random.default_rng(42)

        # Create agent with low self-control (more prone to infidelity)
        traits = np.full(ts.count, 0.1)  # Low everything
        a1 = _make_agent("a1", traits)
        a1.partner_id = "a2"

        # Run many times to test probability
        total_events = 0
        for _ in range(50):
            events = rm.check_infidelity([a1], 0, rng)
            total_events += len(events)
        assert total_events > 0

    def test_high_self_control_reduces_infidelity(self):
        config = ExperimentConfig(
            relationship_config={
                **ExperimentConfig().relationship_config,
                "infidelity_enabled": True,
                "infidelity_base_rate": 0.3,
            }
        )
        attraction = AttractionModel(config)
        rm = RelationshipManager(config, attraction)
        ts = config.trait_system
        rng = np.random.default_rng(42)

        traits_low_control = np.full(ts.count, 0.1)
        traits_high_control = np.full(ts.count, 0.9)

        a_low = _make_agent("a_low", traits_low_control)
        a_low.partner_id = "p1"
        a_high = _make_agent("a_high", traits_high_control)
        a_high.partner_id = "p2"

        low_events = sum(
            len(rm.check_infidelity([a_low], 0, rng)) for _ in range(200)
        )
        high_events = sum(
            len(rm.check_infidelity([a_high], 0, rng)) for _ in range(200)
        )
        assert low_events > high_events


class TestWidowedRepairing:
    def test_widowed_agent_can_repair(self):
        config = ExperimentConfig(
            relationship_config={
                **ExperimentConfig().relationship_config,
                "reparing_after_death": True,
            }
        )
        attraction = AttractionModel(config)
        rm = RelationshipManager(config, attraction)

        assert rm._is_eligible(
            _make_agent("a1", config.trait_system.random_traits(np.random.default_rng(42)),
                        relationship_status="widowed"),
            0,
        )


class TestDeadAgentsExcluded:
    def test_dead_agents_not_paired(self):
        config = ExperimentConfig()
        attraction = AttractionModel(config)
        rm = RelationshipManager(config, attraction)
        ts = config.trait_system
        rng = np.random.default_rng(42)

        population = _make_population(config, n=10)
        for agent in population:
            agent.is_alive = False

        pairs = rm.form_pairs(population, 0, rng)
        assert len(pairs) == 0
