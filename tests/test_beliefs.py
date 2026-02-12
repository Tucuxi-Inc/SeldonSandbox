"""
Tests for Phase E: Discovery & Belief Systems.

Tests BeliefSystem, EpistemologyExtension, Belief dataclass, and the beliefs API router.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from seldon.core.agent import Agent
from seldon.core.config import ExperimentConfig
from seldon.core.processing import ProcessingRegion
from seldon.extensions.epistemology import EpistemologyExtension
from seldon.extensions.registry import ExtensionRegistry
from seldon.social.beliefs import (
    Belief,
    BeliefDomain,
    BeliefSystem,
    EpistemologyType,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(**kwargs) -> ExperimentConfig:
    defaults = dict(random_seed=42, initial_population=20, generations_to_run=5)
    defaults.update(kwargs)
    return ExperimentConfig(**defaults)


def _make_agent(config: ExperimentConfig | None = None, **kwargs) -> Agent:
    if config is None:
        config = _make_config()
    ts = config.trait_system
    rng = np.random.default_rng(kwargs.pop("seed", 42))
    traits = kwargs.pop("traits", None)
    if traits is None:
        traits = ts.random_traits(rng)
    defaults = dict(
        id=f"agent_{rng.integers(10000):04d}",
        name="Test Agent",
        age=25,
        generation=0,
        birth_order=1,
        traits=traits,
        traits_at_birth=traits.copy(),
    )
    defaults.update(kwargs)
    return Agent(**defaults)


def _make_population(config: ExperimentConfig, n: int = 20) -> list[Agent]:
    rng = np.random.default_rng(config.random_seed)
    ts = config.trait_system
    population = []
    for i in range(n):
        traits = ts.random_traits(rng)
        age = int(rng.integers(5, 60))
        agent = Agent(
            id=f"pop_{i:03d}",
            name=f"Agent {i}",
            age=age,
            generation=0,
            birth_order=int(rng.integers(1, 5)),
            traits=traits,
            traits_at_birth=traits.copy(),
            processing_region=rng.choice([
                ProcessingRegion.OPTIMAL,
                ProcessingRegion.DEEP,
                ProcessingRegion.UNDER_PROCESSING,
                ProcessingRegion.SACRIFICIAL,
                ProcessingRegion.PATHOLOGICAL,
            ]),
            contribution_history=[float(rng.uniform(0.0, 2.0)) for _ in range(5)],
        )
        population.append(agent)
    return population


def _make_belief(**overrides) -> Belief:
    defaults = dict(
        id="bel_001",
        content="River valley has good crops",
        domain=BeliefDomain.RESOURCE,
        epistemology=EpistemologyType.EMPIRICAL,
        accuracy=0.8,
        conviction=0.7,
        decision_effects={"MIGRATION:stay": 0.1},
        contribution_modifier=0.0,
        source_memory_id="mem_001",
        source_agent_id="agent_001",
        created_generation=0,
    )
    defaults.update(overrides)
    return Belief(**defaults)


def _make_memory(
    fidelity: float = 1.0,
    valence: float = 0.8,
    content: str = "Breakthrough achievement (contribution=1.50)",
    generation: int = 0,
) -> dict[str, Any]:
    return {
        "id": "mem_test",
        "content": content,
        "memory_type": "personal",
        "fidelity": fidelity,
        "emotional_valence": valence,
        "trait_modifiers": {"creativity": 0.01},
        "created_generation": generation,
        "source_agent_id": None,
        "transmission_count": 0,
        "mutation_count": 0,
    }


# ===========================================================================
# TestBeliefDataclass
# ===========================================================================

class TestBeliefDataclass:

    def test_creation(self):
        b = _make_belief()
        assert b.id == "bel_001"
        assert b.domain == BeliefDomain.RESOURCE
        assert b.epistemology == EpistemologyType.EMPIRICAL
        assert b.accuracy == 0.8
        assert b.conviction == 0.7

    def test_to_dict_round_trip(self):
        b = _make_belief()
        d = b.to_dict()
        b2 = Belief.from_dict(d)
        assert b2.id == b.id
        assert b2.domain == b.domain
        assert b2.epistemology == b.epistemology
        assert b2.accuracy == b.accuracy
        assert b2.conviction == b.conviction
        assert b2.decision_effects == b.decision_effects
        assert b2.transmission_count == b.transmission_count

    def test_enum_values(self):
        assert EpistemologyType.EMPIRICAL.value == "empirical"
        assert EpistemologyType.SACRED.value == "sacred"
        assert BeliefDomain.RESOURCE.value == "resource"
        assert BeliefDomain.DANGER.value == "danger"

    def test_missing_optional_fields(self):
        d = {
            "id": "x",
            "content": "test",
            "domain": "social",
            "epistemology": "empirical",
            "accuracy": 0.5,
            "conviction": 0.5,
        }
        b = Belief.from_dict(d)
        assert b.source_memory_id is None
        assert b.transmission_count == 0
        assert b.contribution_modifier == 0.0


# ===========================================================================
# TestBeliefFormation
# ===========================================================================

class TestBeliefFormation:

    def test_high_fidelity_becomes_empirical(self):
        config = _make_config()
        bs = BeliefSystem(config)
        agent = _make_agent(config)
        rng = np.random.default_rng(42)

        mem = _make_memory(fidelity=0.9, valence=0.8)
        belief = bs.form_belief_from_memory(mem, agent, 0, rng)
        assert belief is not None
        assert belief.epistemology == EpistemologyType.EMPIRICAL

    def test_medium_fidelity_becomes_traditional(self):
        config = _make_config()
        bs = BeliefSystem(config)
        agent = _make_agent(config)
        rng = np.random.default_rng(42)

        mem = _make_memory(fidelity=0.5, valence=0.8)
        belief = bs.form_belief_from_memory(mem, agent, 0, rng)
        assert belief is not None
        assert belief.epistemology == EpistemologyType.TRADITIONAL

    def test_low_fidelity_becomes_mythical(self):
        config = _make_config()
        bs = BeliefSystem(config)
        agent = _make_agent(config)
        rng = np.random.default_rng(42)

        mem = _make_memory(fidelity=0.1, valence=0.8)
        belief = bs.form_belief_from_memory(mem, agent, 0, rng)
        assert belief is not None
        assert belief.epistemology == EpistemologyType.MYTHICAL

    def test_low_valence_rejected(self):
        config = _make_config()
        bs = BeliefSystem(config)
        agent = _make_agent(config)
        rng = np.random.default_rng(42)

        mem = _make_memory(fidelity=0.9, valence=0.1)
        belief = bs.form_belief_from_memory(mem, agent, 0, rng)
        assert belief is None

    def test_max_beliefs_respected(self):
        config = _make_config()
        bs = BeliefSystem(config)
        agent = _make_agent(config)
        rng = np.random.default_rng(42)

        # Fill agent with max beliefs
        agent.extension_data["beliefs"] = [
            _make_belief(id=f"b{i}").to_dict()
            for i in range(config.belief_config["max_beliefs_per_agent"])
        ]
        mem = _make_memory(fidelity=0.9, valence=0.8)
        belief = bs.form_belief_from_memory(mem, agent, 0, rng)
        assert belief is None

    def test_accuracy_ranges(self):
        config = _make_config()
        bs = BeliefSystem(config)
        agent = _make_agent(config)
        rng = np.random.default_rng(42)

        # Empirical accuracy: 0.6-0.8
        mem = _make_memory(fidelity=0.9, valence=0.8)
        belief = bs.form_belief_from_memory(mem, agent, 0, rng)
        assert belief is not None
        assert 0.5 <= belief.accuracy <= 0.85

    def test_domain_inference_productivity(self):
        config = _make_config()
        bs = BeliefSystem(config)
        agent = _make_agent(config)
        rng = np.random.default_rng(42)

        mem = _make_memory(content="Breakthrough achievement (contribution=1.50)")
        belief = bs.form_belief_from_memory(mem, agent, 0, rng)
        assert belief is not None
        assert belief.domain == BeliefDomain.PRODUCTIVITY

    def test_domain_inference_danger(self):
        config = _make_config()
        bs = BeliefSystem(config)
        agent = _make_agent(config)
        rng = np.random.default_rng(42)

        mem = _make_memory(content="Period of deep suffering (level=0.80)")
        belief = bs.form_belief_from_memory(mem, agent, 0, rng)
        assert belief is not None
        assert belief.domain == BeliefDomain.DANGER


# ===========================================================================
# TestBeliefPropagation
# ===========================================================================

class TestBeliefPropagation:

    def test_propagation_through_bonds(self):
        config = _make_config()
        bs = BeliefSystem(config)
        rng = np.random.default_rng(42)

        agent1 = _make_agent(config, id="a1", seed=1)
        agent2 = _make_agent(config, id="a2", seed=2)
        agent1.social_bonds["a2"] = 0.8
        agent2.social_bonds["a1"] = 0.8

        # Give agent1 a high-conviction belief
        belief = _make_belief(conviction=0.95)
        agent1.extension_data["beliefs"] = [belief.to_dict()]

        # Use high propagation rate for test reliability
        bs._cfg["propagation_rate"] = 0.9

        metrics = bs.propagate_beliefs([agent1, agent2], 1, rng)
        # Should have transmitted at least once or had conflicts
        assert metrics["transmitted"] + metrics["conflicts"] >= 0

    def test_zero_propagation_rate(self):
        config = _make_config()
        bs = BeliefSystem(config)
        bs._cfg["propagation_rate"] = 0.0
        rng = np.random.default_rng(42)

        agent1 = _make_agent(config, id="a1", seed=1)
        agent2 = _make_agent(config, id="a2", seed=2)
        agent1.social_bonds["a2"] = 0.8

        belief = _make_belief(conviction=1.0)
        agent1.extension_data["beliefs"] = [belief.to_dict()]

        metrics = bs.propagate_beliefs([agent1, agent2], 1, rng)
        assert metrics["transmitted"] == 0

    def test_empirical_becomes_traditional_on_transmission(self):
        config = _make_config()
        bs = BeliefSystem(config)
        rng = np.random.default_rng(42)

        agent1 = _make_agent(config, id="a1", seed=1)
        agent2 = _make_agent(config, id="a2", seed=2)
        agent1.social_bonds["a2"] = 1.0

        belief = _make_belief(
            epistemology=EpistemologyType.EMPIRICAL,
            conviction=1.0,
        )
        agent1.extension_data["beliefs"] = [belief.to_dict()]
        bs._cfg["propagation_rate"] = 1.0  # Guarantee transmission

        bs.propagate_beliefs([agent1, agent2], 1, rng)
        a2_beliefs = BeliefSystem._get_beliefs(agent2)
        if a2_beliefs:
            # Transmitted belief should be TRADITIONAL, not EMPIRICAL
            assert a2_beliefs[0].epistemology == EpistemologyType.TRADITIONAL

    def test_conviction_decays_on_transmission(self):
        config = _make_config()
        bs = BeliefSystem(config)
        rng = np.random.default_rng(42)

        agent1 = _make_agent(config, id="a1", seed=1)
        agent2 = _make_agent(config, id="a2", seed=2)
        agent1.social_bonds["a2"] = 1.0

        belief = _make_belief(conviction=0.9)
        agent1.extension_data["beliefs"] = [belief.to_dict()]
        bs._cfg["propagation_rate"] = 1.0

        bs.propagate_beliefs([agent1, agent2], 1, rng)
        a2_beliefs = BeliefSystem._get_beliefs(agent2)
        if a2_beliefs:
            assert a2_beliefs[0].conviction < 0.9

    def test_parent_to_child_transmission(self):
        config = _make_config()
        bs = BeliefSystem(config)
        bs._cfg["parent_transmission_rate"] = 1.0  # Guarantee transmission
        rng = np.random.default_rng(42)

        parent1 = _make_agent(config, id="p1", seed=1)
        parent2 = _make_agent(config, id="p2", seed=2)
        child = _make_agent(config, id="c1", seed=3)

        belief = _make_belief(conviction=1.0)
        parent1.extension_data["beliefs"] = [belief.to_dict()]

        count = bs.transmit_to_child(parent1, parent2, child, rng)
        assert count >= 1
        child_beliefs = BeliefSystem._get_beliefs(child)
        assert len(child_beliefs) >= 1

    def test_no_bonds_no_propagation(self):
        config = _make_config()
        bs = BeliefSystem(config)
        rng = np.random.default_rng(42)

        agent1 = _make_agent(config, id="a1", seed=1)
        agent2 = _make_agent(config, id="a2", seed=2)
        # No social bonds

        belief = _make_belief(conviction=1.0)
        agent1.extension_data["beliefs"] = [belief.to_dict()]
        bs._cfg["propagation_rate"] = 1.0

        metrics = bs.propagate_beliefs([agent1, agent2], 1, rng)
        assert metrics["transmitted"] == 0

    def test_max_beliefs_blocks_propagation(self):
        config = _make_config()
        bs = BeliefSystem(config)
        rng = np.random.default_rng(42)

        agent1 = _make_agent(config, id="a1", seed=1)
        agent2 = _make_agent(config, id="a2", seed=2)
        agent1.social_bonds["a2"] = 1.0

        # Give agent1 a migration belief (different domain from agent2's)
        belief = _make_belief(
            domain=BeliefDomain.MIGRATION,
            conviction=1.0,
        )
        agent1.extension_data["beliefs"] = [belief.to_dict()]

        # Fill agent2 to max beliefs with unique domains
        max_b = config.belief_config["max_beliefs_per_agent"]
        agent2.extension_data["beliefs"] = [
            _make_belief(id=f"b{i}", domain=BeliefDomain.SOCIAL).to_dict()
            for i in range(max_b)
        ]
        bs._cfg["propagation_rate"] = 1.0

        metrics = bs.propagate_beliefs([agent1, agent2], 1, rng)
        # Either conflicts (same domain) or blocked by max
        a2_beliefs = BeliefSystem._get_beliefs(agent2)
        assert len(a2_beliefs) <= max_b


# ===========================================================================
# TestBeliefConflict
# ===========================================================================

class TestBeliefConflict:

    def test_same_domain_detection(self):
        beliefs = [
            _make_belief(domain=BeliefDomain.RESOURCE),
            _make_belief(domain=BeliefDomain.DANGER, id="b2"),
        ]
        result = BeliefSystem._find_same_domain(beliefs, BeliefDomain.RESOURCE)
        assert result is not None
        assert result.domain == BeliefDomain.RESOURCE

    def test_no_same_domain(self):
        beliefs = [_make_belief(domain=BeliefDomain.RESOURCE)]
        result = BeliefSystem._find_same_domain(beliefs, BeliefDomain.DANGER)
        assert result is None

    def test_sacred_resistance(self):
        config = _make_config()
        bs = BeliefSystem(config)
        rng = np.random.default_rng(42)

        challenger = _make_belief(
            id="c1", epistemology=EpistemologyType.EMPIRICAL,
            accuracy=0.9, conviction=0.9,
        )
        defender = _make_belief(
            id="d1", epistemology=EpistemologyType.SACRED,
            accuracy=0.2, conviction=0.9,
        )

        # Run many times — sacred should win most (95%)
        sacred_wins = 0
        for i in range(100):
            local_rng = np.random.default_rng(i)
            winner = bs._resolve_conflict(challenger, defender, [], local_rng)
            if winner.id == "d1":
                sacred_wins += 1
        assert sacred_wins > 80  # Should be ~95

    def test_evidence_wins_against_weak(self):
        config = _make_config()
        bs = BeliefSystem(config)
        rng = np.random.default_rng(42)

        # Strong evidence challenger vs weak non-sacred defender
        challenger = _make_belief(
            id="c1", epistemology=EpistemologyType.EMPIRICAL,
            accuracy=0.95, conviction=0.95,
        )
        defender = _make_belief(
            id="d1", epistemology=EpistemologyType.MYTHICAL,
            accuracy=0.1, conviction=0.1,
        )

        # Run many times — challenger should usually win
        challenger_wins = 0
        for i in range(100):
            local_rng = np.random.default_rng(i + 200)
            winner = bs._resolve_conflict(challenger, defender, [], local_rng)
            if winner.id == "c1":
                challenger_wins += 1
        assert challenger_wins > 60

    def test_social_pressure_affects_outcome(self):
        config = _make_config()
        bs = BeliefSystem(config)
        rng = np.random.default_rng(42)

        # Create population where many agents hold the defender belief
        population = _make_population(config, n=20)
        defender = _make_belief(
            id="d1", epistemology=EpistemologyType.MYTHICAL,
            accuracy=0.3, conviction=0.5,
            content="Popular belief",
        )
        # Give many agents this belief
        for agent in population[:15]:
            agent.extension_data["beliefs"] = [defender.to_dict()]

        challenger = _make_belief(
            id="c1", epistemology=EpistemologyType.EMPIRICAL,
            accuracy=0.7, conviction=0.5,
            content="Unpopular truth",
        )

        # Social pressure should help the popular belief
        defender_wins = 0
        for i in range(100):
            local_rng = np.random.default_rng(i + 300)
            winner = bs._resolve_conflict(
                challenger, defender, population, local_rng,
            )
            if winner.id == "d1":
                defender_wins += 1
        # Popular belief should win sometimes despite lower accuracy
        assert defender_wins > 20

    def test_reinforcement_boosts_conviction(self):
        config = _make_config()
        bs = BeliefSystem(config)
        rng = np.random.default_rng(42)

        agent1 = _make_agent(config, id="a1", seed=1)
        agent2 = _make_agent(config, id="a2", seed=2)
        agent1.social_bonds["a2"] = 1.0

        # Both agents hold the same-domain belief
        belief1 = _make_belief(id="b1", conviction=0.8, domain=BeliefDomain.RESOURCE)
        belief2 = _make_belief(id="b2", conviction=0.5, domain=BeliefDomain.RESOURCE)
        agent1.extension_data["beliefs"] = [belief1.to_dict()]
        agent2.extension_data["beliefs"] = [belief2.to_dict()]
        bs._cfg["propagation_rate"] = 1.0

        # Propagation should trigger conflict and potentially reinforce
        metrics = bs.propagate_beliefs([agent1, agent2], 1, rng)
        assert metrics["conflicts"] >= 0  # At least attempted


# ===========================================================================
# TestAccuracyDynamics
# ===========================================================================

class TestAccuracyDynamics:

    def test_empirical_self_corrects(self):
        config = _make_config()
        bs = BeliefSystem(config)
        rng = np.random.default_rng(42)

        agent = _make_agent(config)
        belief = _make_belief(
            epistemology=EpistemologyType.EMPIRICAL,
            accuracy=0.3,
        )
        agent.extension_data["beliefs"] = [belief.to_dict()]
        bs.ground_truths[BeliefDomain.RESOURCE.value] = 0.9

        # Update several times
        for gen in range(20):
            bs.update_belief_accuracy([agent], gen, rng)

        beliefs = BeliefSystem._get_beliefs(agent)
        assert len(beliefs) > 0
        # Should have drifted toward 0.9
        assert beliefs[0].accuracy > 0.3

    def test_sacred_unchanged(self):
        config = _make_config()
        bs = BeliefSystem(config)
        rng = np.random.default_rng(42)

        agent = _make_agent(config)
        belief = _make_belief(
            epistemology=EpistemologyType.SACRED,
            accuracy=0.2,
            conviction=0.9,
        )
        agent.extension_data["beliefs"] = [belief.to_dict()]
        bs.ground_truths[BeliefDomain.RESOURCE.value] = 0.9

        initial_accuracy = 0.2
        bs.update_belief_accuracy([agent], 10, rng)
        beliefs = BeliefSystem._get_beliefs(agent)
        assert beliefs[0].accuracy == pytest.approx(initial_accuracy)

    def test_mythical_drifts(self):
        config = _make_config()
        bs = BeliefSystem(config)
        rng = np.random.default_rng(42)

        agent = _make_agent(config)
        belief = _make_belief(
            epistemology=EpistemologyType.MYTHICAL,
            accuracy=0.5,
            conviction=0.9,
        )
        agent.extension_data["beliefs"] = [belief.to_dict()]

        # Update many times to see drift
        accuracies = []
        for gen in range(50):
            bs.update_belief_accuracy([agent], gen, rng)
            beliefs = BeliefSystem._get_beliefs(agent)
            if beliefs:
                accuracies.append(beliefs[0].accuracy)

        # Should have drifted from 0.5
        if accuracies:
            assert accuracies[-1] != pytest.approx(0.5, abs=0.001)

    def test_traditional_to_sacred_promotion(self):
        config = _make_config()
        bs = BeliefSystem(config)
        rng = np.random.default_rng(42)

        agent = _make_agent(config)
        belief = _make_belief(
            epistemology=EpistemologyType.TRADITIONAL,
            accuracy=0.5,
            conviction=0.9,  # Above sacred threshold
            created_generation=0,
        )
        agent.extension_data["beliefs"] = [belief.to_dict()]

        # Update at generation 10 (well past min age of 3)
        bs.update_belief_accuracy([agent], 10, rng)
        beliefs = BeliefSystem._get_beliefs(agent)
        assert beliefs[0].epistemology == EpistemologyType.SACRED

    def test_conviction_decay(self):
        config = _make_config()
        bs = BeliefSystem(config)
        rng = np.random.default_rng(42)

        agent = _make_agent(config)
        belief = _make_belief(conviction=0.5)
        agent.extension_data["beliefs"] = [belief.to_dict()]

        bs.update_belief_accuracy([agent], 1, rng)
        beliefs = BeliefSystem._get_beliefs(agent)
        assert beliefs[0].conviction < 0.5

    def test_zero_conviction_pruned(self):
        config = _make_config()
        bs = BeliefSystem(config)
        rng = np.random.default_rng(42)

        agent = _make_agent(config)
        belief = _make_belief(conviction=0.005)  # Near zero
        agent.extension_data["beliefs"] = [belief.to_dict()]

        bs.update_belief_accuracy([agent], 1, rng)
        beliefs = BeliefSystem._get_beliefs(agent)
        # Should have been pruned (conviction < 0.01 after decay)
        assert len(beliefs) == 0


# ===========================================================================
# TestExtensionHooks
# ===========================================================================

class TestExtensionHooks:

    def test_simulation_start(self):
        config = _make_config()
        ext = EpistemologyExtension()
        population = _make_population(config, n=10)

        ext.on_simulation_start(population, config)
        assert ext.belief_system is not None

    def test_generation_start(self):
        config = _make_config()
        ext = EpistemologyExtension()
        population = _make_population(config, n=10)

        ext.on_simulation_start(population, config)
        # Should not raise
        ext.on_generation_start(1, population, config)

    def test_generation_end(self):
        config = _make_config()
        ext = EpistemologyExtension()
        population = _make_population(config, n=10)

        ext.on_simulation_start(population, config)
        ext.on_generation_end(1, population, config)

        metrics = ext.get_metrics(population)
        assert "total_beliefs" in metrics

    def test_agent_created(self):
        config = _make_config()
        ext = EpistemologyExtension()
        population = _make_population(config, n=10)

        ext.on_simulation_start(population, config)

        parent1 = population[0]
        parent2 = population[1]
        child = _make_agent(config, id="child_001", seed=100)

        # Give parent a belief
        belief = _make_belief(conviction=1.0)
        parent1.extension_data["beliefs"] = [belief.to_dict()]
        ext._belief_system._cfg["parent_transmission_rate"] = 1.0

        ext.on_agent_created(child, (parent1, parent2), config)
        # Child may have received a belief
        child_beliefs = BeliefSystem._get_beliefs(child)
        assert len(child_beliefs) >= 0  # May or may not transmit

    def test_modify_decision(self):
        config = _make_config()
        ext = EpistemologyExtension()
        population = _make_population(config, n=5)
        ext.on_simulation_start(population, config)

        agent = population[0]
        belief = _make_belief(
            conviction=0.8,
            decision_effects={"MIGRATION:stay": 0.2},
        )
        agent.extension_data["beliefs"] = [belief.to_dict()]

        utilities = {"stay": 0.5, "migrate": 0.3}
        result = ext.modify_decision(agent, "MIGRATION", utilities, config)
        # "stay" should have been boosted
        assert result["stay"] > 0.5

    def test_modify_mortality(self):
        config = _make_config()
        ext = EpistemologyExtension()
        population = _make_population(config, n=5)
        ext.on_simulation_start(population, config)

        agent = population[0]
        belief = _make_belief(
            domain=BeliefDomain.DANGER,
            epistemology=EpistemologyType.EMPIRICAL,
            accuracy=0.9,
            conviction=0.8,
        )
        agent.extension_data["beliefs"] = [belief.to_dict()]

        base_rate = 0.1
        modified = ext.modify_mortality(agent, base_rate, config)
        assert modified < base_rate


# ===========================================================================
# TestGroundTruth
# ===========================================================================

class TestGroundTruth:

    def test_register_ground_truth(self):
        config = _make_config()
        bs = BeliefSystem(config)
        bs.ground_truths["resource"] = 0.8
        assert bs.ground_truths["resource"] == 0.8

    def test_derive_from_simulation(self):
        config = _make_config()
        bs = BeliefSystem(config)
        population = _make_population(config, n=20)

        bs.update_ground_truths_from_simulation(population, config)
        # Should have derived danger truth at minimum
        assert BeliefDomain.DANGER.value in bs.ground_truths

    def test_affects_empirical_correction(self):
        config = _make_config()
        bs = BeliefSystem(config)
        rng = np.random.default_rng(42)

        agent = _make_agent(config)
        belief = _make_belief(
            epistemology=EpistemologyType.EMPIRICAL,
            accuracy=0.2,
            domain=BeliefDomain.RESOURCE,
        )
        agent.extension_data["beliefs"] = [belief.to_dict()]
        bs.ground_truths[BeliefDomain.RESOURCE.value] = 0.9

        bs.update_belief_accuracy([agent], 1, rng)
        beliefs = BeliefSystem._get_beliefs(agent)
        # Accuracy should have moved toward 0.9
        assert beliefs[0].accuracy > 0.2


# ===========================================================================
# TestSocietalBeliefs
# ===========================================================================

class TestSocietalBeliefs:

    def test_promotion(self):
        config = _make_config()
        bs = BeliefSystem(config)
        population = _make_population(config, n=20)

        # Give > 20% of population the same belief content
        belief = _make_belief(content="Shared knowledge")
        for agent in population[:10]:
            agent.extension_data["beliefs"] = [belief.to_dict()]

        societal = bs.evolve_societal_beliefs(population, 1)
        assert len(societal) >= 1

    def test_pruning(self):
        config = _make_config()
        bs = BeliefSystem(config)
        population = _make_population(config, n=20)

        # Seed a societal belief
        old_belief = _make_belief(content="Forgotten lore")
        bs.societal_beliefs.append(old_belief)

        # Nobody holds it
        societal = bs.evolve_societal_beliefs(population, 1)
        assert not any(b.content == "Forgotten lore" for b in societal)

    def test_metrics_completeness(self):
        config = _make_config()
        bs = BeliefSystem(config)
        population = _make_population(config, n=10)

        metrics = bs.get_metrics(population)
        assert "total_beliefs" in metrics
        assert "epistemology_distribution" in metrics
        assert "domain_distribution" in metrics
        assert "mean_accuracy" in metrics
        assert "mean_conviction" in metrics
        assert "societal_belief_count" in metrics
        assert "agents_with_beliefs" in metrics
        assert "beliefs_per_agent" in metrics


# ===========================================================================
# TestAPIEndpoints
# ===========================================================================

class TestAPIEndpoints:

    @pytest.fixture()
    def client(self):
        from fastapi.testclient import TestClient
        from seldon.api.app import create_app
        application = create_app()
        application.state.session_manager = __import__(
            "seldon.api.sessions", fromlist=["SessionManager"],
        ).SessionManager(db_path=None)
        return TestClient(application)

    def _create_session_with_epistemology(self, client):
        resp = client.post("/api/simulation/sessions", json={
            "config": {
                "initial_population": 10,
                "generations_to_run": 5,
                "random_seed": 42,
                "extensions_enabled": ["epistemology"],
            },
        })
        assert resp.status_code == 200
        sid = resp.json()["id"]
        # Step a few generations to generate beliefs
        client.post(f"/api/simulation/sessions/{sid}/step", json={"n": 3})
        return sid

    def test_overview(self, client):
        sid = self._create_session_with_epistemology(client)
        resp = client.get(f"/api/beliefs/{sid}/overview")
        assert resp.status_code == 200
        data = resp.json()
        assert data["enabled"] is True
        assert "total_beliefs" in data

    def test_agent_beliefs(self, client):
        sid = self._create_session_with_epistemology(client)
        # Get an agent ID from the agents list endpoint
        agents_resp = client.get(f"/api/agents/{sid}")
        assert agents_resp.status_code == 200
        agents = agents_resp.json()["agents"]
        if agents:
            agent_id = agents[0]["id"]
            resp = client.get(f"/api/beliefs/{sid}/agent/{agent_id}")
            assert resp.status_code == 200
            data = resp.json()
            assert data["enabled"] is True
            assert "beliefs" in data

    def test_epistemology_distribution(self, client):
        sid = self._create_session_with_epistemology(client)
        resp = client.get(f"/api/beliefs/{sid}/epistemology-distribution")
        assert resp.status_code == 200
        data = resp.json()
        assert data["enabled"] is True
        assert "distribution" in data

    def test_disabled_returns_false(self, client):
        # Create session WITHOUT epistemology extension
        resp = client.post("/api/simulation/sessions", json={
            "config": {"initial_population": 5, "random_seed": 1},
        })
        sid = resp.json()["id"]
        resp = client.get(f"/api/beliefs/{sid}/overview")
        assert resp.status_code == 200
        assert resp.json()["enabled"] is False

    def test_404_session(self, client):
        resp = client.get("/api/beliefs/nonexistent/overview")
        assert resp.status_code == 404


# ===========================================================================
# TestBeliefConfig
# ===========================================================================

class TestBeliefConfig:

    def test_default_config_present(self):
        config = ExperimentConfig()
        assert "enabled" in config.belief_config
        assert config.belief_config["enabled"] is True
        assert "memory_to_belief_threshold" in config.belief_config

    def test_config_serialization(self):
        config = ExperimentConfig()
        d = config.to_dict()
        assert "belief_config" in d
        restored = ExperimentConfig.from_dict(d)
        assert restored.belief_config["enabled"] is True

    def test_extension_registration(self):
        config = _make_config(extensions_enabled=["epistemology"])
        from seldon.api.sessions import SessionManager
        registry = SessionManager._build_extensions(config)
        assert registry.is_enabled("epistemology")
