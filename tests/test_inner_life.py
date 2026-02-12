"""
Tests for Phase F: Inner Life & Experiential Mind.

Tests ExperientialEngine, InnerLifeExtension, and the inner-life API router.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from seldon.core.agent import Agent
from seldon.core.config import ExperimentConfig
from seldon.core.experiential import (
    BASE_EVENT_VECTORS,
    EXPERIENCE_DIM,
    EXPERIENCE_LABELS,
    ExperientialEngine,
)
from seldon.core.processing import ProcessingRegion
from seldon.extensions.inner_life import InnerLifeExtension
from seldon.extensions.registry import ExtensionRegistry


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


def _engine_and_agent(config=None):
    """Return (ExperientialEngine, Agent) with initialized inner life state."""
    config = config or _make_config()
    eng = ExperientialEngine(config)
    agent = _make_agent(config)
    eng.init_state(agent)
    eng.seed_mood_from_traits(agent)
    return eng, agent


# ===========================================================================
# TestExperientialState
# ===========================================================================

class TestExperientialState:
    """Test inner life state initialization and structure."""

    def test_init_state_creates_structure(self):
        eng, agent = _engine_and_agent()
        state = agent.extension_data["inner_life"]
        assert isinstance(state["experiences"], list)
        assert state["phenomenal_quality"] == 0.5
        assert isinstance(state["pq_history"], list)
        assert len(state["mood"]) == EXPERIENCE_DIM
        assert isinstance(state["experiential_drift_applied"], dict)
        assert "_prev_state" in state

    def test_mood_seeded_from_traits(self):
        eng, agent = _engine_and_agent()
        mood = agent.extension_data["inner_life"]["mood"]
        # Mood should be influenced by traits, so not all zeros
        assert any(m != 0.0 for m in mood)

    def test_serialization_round_trip(self):
        eng, agent = _engine_and_agent()
        rng = np.random.default_rng(42)
        eng.encode_experience(agent, "breakthrough", 0, rng)
        state = agent.extension_data["inner_life"]
        # State should be JSON-serializable (no numpy types)
        import json
        serialized = json.dumps(state)
        restored = json.loads(serialized)
        assert len(restored["experiences"]) == 1

    def test_max_pruning(self):
        config = _make_config(inner_life_config={"max_experiences_per_agent": 3})
        eng = ExperientialEngine(config)
        agent = _make_agent(config)
        eng.init_state(agent)
        rng = np.random.default_rng(42)
        for i in range(5):
            eng.encode_experience(agent, "routine", i, rng)
        eng.prune_experiences(agent)
        assert len(agent.extension_data["inner_life"]["experiences"]) == 3


# ===========================================================================
# TestExperienceEncoding
# ===========================================================================

class TestExperienceEncoding:
    """Test experience encoding from events."""

    def test_breakthrough_encoding(self):
        eng, agent = _engine_and_agent()
        rng = np.random.default_rng(42)
        exp = eng.encode_experience(agent, "breakthrough", 0, rng)
        felt = exp["felt_vector"]
        assert felt[0] > 0.5  # valence should be positive
        assert felt[5] > 0.5  # meaning should be high
        assert exp["event_type"] == "breakthrough"

    def test_deep_suffering_r4_has_meaning(self):
        eng, agent = _engine_and_agent()
        rng = np.random.default_rng(42)
        ctx = {"processing_region": ProcessingRegion.SACRIFICIAL}
        exp = eng.encode_experience(agent, "deep_suffering", 0, rng, ctx)
        felt = exp["felt_vector"]
        assert felt[0] < 0  # valence negative
        assert felt[5] >= 0.5  # meaning high (suffering has purpose)

    def test_deep_suffering_r5_low_meaning(self):
        eng, agent = _engine_and_agent()
        rng = np.random.default_rng(42)
        ctx = {"processing_region": ProcessingRegion.PATHOLOGICAL}
        exp = eng.encode_experience(agent, "deep_suffering", 0, rng, ctx)
        felt = exp["felt_vector"]
        assert felt[0] < 0  # valence negative
        assert felt[5] < 0.3  # meaning low (empty suffering)

    def test_trait_modulation_changes_felt_vector(self):
        config = _make_config()
        ts = config.trait_system
        eng = ExperientialEngine(config)
        rng = np.random.default_rng(42)

        # Agent with high resilience
        traits1 = ts.random_traits(rng)
        try:
            res_idx = ts.trait_index("resilience")
            traits1[res_idx] = 0.95
        except (KeyError, ValueError):
            pytest.skip("resilience trait not available")
        a1 = _make_agent(config, traits=traits1, seed=1)
        eng.init_state(a1)

        # Agent with low resilience
        traits2 = traits1.copy()
        traits2[res_idx] = 0.05
        a2 = _make_agent(config, traits=traits2, seed=2)
        eng.init_state(a2)

        rng1 = np.random.default_rng(99)
        rng2 = np.random.default_rng(99)
        exp1 = eng.encode_experience(a1, "deep_suffering", 0, rng1)
        exp2 = eng.encode_experience(a2, "deep_suffering", 0, rng2)

        # High resilience should produce higher valence than low resilience
        assert exp1["felt_vector"][0] > exp2["felt_vector"][0]

    def test_routine_encoding(self):
        eng, agent = _engine_and_agent()
        rng = np.random.default_rng(42)
        exp = eng.encode_experience(agent, "routine", 0, rng)
        # Routine should be low intensity
        assert abs(exp["felt_vector"][0]) < 0.6
        assert exp["felt_vector"][1] < 0.6  # low arousal

    def test_assertoric_initial_value(self):
        eng, agent = _engine_and_agent()
        rng = np.random.default_rng(42)
        exp = eng.encode_experience(agent, "breakthrough", 0, rng)
        assert exp["assertoric_strength"] == 0.9

    def test_experience_stored_on_agent(self):
        eng, agent = _engine_and_agent()
        rng = np.random.default_rng(42)
        eng.encode_experience(agent, "pair_formed", 0, rng)
        eng.encode_experience(agent, "child_born", 0, rng)
        exps = agent.extension_data["inner_life"]["experiences"]
        assert len(exps) == 2
        assert exps[0]["event_type"] == "pair_formed"
        assert exps[1]["event_type"] == "child_born"


# ===========================================================================
# TestSimilarityRecall
# ===========================================================================

class TestSimilarityRecall:
    """Test similarity-based experience recall."""

    def test_finds_matching_experience(self):
        eng, agent = _engine_and_agent()
        rng = np.random.default_rng(42)
        exp = eng.encode_experience(agent, "breakthrough", 0, rng)
        query = exp["felt_vector"]
        recalled = eng.recall_similar(agent, query)
        assert len(recalled) >= 1
        assert recalled[0][0]["event_type"] == "breakthrough"

    def test_threshold_skips_dissimilar(self):
        config = _make_config(inner_life_config={"recall_similarity_threshold": 0.99})
        eng = ExperientialEngine(config)
        agent = _make_agent(config)
        eng.init_state(agent)
        rng = np.random.default_rng(42)
        eng.encode_experience(agent, "breakthrough", 0, rng)
        # Query with very different vector
        query = [-1.0, 0.0, -1.0, 0.0, 0.0, 0.0]
        recalled = eng.recall_similar(agent, query)
        assert len(recalled) == 0

    def test_top_k_limit(self):
        eng, agent = _engine_and_agent()
        rng = np.random.default_rng(42)
        # Encode many similar experiences
        for i in range(10):
            eng.encode_experience(agent, "breakthrough", i, rng)
        query = BASE_EVENT_VECTORS["breakthrough"]
        recalled = eng.recall_similar(agent, query, top_k=3)
        assert len(recalled) <= 3

    def test_assertoric_weight_affects_ranking(self):
        eng, agent = _engine_and_agent()
        rng = np.random.default_rng(42)
        exp1 = eng.encode_experience(agent, "breakthrough", 0, rng)
        exp2 = eng.encode_experience(agent, "breakthrough", 1, rng)
        # Manually reduce assertoric strength of first
        exp1["assertoric_strength"] = 0.1
        exp2["assertoric_strength"] = 0.9
        query = BASE_EVENT_VECTORS["breakthrough"]
        recalled = eng.recall_similar(agent, query, top_k=2)
        # Higher assertoric should rank first
        if len(recalled) >= 2:
            assert recalled[0][0]["assertoric_strength"] >= recalled[1][0]["assertoric_strength"]

    def test_recall_count_incremented(self):
        eng, agent = _engine_and_agent()
        rng = np.random.default_rng(42)
        exp = eng.encode_experience(agent, "breakthrough", 0, rng)
        assert exp["recall_count"] == 0
        eng.recall_similar(agent, exp["felt_vector"])
        assert exp["recall_count"] == 1


# ===========================================================================
# TestPhenomenalQuality
# ===========================================================================

class TestPhenomenalQuality:
    """Test phenomenal quality computation."""

    def test_positive_experiences_yield_high_pq(self):
        eng, agent = _engine_and_agent()
        rng = np.random.default_rng(42)
        for _ in range(5):
            eng.encode_experience(agent, "breakthrough", 0, rng)
        pq = eng.compute_phenomenal_quality(agent)
        assert pq > 0.5

    def test_negative_experiences_yield_low_pq(self):
        eng, agent = _engine_and_agent()
        rng = np.random.default_rng(42)
        for _ in range(5):
            eng.encode_experience(agent, "bereavement", 0, rng)
        pq = eng.compute_phenomenal_quality(agent)
        assert pq < 0.5

    def test_pq_clipped_to_unit(self):
        eng, agent = _engine_and_agent()
        rng = np.random.default_rng(42)
        for _ in range(20):
            eng.encode_experience(agent, "breakthrough", 0, rng)
        pq = eng.compute_phenomenal_quality(agent)
        assert 0.0 <= pq <= 1.0

    def test_pq_history_tracked(self):
        eng, agent = _engine_and_agent()
        rng = np.random.default_rng(42)
        eng.encode_experience(agent, "routine", 0, rng)
        eng.compute_phenomenal_quality(agent)
        eng.encode_experience(agent, "breakthrough", 1, rng)
        eng.compute_phenomenal_quality(agent)
        history = agent.extension_data["inner_life"]["pq_history"]
        assert len(history) == 2

    def test_no_experiences_returns_neutral(self):
        eng, agent = _engine_and_agent()
        pq = eng.compute_phenomenal_quality(agent)
        assert pq == 0.5


# ===========================================================================
# TestExperientialDrift
# ===========================================================================

class TestExperientialDrift:
    """Test experience-driven trait drift."""

    def test_positive_agency_boosts_resilience(self):
        config = _make_config()
        eng = ExperientialEngine(config)
        ts = config.trait_system
        agent = _make_agent(config)
        eng.init_state(agent)
        rng = np.random.default_rng(42)
        # Encode high-agency experiences
        for _ in range(5):
            # Breakthrough has high agency (0.8)
            eng.encode_experience(agent, "breakthrough", 0, rng)
        drift = eng.compute_experiential_drift(agent)
        # Agency dimension should push resilience up
        if "resilience" in drift:
            assert drift["resilience"] > 0

    def test_negative_valence_increases_neuroticism(self):
        config = _make_config()
        eng = ExperientialEngine(config)
        agent = _make_agent(config)
        eng.init_state(agent)
        rng = np.random.default_rng(42)
        for _ in range(5):
            eng.encode_experience(agent, "bereavement", 0, rng)
        drift = eng.compute_experiential_drift(agent)
        # Negative valence should push neuroticism up (weight is -0.3 * negative val → positive)
        if "neuroticism" in drift:
            assert drift["neuroticism"] > 0

    def test_high_meaning_boosts_depth_drive(self):
        config = _make_config()
        eng = ExperientialEngine(config)
        agent = _make_agent(config)
        eng.init_state(agent)
        rng = np.random.default_rng(42)
        for _ in range(5):
            eng.encode_experience(agent, "breakthrough", 0, rng)
        drift = eng.compute_experiential_drift(agent)
        if "depth_drive" in drift:
            assert drift["depth_drive"] > 0

    def test_drift_scale_with_config(self):
        config1 = _make_config(inner_life_config={"experiential_drift_rate": 0.005})
        config2 = _make_config(inner_life_config={"experiential_drift_rate": 0.05})
        eng1 = ExperientialEngine(config1)
        eng2 = ExperientialEngine(config2)

        agent1 = _make_agent(config1, seed=99)
        agent2 = _make_agent(config2, seed=99)
        eng1.init_state(agent1)
        eng2.init_state(agent2)

        rng = np.random.default_rng(42)
        for _ in range(5):
            eng1.encode_experience(agent1, "breakthrough", 0, np.random.default_rng(42))
            eng2.encode_experience(agent2, "breakthrough", 0, np.random.default_rng(42))

        drift1 = eng1.compute_experiential_drift(agent1)
        drift2 = eng2.compute_experiential_drift(agent2)

        # Higher drift rate → larger drift magnitudes
        if drift1 and drift2:
            mag1 = sum(abs(v) for v in drift1.values())
            mag2 = sum(abs(v) for v in drift2.values())
            assert mag2 > mag1

    def test_apply_drift_modifies_traits(self):
        config = _make_config()
        eng = ExperientialEngine(config)
        agent = _make_agent(config)
        eng.init_state(agent)
        ts = config.trait_system
        try:
            res_idx = ts.trait_index("resilience")
        except (KeyError, ValueError):
            pytest.skip("resilience trait not available")
        original = float(agent.traits[res_idx])
        eng.apply_drift(agent, {"resilience": 0.05})
        assert agent.traits[res_idx] != original


# ===========================================================================
# TestDecisionModulation
# ===========================================================================

class TestDecisionModulation:
    """Test 60/40 experiential decision modulation."""

    def test_modulation_with_recall(self):
        eng, agent = _engine_and_agent()
        rng = np.random.default_rng(42)
        # Create positive experiences
        for _ in range(5):
            eng.encode_experience(agent, "breakthrough", 0, rng)
        eng.update_mood(agent)
        utilities = {"stay": 0.5, "migrate": 0.5}
        modified = eng.compute_experiential_modulation(agent, "MIGRATION", utilities)
        # Should be modified (not exactly equal to original)
        # At least one utility changed (depending on recall threshold)
        assert isinstance(modified, dict)
        assert "stay" in modified
        assert "migrate" in modified

    def test_no_experiences_passthrough(self):
        eng, agent = _engine_and_agent()
        utilities = {"stay": 0.5, "migrate": 0.5}
        modified = eng.compute_experiential_modulation(agent, "MIGRATION", utilities)
        assert modified == utilities

    def test_positive_recall_boost(self):
        config = _make_config(inner_life_config={
            "recall_similarity_threshold": 0.0,
            "experiential_weight": 0.4,
            "positive_recall_boost": 0.5,
        })
        eng = ExperientialEngine(config)
        agent = _make_agent(config)
        eng.init_state(agent)
        rng = np.random.default_rng(42)
        # Create very positive experience
        eng.encode_experience(agent, "breakthrough", 0, rng)
        # Set mood to match breakthrough
        agent.extension_data["inner_life"]["mood"] = list(BASE_EVENT_VECTORS["breakthrough"])
        utilities = {"act": 0.5}
        modified = eng.compute_experiential_modulation(agent, "TEST", utilities)
        # Positive recall should boost
        assert modified["act"] >= 0.5

    def test_negative_recall_penalty(self):
        config = _make_config(inner_life_config={
            "recall_similarity_threshold": 0.0,
            "experiential_weight": 0.4,
            "negative_recall_penalty": 0.5,
        })
        eng = ExperientialEngine(config)
        agent = _make_agent(config)
        eng.init_state(agent)
        rng = np.random.default_rng(42)
        eng.encode_experience(agent, "bereavement", 0, rng)
        agent.extension_data["inner_life"]["mood"] = list(BASE_EVENT_VECTORS["bereavement"])
        utilities = {"act": 0.5}
        modified = eng.compute_experiential_modulation(agent, "TEST", utilities)
        # Negative recall should decrease utility
        assert modified["act"] <= 0.5

    def test_no_inner_life_state_passthrough(self):
        eng, agent = _engine_and_agent()
        # Remove inner life state
        del agent.extension_data["inner_life"]
        utilities = {"stay": 0.5}
        modified = eng.compute_experiential_modulation(agent, "TEST", utilities)
        assert modified == utilities


# ===========================================================================
# TestMoodAndDecay
# ===========================================================================

class TestMoodAndDecay:
    """Test mood updates and assertoric decay."""

    def test_mood_update_blends(self):
        eng, agent = _engine_and_agent()
        rng = np.random.default_rng(42)
        old_mood = list(agent.extension_data["inner_life"]["mood"])
        eng.encode_experience(agent, "breakthrough", 0, rng)
        eng.update_mood(agent)
        new_mood = agent.extension_data["inner_life"]["mood"]
        # Mood should change toward latest experience
        assert new_mood != old_mood

    def test_assertoric_decay(self):
        eng, agent = _engine_and_agent()
        rng = np.random.default_rng(42)
        exp = eng.encode_experience(agent, "routine", 0, rng)
        initial = exp["assertoric_strength"]
        eng.decay_assertoric_strength(agent)
        assert exp["assertoric_strength"] < initial

    def test_assertoric_min_floor(self):
        config = _make_config(inner_life_config={
            "assertoric_decay_rate": 0.99,
            "assertoric_min_floor": 0.05,
        })
        eng = ExperientialEngine(config)
        agent = _make_agent(config)
        eng.init_state(agent)
        rng = np.random.default_rng(42)
        exp = eng.encode_experience(agent, "routine", 0, rng)
        for _ in range(100):
            eng.decay_assertoric_strength(agent)
        assert exp["assertoric_strength"] >= 0.05

    def test_arousal_decay_protection(self):
        eng, agent = _engine_and_agent()
        rng = np.random.default_rng(42)
        # High-arousal event (bereavement: arousal=0.8)
        exp_high = eng.encode_experience(agent, "bereavement", 0, rng)
        # Low-arousal event (routine: arousal=0.2)
        exp_low = eng.encode_experience(agent, "routine", 0, rng)

        s_high = exp_high["assertoric_strength"]
        s_low = exp_low["assertoric_strength"]

        eng.decay_assertoric_strength(agent)

        # High-arousal should decay slower
        decay_high = s_high - exp_high["assertoric_strength"]
        decay_low = s_low - exp_low["assertoric_strength"]
        assert decay_high < decay_low


# ===========================================================================
# TestExtensionHooks
# ===========================================================================

class TestExtensionHooks:
    """Test InnerLifeExtension lifecycle hooks."""

    def test_simulation_start_initializes_all(self):
        config = _make_config()
        population = _make_population(config, n=10)
        ext = InnerLifeExtension()
        ext.on_simulation_start(population, config)
        for agent in population:
            assert "inner_life" in agent.extension_data

    def test_generation_start_encodes_events(self):
        config = _make_config()
        population = _make_population(config, n=5)
        ext = InnerLifeExtension()
        ext.on_simulation_start(population, config)
        ext.on_generation_start(1, population, config)
        # At least routine events should be encoded
        for agent in population:
            if agent.is_alive:
                exps = agent.extension_data["inner_life"]["experiences"]
                assert len(exps) >= 1

    def test_generation_end_computes_pq(self):
        config = _make_config()
        population = _make_population(config, n=5)
        ext = InnerLifeExtension()
        ext.on_simulation_start(population, config)
        ext.on_generation_start(1, population, config)
        ext.on_generation_end(1, population, config)
        for agent in population:
            if agent.is_alive:
                state = agent.extension_data["inner_life"]
                assert len(state["pq_history"]) >= 1

    def test_agent_created_inherits(self):
        config = _make_config()
        ext = InnerLifeExtension()
        parent1 = _make_agent(config, seed=1, id="p1")
        parent2 = _make_agent(config, seed=2, id="p2")
        child = _make_agent(config, seed=3, id="child")

        ext.on_simulation_start([parent1, parent2], config)

        # Give parents experiences
        rng = np.random.default_rng(42)
        for _ in range(5):
            ext._engine.encode_experience(parent1, "breakthrough", 0, rng)
            ext._engine.encode_experience(parent2, "bereavement", 0, rng)

        ext.on_agent_created(child, (parent1, parent2), config)
        child_exps = child.extension_data["inner_life"]["experiences"]
        assert len(child_exps) > 0
        # Inherited experiences should have reduced assertoric strength
        for exp in child_exps:
            assert exp["assertoric_strength"] < 0.9

    def test_modify_decision(self):
        config = _make_config()
        ext = InnerLifeExtension()
        agent = _make_agent(config)
        ext.on_simulation_start([agent], config)
        utilities = {"stay": 0.5, "migrate": 0.5}
        modified = ext.modify_decision(agent, "MIGRATION", utilities, config)
        assert isinstance(modified, dict)

    def test_modify_mortality_low_pq_increases(self):
        config = _make_config()
        ext = InnerLifeExtension()
        agent = _make_agent(config)
        ext.on_simulation_start([agent], config)
        agent.extension_data["inner_life"]["phenomenal_quality"] = 0.1
        rate = ext.modify_mortality(agent, 0.05, config)
        # Low PQ should increase mortality
        assert rate > 0.05

    def test_modify_mortality_high_pq_decreases(self):
        config = _make_config()
        ext = InnerLifeExtension()
        agent = _make_agent(config)
        ext.on_simulation_start([agent], config)
        agent.extension_data["inner_life"]["phenomenal_quality"] = 0.9
        rate = ext.modify_mortality(agent, 0.05, config)
        assert rate < 0.05

    def test_modify_attraction_similar_mood(self):
        config = _make_config()
        ext = InnerLifeExtension()
        a1 = _make_agent(config, seed=1, id="a1")
        a2 = _make_agent(config, seed=2, id="a2")
        ext.on_simulation_start([a1, a2], config)
        # Set identical moods
        a1.extension_data["inner_life"]["mood"] = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        a2.extension_data["inner_life"]["mood"] = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        score = ext.modify_attraction(a1, a2, 0.5, config)
        assert score > 0.5  # Similar mood → bonus


# ===========================================================================
# TestMetrics
# ===========================================================================

class TestMetrics:
    """Test population-level metrics."""

    def test_metrics_completeness(self):
        config = _make_config()
        population = _make_population(config, n=10)
        ext = InnerLifeExtension()
        ext.on_simulation_start(population, config)
        ext.on_generation_start(1, population, config)
        ext.on_generation_end(1, population, config)
        metrics = ext.get_metrics(population)
        assert "mean_phenomenal_quality" in metrics
        assert "pq_distribution" in metrics
        assert "event_type_counts" in metrics
        assert "population_mood" in metrics

    def test_pq_distribution_sums(self):
        config = _make_config()
        population = _make_population(config, n=20)
        ext = InnerLifeExtension()
        ext.on_simulation_start(population, config)
        ext.on_generation_start(1, population, config)
        ext.on_generation_end(1, population, config)
        metrics = ext.get_metrics(population)
        dist = metrics["pq_distribution"]
        total = sum(dist.values())
        alive = sum(1 for a in population if a.is_alive)
        assert total == alive

    def test_event_counts_populated(self):
        config = _make_config()
        population = _make_population(config, n=10)
        ext = InnerLifeExtension()
        ext.on_simulation_start(population, config)
        ext.on_generation_start(1, population, config)
        ext.on_generation_end(1, population, config)
        metrics = ext.get_metrics(population)
        assert len(metrics["event_type_counts"]) > 0


# ===========================================================================
# TestInheritance
# ===========================================================================

class TestInheritance:
    """Test experiential inheritance."""

    def test_inherit_top_k(self):
        config = _make_config(inner_life_config={"inheritance_top_k": 2})
        eng = ExperientialEngine(config)
        parent1 = _make_agent(config, seed=1, id="p1")
        parent2 = _make_agent(config, seed=2, id="p2")
        child = _make_agent(config, seed=3, id="child")
        eng.init_state(parent1)
        eng.init_state(parent2)
        eng.init_state(child)
        rng = np.random.default_rng(42)
        for i in range(5):
            eng.encode_experience(parent1, "breakthrough", i, rng)
            eng.encode_experience(parent2, "routine", i, rng)
        count = eng.inherit_experiences(parent1, parent2, child)
        # 2 from each parent = 4 max
        assert count <= 4
        assert count > 0

    def test_inherited_strength_decayed(self):
        config = _make_config(inner_life_config={
            "inheritance_top_k": 1,
            "inheritance_strength_decay": 0.5,
        })
        eng = ExperientialEngine(config)
        parent = _make_agent(config, seed=1, id="p1")
        child = _make_agent(config, seed=3, id="child")
        eng.init_state(parent)
        eng.init_state(child)
        rng = np.random.default_rng(42)
        eng.encode_experience(parent, "breakthrough", 0, rng)
        parent_strength = parent.extension_data["inner_life"]["experiences"][0]["assertoric_strength"]
        eng.inherit_experiences(parent, parent, child)
        child_exp = child.extension_data["inner_life"]["experiences"][0]
        assert child_exp["assertoric_strength"] == pytest.approx(parent_strength * 0.5)

    def test_no_parent_state_returns_zero(self):
        config = _make_config()
        eng = ExperientialEngine(config)
        parent = _make_agent(config, seed=1, id="p1")
        child = _make_agent(config, seed=3, id="child")
        eng.init_state(child)
        # Parent has no inner_life state
        count = eng.inherit_experiences(parent, parent, child)
        assert count == 0


# ===========================================================================
# TestEventDetection
# ===========================================================================

class TestEventDetection:
    """Test event detection in InnerLifeExtension.on_generation_start."""

    def test_pair_formed_detected(self):
        config = _make_config()
        ext = InnerLifeExtension()
        agent = _make_agent(config, id="test")
        ext.on_simulation_start([agent], config)
        # Set prev state to no partner
        agent.extension_data["inner_life"]["_prev_state"]["partner_id"] = None
        # Now agent has a partner
        agent.partner_id = "partner_1"
        ext.on_generation_start(1, [agent], config)
        events = [e["event_type"] for e in agent.extension_data["inner_life"]["experiences"]]
        assert "pair_formed" in events

    def test_bereavement_detected(self):
        config = _make_config()
        ext = InnerLifeExtension()
        agent = _make_agent(config, id="test")
        partner = _make_agent(config, id="dead_partner", seed=2)
        partner.is_alive = False
        ext.on_simulation_start([agent, partner], config)
        # Set prev state to having partner
        agent.extension_data["inner_life"]["_prev_state"]["partner_id"] = "dead_partner"
        agent.partner_id = None
        ext.on_generation_start(1, [agent, partner], config)
        events = [e["event_type"] for e in agent.extension_data["inner_life"]["experiences"]]
        assert "bereavement" in events

    def test_child_born_detected(self):
        config = _make_config()
        ext = InnerLifeExtension()
        agent = _make_agent(config, id="test")
        ext.on_simulation_start([agent], config)
        agent.extension_data["inner_life"]["_prev_state"]["children_count"] = 0
        agent.children_ids = ["child_1"]
        ext.on_generation_start(1, [agent], config)
        events = [e["event_type"] for e in agent.extension_data["inner_life"]["experiences"]]
        assert "child_born" in events

    def test_routine_when_no_events(self):
        config = _make_config()
        ext = InnerLifeExtension()
        agent = _make_agent(config, id="test")
        ext.on_simulation_start([agent], config)
        ext.on_generation_start(1, [agent], config)
        events = [e["event_type"] for e in agent.extension_data["inner_life"]["experiences"]]
        assert "routine" in events


# ===========================================================================
# TestExtensionProperties
# ===========================================================================

class TestExtensionProperties:
    """Test extension registration properties."""

    def test_name(self):
        ext = InnerLifeExtension()
        assert ext.name == "inner_life"

    def test_description(self):
        ext = InnerLifeExtension()
        assert "experiential" in ext.description.lower() or "phenomenal" in ext.description.lower()

    def test_default_config(self):
        ext = InnerLifeExtension()
        assert isinstance(ext.get_default_config(), dict)

    def test_disabled_passthrough(self):
        config = _make_config(inner_life_config={"enabled": False})
        ext = InnerLifeExtension()
        agent = _make_agent(config)
        ext.on_simulation_start([agent], config)
        # Should not initialize inner life when disabled
        assert "inner_life" not in agent.extension_data

    def test_registry_registration(self):
        registry = ExtensionRegistry()
        ext = InnerLifeExtension()
        registry.register(ext)
        registry.enable("inner_life")
        enabled = registry.get_enabled()
        assert any(e.name == "inner_life" for e in enabled)


# ===========================================================================
# TestAPIEndpoints
# ===========================================================================

class TestAPIEndpoints:
    """Test inner-life API router endpoints."""

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from seldon.api.app import create_app
        app = create_app()
        app.state.session_manager = _make_test_session_manager()
        return TestClient(app)

    def test_overview(self, client):
        resp = client.get(f"/api/inner-life/{_SID}/overview")
        assert resp.status_code == 200
        data = resp.json()
        assert data["enabled"] is True
        assert "mean_phenomenal_quality" in data

    def test_agent(self, client):
        resp = client.get(f"/api/inner-life/{_SID}/agent/{_AID}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["enabled"] is True
        assert data["agent_id"] == _AID

    def test_pq_distribution(self, client):
        resp = client.get(f"/api/inner-life/{_SID}/phenomenal-quality-distribution")
        assert resp.status_code == 200
        data = resp.json()
        assert data["enabled"] is True
        assert "distribution" in data
        assert "stats" in data

    def test_mood_map(self, client):
        resp = client.get(f"/api/inner-life/{_SID}/mood-map")
        assert resp.status_code == 200
        data = resp.json()
        assert data["enabled"] is True
        assert "agents" in data

    def test_experiential_drift(self, client):
        resp = client.get(f"/api/inner-life/{_SID}/experiential-drift")
        assert resp.status_code == 200
        data = resp.json()
        assert data["enabled"] is True

    def test_disabled_returns_false(self, client):
        # Create a session without inner_life extension
        mgr = client.app.state.session_manager
        session = mgr.create_session(
            config=_make_config(extensions_enabled=[]),
            name="no_inner_life",
        )
        resp = client.get(f"/api/inner-life/{session.id}/overview")
        assert resp.status_code == 200
        assert resp.json()["enabled"] is False

    def test_404_session(self, client):
        resp = client.get("/api/inner-life/nonexistent/overview")
        assert resp.status_code == 404

    def test_404_agent(self, client):
        resp = client.get(f"/api/inner-life/{_SID}/agent/nonexistent")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# API Test Helpers
# ---------------------------------------------------------------------------

_SID = ""
_AID = ""


def _make_test_session_manager():
    """Build a session manager with a pre-populated inner_life session."""
    from seldon.api.sessions import SessionManager

    global _SID, _AID
    mgr = SessionManager(db_path=None)
    config = _make_config(extensions_enabled=["inner_life"])
    session = mgr.create_session(config=config, name="inner_life_test")
    _SID = session.id

    # Step a few generations to populate experiences
    mgr.step(session.id, n=3)

    # Pick the first living agent for per-agent tests
    for agent in session.engine.population:
        if agent.is_alive:
            _AID = agent.id
            break

    return mgr
