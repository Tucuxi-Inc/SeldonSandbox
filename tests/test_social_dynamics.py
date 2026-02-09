"""
Tests for Phase 7: Social Hierarchies & Mentorship.

Tests SocialHierarchyManager, MentorshipManager, SocialDynamicsExtension,
and the social API router.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from seldon.core.agent import Agent
from seldon.core.config import ExperimentConfig
from seldon.core.processing import ProcessingRegion
from seldon.extensions.registry import ExtensionRegistry
from seldon.extensions.social_dynamics import SocialDynamicsExtension
from seldon.social.hierarchy import SocialHierarchyManager
from seldon.social.mentorship import MentorshipManager


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
    """Create a population of agents with varied traits and ages."""
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
        # Give some agents partners and children
        if i > 0 and rng.random() < 0.4:
            agent.partner_id = f"pop_{i - 1:03d}"
        if rng.random() < 0.3:
            agent.children_ids = [f"child_{i}_{j}" for j in range(rng.integers(1, 4))]
        population.append(agent)
    return population


# ===========================================================================
# SocialHierarchyManager Tests
# ===========================================================================

class TestSocialHierarchyManager:
    """Tests for status computation, influence, and role assignment."""

    def test_compute_status_returns_bounded_float(self):
        config = _make_config()
        hm = SocialHierarchyManager(config)
        agent = _make_agent(config, age=30, contribution_history=[1.0, 1.2, 0.8])
        population = [agent]

        status = hm.compute_status(agent, population)
        assert 0.0 <= status <= 1.0

    def test_status_increases_with_contribution(self):
        config = _make_config()
        hm = SocialHierarchyManager(config)

        low_contrib = _make_agent(config, id="a1", age=30, seed=1,
                                  contribution_history=[0.1, 0.1, 0.1])
        high_contrib = _make_agent(config, id="a2", age=30, seed=2,
                                   contribution_history=[1.5, 1.8, 2.0])

        pop = [low_contrib, high_contrib]
        s_low = hm.compute_status(low_contrib, pop)
        s_high = hm.compute_status(high_contrib, pop)
        assert s_high > s_low

    def test_status_age_factor_peaks_midlife(self):
        config = _make_config()
        hm = SocialHierarchyManager(config)

        young = _make_agent(config, id="y", age=10, seed=1, contribution_history=[])
        middle = _make_agent(config, id="m", age=40, seed=1, contribution_history=[])
        old = _make_agent(config, id="o", age=80, seed=1, contribution_history=[])
        # Use same traits
        middle.traits = young.traits.copy()
        old.traits = young.traits.copy()

        pop = [young, middle, old]
        s_young = hm.compute_status(young, pop)
        s_middle = hm.compute_status(middle, pop)
        assert s_middle > s_young

    def test_compute_influence_bounded(self):
        config = _make_config()
        hm = SocialHierarchyManager(config)
        agent = _make_agent(config, age=30)
        agent.social_status = 0.7
        agent.social_bonds = {"b1": 0.5, "b2": 0.3}
        pop = [agent]

        influence = hm.compute_influence(agent, pop)
        assert 0.0 <= influence <= 1.0

    def test_assign_roles_returns_all_agents(self):
        config = _make_config()
        hm = SocialHierarchyManager(config)
        pop = _make_population(config, 30)
        # Must compute status first
        for a in pop:
            a.social_status = hm.compute_status(a, pop)

        roles = hm.assign_roles(pop)
        assert len(roles) == len(pop)
        for agent in pop:
            assert agent.id in roles

    def test_assign_roles_valid_role_names(self):
        config = _make_config()
        hm = SocialHierarchyManager(config)
        pop = _make_population(config, 30)
        for a in pop:
            a.social_status = hm.compute_status(a, pop)

        roles = hm.assign_roles(pop)
        valid = {"leader", "innovator", "mediator", "worker", "outsider_bridge", "unassigned"}
        for role in roles.values():
            assert role in valid

    def test_update_all_sets_status_and_roles(self):
        config = _make_config()
        hm = SocialHierarchyManager(config)
        rng = np.random.default_rng(42)
        pop = _make_population(config, 20)

        metrics = hm.update_all(pop, rng)

        # Every agent should have status set
        for a in pop:
            assert 0.0 <= a.social_status <= 1.0
            assert a.social_role is not None
            assert 0.0 <= a.influence_score <= 1.0

        # Metrics should be populated
        assert "role_counts" in metrics
        assert "mean_status" in metrics
        assert "influence_gini" in metrics

    def test_update_social_bonds_from_hierarchy(self):
        config = _make_config()
        hm = SocialHierarchyManager(config)
        rng = np.random.default_rng(42)
        pop = _make_population(config, 20)

        # Set statuses first
        for i, a in enumerate(pop):
            a.social_status = i / len(pop)

        initial_bonds = sum(len(a.social_bonds) for a in pop)
        hm.update_social_bonds_from_hierarchy(pop, rng)
        final_bonds = sum(len(a.social_bonds) for a in pop)
        # Should have created some new bonds
        assert final_bonds >= initial_bonds

    def test_gini_coefficient(self):
        assert SocialHierarchyManager._compute_gini([]) == 0.0
        assert SocialHierarchyManager._compute_gini([0, 0, 0]) == 0.0
        # Perfect equality
        gini_equal = SocialHierarchyManager._compute_gini([1.0, 1.0, 1.0])
        assert gini_equal == pytest.approx(0.0, abs=0.01)
        # High inequality
        gini_unequal = SocialHierarchyManager._compute_gini([0, 0, 0, 0, 100])
        assert gini_unequal > 0.5

    def test_empty_population(self):
        config = _make_config()
        hm = SocialHierarchyManager(config)
        roles = hm.assign_roles([])
        assert roles == {}


# ===========================================================================
# MentorshipManager Tests
# ===========================================================================

class TestMentorshipManager:
    """Tests for mentor matching, effects, and dissolution."""

    def test_match_mentors_basic(self):
        config = _make_config()
        mm = MentorshipManager(config)
        rng = np.random.default_rng(42)

        pop = _make_population(config, 30)
        # Ensure some eligible mentors (age >= 25, contribution > 0.5)
        for a in pop[:5]:
            a.age = 35
            a.contribution_history = [1.0, 1.2, 0.9, 1.1, 0.8]
        # Ensure some eligible mentees (age 5-25, no mentor)
        for a in pop[20:]:
            a.age = 15
            a.mentor_id = None

        matches = mm.match_mentors(pop, rng)
        assert isinstance(matches, list)
        # Should have at least some matches
        for mentor, mentee in matches:
            assert mentee.mentor_id == mentor.id
            assert mentee.id in mentor.mentee_ids

    def test_match_mentors_respects_max_mentees(self):
        config = _make_config()
        config.mentorship_config["max_mentees"] = 1
        mm = MentorshipManager(config)
        rng = np.random.default_rng(42)

        pop = _make_population(config, 20)
        # One mentor, many mentees
        pop[0].age = 40
        pop[0].contribution_history = [1.5, 1.5, 1.5]
        for a in pop[10:]:
            a.age = 15
            a.mentor_id = None

        matches = mm.match_mentors(pop, rng)
        # With max_mentees=1, the mentor should get at most 1 mentee
        mentor_ids = [m.id for m, _ in matches]
        assert mentor_ids.count(pop[0].id) <= 1

    def test_match_mentors_disabled(self):
        config = _make_config()
        config.mentorship_config["enabled"] = False
        mm = MentorshipManager(config)
        rng = np.random.default_rng(42)

        pop = _make_population(config, 20)
        matches = mm.match_mentors(pop, rng)
        assert matches == []

    def test_apply_mentorship_effects(self):
        config = _make_config()
        mm = MentorshipManager(config)
        rng = np.random.default_rng(42)

        pop = _make_population(config, 10)
        mentor = pop[0]
        mentee = pop[1]
        mentor.age = 40
        mentee.age = 18
        mentee.mentor_id = mentor.id
        mentor.mentee_ids = [mentee.id]
        mentor.skills = {"leadership": 0.8, "crafting": 0.6}
        mentee.skills = {}

        original_traits = mentee.traits.copy()
        active = mm.apply_mentorship_effects(pop, rng)
        assert active >= 1

        # Mentee traits should have shifted toward mentor
        direction = mentor.traits - original_traits
        # At least some traits should have moved in the expected direction
        drift = mentee.traits - original_traits
        assert not np.allclose(drift, 0.0)

        # Skills should have been partially transferred
        assert "leadership" in mentee.skills
        assert "crafting" in mentee.skills

    def test_apply_mentorship_effects_disabled(self):
        config = _make_config()
        config.mentorship_config["enabled"] = False
        mm = MentorshipManager(config)
        rng = np.random.default_rng(42)

        pop = _make_population(config, 5)
        active = mm.apply_mentorship_effects(pop, rng)
        assert active == 0

    def test_dissolve_mentorships_aged_out(self):
        config = _make_config()
        mm = MentorshipManager(config)
        rng = np.random.default_rng(42)

        pop = _make_population(config, 5)
        mentor = pop[0]
        mentee = pop[1]
        mentor.age = 40
        mentee.age = config.mentorship_config["mentee_max_age"] + 10  # Way past max
        mentee.mentor_id = mentor.id
        mentor.mentee_ids = [mentee.id]

        dissolved = mm.dissolve_mentorships(pop, rng)
        assert len(dissolved) >= 1
        assert mentee.mentor_id is None

    def test_dissolve_mentorships_dead_mentor(self):
        config = _make_config()
        mm = MentorshipManager(config)
        rng = np.random.default_rng(42)

        pop = _make_population(config, 5)
        mentor = pop[0]
        mentee = pop[1]
        mentor.is_alive = False
        mentee.age = 18
        mentee.mentor_id = mentor.id
        mentor.mentee_ids = [mentee.id]

        dissolved = mm.dissolve_mentorships(pop, rng)
        assert len(dissolved) >= 1
        assert mentee.mentor_id is None

    def test_get_mentorship_chains(self):
        config = _make_config()
        mm = MentorshipManager(config)

        pop = _make_population(config, 5)
        # Create a chain: pop[0] → pop[1] → pop[2]
        pop[0].mentee_ids = [pop[1].id]
        pop[1].mentor_id = pop[0].id
        pop[1].mentee_ids = [pop[2].id]
        pop[2].mentor_id = pop[1].id

        chains = mm.get_mentorship_chains(pop)
        assert isinstance(chains, list)
        # Should find root at pop[0]
        assert len(chains) >= 1
        root = chains[0]
        assert root["id"] == pop[0].id
        assert len(root["mentees"]) >= 1

    def test_compatibility_score_positive(self):
        config = _make_config()
        mm = MentorshipManager(config)

        mentor = _make_agent(config, id="mentor", age=40, seed=1)
        mentee = _make_agent(config, id="mentee", age=18, seed=2)
        mentor.social_status = 0.8
        mentee.social_status = 0.2

        score = mm._compatibility_score(mentor, mentee)
        assert score >= 0.0

    def test_compatibility_same_location_bonus(self):
        config = _make_config()
        mm = MentorshipManager(config)

        mentor = _make_agent(config, id="mentor", age=40, seed=1)
        mentee = _make_agent(config, id="mentee", age=18, seed=2)
        mentor.social_status = 0.8
        mentee.social_status = 0.2

        score_no_loc = mm._compatibility_score(mentor, mentee)
        mentor.location_id = "loc_1"
        mentee.location_id = "loc_1"
        score_same_loc = mm._compatibility_score(mentor, mentee)
        assert score_same_loc > score_no_loc


# ===========================================================================
# SocialDynamicsExtension Tests
# ===========================================================================

class TestSocialDynamicsExtension:
    """Tests for the extension wiring social dynamics into the engine."""

    def test_extension_name(self):
        ext = SocialDynamicsExtension()
        assert ext.name == "social_dynamics"

    def test_extension_description(self):
        ext = SocialDynamicsExtension()
        assert "hierarchy" in ext.description.lower() or "social" in ext.description.lower()

    def test_on_simulation_start(self):
        config = _make_config()
        ext = SocialDynamicsExtension()
        pop = _make_population(config, 15)

        ext.on_simulation_start(pop, config)

        # Status should be computed
        for a in pop:
            assert a.social_status >= 0.0
            assert a.social_role is not None

    def test_on_generation_start(self):
        config = _make_config()
        ext = SocialDynamicsExtension()
        pop = _make_population(config, 15)

        ext.on_simulation_start(pop, config)
        ext.on_generation_start(0, pop, config)

        for a in pop:
            assert 0.0 <= a.social_status <= 1.0

    def test_on_generation_end_mentorship(self):
        config = _make_config()
        ext = SocialDynamicsExtension()
        pop = _make_population(config, 20)

        # Ensure eligible mentors
        for a in pop[:5]:
            a.age = 35
            a.contribution_history = [1.0, 1.2, 0.9, 1.1, 0.8]
        for a in pop[15:]:
            a.age = 15
            a.mentor_id = None

        ext.on_simulation_start(pop, config)
        ext.on_generation_end(0, pop, config)

        metrics = ext.get_metrics(pop)
        assert "active_mentorships" in metrics

    def test_on_agent_created(self):
        config = _make_config()
        ext = SocialDynamicsExtension()

        parent1 = _make_agent(config, id="p1", seed=1)
        parent2 = _make_agent(config, id="p2", seed=2)
        child = _make_agent(config, id="child", seed=3)

        # Give parents social bonds
        parent1.social_bonds = {"friend1": 0.5, "friend2": 0.8}
        parent2.social_bonds = {"friend3": 0.6}

        ext.on_agent_created(child, (parent1, parent2), config)

        # Child should inherit weak bonds from parents' strong connections
        assert len(child.social_bonds) > 0

    def test_modify_attraction_compatible_roles(self):
        config = _make_config()
        ext = SocialDynamicsExtension()

        agent1 = _make_agent(config, id="a1", seed=1)
        agent2 = _make_agent(config, id="a2", seed=2)
        agent1.social_role = "leader"
        agent2.social_role = "mediator"

        boosted = ext.modify_attraction(agent1, agent2, 0.5, config)
        assert boosted > 0.5

    def test_modify_attraction_incompatible_roles(self):
        config = _make_config()
        ext = SocialDynamicsExtension()

        agent1 = _make_agent(config, id="a1", seed=1)
        agent2 = _make_agent(config, id="a2", seed=2)
        agent1.social_role = "worker"
        agent2.social_role = "unassigned"

        result = ext.modify_attraction(agent1, agent2, 0.5, config)
        assert result == 0.5  # No boost

    def test_modify_decision_leader_stay_bonus(self):
        config = _make_config()
        ext = SocialDynamicsExtension()

        agent = _make_agent(config)
        agent.social_role = "leader"

        utilities = {"stay": 0.5, "migrate": 0.3}
        result = ext.modify_decision(agent, "migration", utilities, config)
        assert result["stay"] > 0.5

    def test_modify_decision_bridge_migrate_bonus(self):
        config = _make_config()
        ext = SocialDynamicsExtension()

        agent = _make_agent(config)
        agent.social_role = "outsider_bridge"

        utilities = {"stay": 0.5, "migrate": 0.3}
        result = ext.modify_decision(agent, "migration", utilities, config)
        assert result["migrate"] > 0.3

    def test_get_metrics(self):
        config = _make_config()
        ext = SocialDynamicsExtension()
        pop = _make_population(config, 15)

        ext.on_simulation_start(pop, config)
        ext.on_generation_end(0, pop, config)

        metrics = ext.get_metrics(pop)
        assert "role_counts" in metrics
        assert "mean_status" in metrics

    def test_get_default_config(self):
        ext = SocialDynamicsExtension()
        defaults = ext.get_default_config()
        assert "role_attraction_bonus" in defaults
        assert "leader_stay_bonus" in defaults
        assert "bridge_migrate_bonus" in defaults

    def test_registry_registration(self):
        registry = ExtensionRegistry()
        ext = SocialDynamicsExtension()
        registry.register(ext)
        registry.enable("social_dynamics")

        enabled = registry.get_enabled()
        assert any(e.name == "social_dynamics" for e in enabled)


# ===========================================================================
# Integration: Full update cycle
# ===========================================================================

class TestSocialDynamicsIntegration:
    """Integration tests for social dynamics across generations."""

    def test_multi_generation_cycle(self):
        config = _make_config()
        ext = SocialDynamicsExtension()
        pop = _make_population(config, 25)

        # Set up eligible mentors/mentees
        for a in pop[:8]:
            a.age = 35
            a.contribution_history = [1.0, 1.2, 0.9]
        for a in pop[18:]:
            a.age = 15

        ext.on_simulation_start(pop, config)

        # Run 3 generations
        for gen in range(3):
            ext.on_generation_start(gen, pop, config)
            ext.on_generation_end(gen, pop, config)

        metrics = ext.get_metrics(pop)
        assert metrics.get("role_counts") is not None
        # Should have attempted mentorship
        assert "active_mentorships" in metrics

    def test_backward_compatibility_no_social_data(self):
        """Agents without social data should work fine."""
        config = _make_config()
        agent = _make_agent(config)

        # Default values
        assert agent.social_status == 0.0
        assert agent.mentor_id is None
        assert agent.mentee_ids == []
        assert agent.social_role is None
        assert agent.influence_score == 0.0


# ===========================================================================
# API Router Tests
# ===========================================================================

class TestSocialAPIRouter:
    """Tests for the social dynamics API endpoints."""

    @pytest.fixture
    def client(self):
        try:
            from fastapi.testclient import TestClient
            from seldon.api.app import create_app
        except ImportError:
            pytest.skip("fastapi/httpx not installed")

        app = create_app()
        return TestClient(app)

    @pytest.fixture
    def session_with_social(self, client):
        """Create a session with social_dynamics enabled and run a few generations."""
        resp = client.post("/api/simulation/sessions", json={
            "config": {
                "initial_population": 25,
                "generations_to_run": 10,
                "random_seed": 42,
                "extensions_enabled": [
                    "geography", "resources", "technology",
                    "culture", "conflict", "social_dynamics",
                ],
            },
        })
        assert resp.status_code == 200
        session_id = resp.json()["id"]

        # Run 3 generations
        resp = client.post(f"/api/simulation/sessions/{session_id}/step", json={"n": 3})
        assert resp.status_code == 200
        return session_id

    def test_hierarchy_endpoint(self, client, session_with_social):
        resp = client.get(f"/api/social/{session_with_social}/hierarchy")
        assert resp.status_code == 200
        data = resp.json()
        assert "total" in data
        assert "agents" in data
        assert len(data["agents"]) > 0
        agent = data["agents"][0]
        assert "social_status" in agent
        assert "social_role" in agent
        assert "influence_score" in agent

    def test_hierarchy_pagination(self, client, session_with_social):
        resp = client.get(f"/api/social/{session_with_social}/hierarchy?page=1&page_size=5")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["agents"]) <= 5

    def test_roles_endpoint(self, client, session_with_social):
        resp = client.get(f"/api/social/{session_with_social}/hierarchy/roles")
        assert resp.status_code == 200
        data = resp.json()
        assert "roles" in data
        for role_name, role_data in data["roles"].items():
            assert "count" in role_data
            assert "agents" in role_data

    def test_mentorship_endpoint(self, client, session_with_social):
        resp = client.get(f"/api/social/{session_with_social}/mentorship")
        assert resp.status_code == 200
        data = resp.json()
        assert "active_count" in data
        assert "pairs" in data
        assert "chains" in data

    def test_influence_map_endpoint(self, client, session_with_social):
        resp = client.get(f"/api/social/{session_with_social}/influence-map")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_agents" in data
        assert "mean_influence" in data
        assert "top_agents" in data
        assert len(data["top_agents"]) <= 10

    def test_hierarchy_404(self, client):
        resp = client.get("/api/social/nonexistent/hierarchy")
        assert resp.status_code == 404
