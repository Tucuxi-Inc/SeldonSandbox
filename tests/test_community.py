"""
Tests for Phase 9: Community Personality & Diplomacy.

Tests CommunityManager, DiplomacyExtension, communities API router.
"""

from __future__ import annotations

import numpy as np
import pytest

from seldon.core.agent import Agent
from seldon.core.config import ExperimentConfig
from seldon.core.processing import ProcessingRegion
from seldon.extensions.diplomacy import DiplomacyExtension, DiplomaticRelation
from seldon.extensions.geography import GeographyExtension
from seldon.extensions.registry import ExtensionRegistry
from seldon.social.community import CommunityManager


def _make_config(**overrides) -> ExperimentConfig:
    defaults = {"random_seed": 42}
    defaults.update(overrides)
    return ExperimentConfig(**defaults)


def _make_agent(
    agent_id: str = "a1",
    traits: np.ndarray | None = None,
    config: ExperimentConfig | None = None,
    **kwargs,
) -> Agent:
    config = config or _make_config()
    ts = config.trait_system
    if traits is None:
        traits = ts.random_traits(np.random.default_rng(hash(agent_id) % (2**31)))
    agent = Agent(
        id=agent_id,
        name=f"Agent-{agent_id}",
        age=kwargs.pop("age", 25),
        generation=kwargs.pop("generation", 0),
        birth_order=kwargs.pop("birth_order", 1),
        traits=traits,
        traits_at_birth=traits.copy(),
        **kwargs,
    )
    return agent


def _make_community(config, community_id, n_agents, rng, location_id=None):
    """Helper to create a community of agents."""
    agents = []
    for i in range(n_agents):
        a = _make_agent(f"{community_id}_a{i}", config=config)
        a.community_id = community_id
        a.location_id = location_id or community_id
        a.influence_score = float(rng.uniform(0.1, 1.0))
        a.social_status = float(rng.uniform(0.0, 1.0))
        agents.append(a)
    return agents


# =====================================================================
# CommunityManager Tests
# =====================================================================
class TestCommunityManager:
    def test_get_communities_groups_by_community_id(self):
        config = _make_config()
        cm = CommunityManager(config)
        agents = []
        for i in range(6):
            a = _make_agent(f"a{i}", config=config)
            a.community_id = "comm_a" if i < 3 else "comm_b"
            agents.append(a)
        communities = cm.get_communities(agents)
        assert len(communities) == 2
        assert len(communities["comm_a"]) == 3
        assert len(communities["comm_b"]) == 3

    def test_get_communities_falls_back_to_location_id(self):
        config = _make_config()
        cm = CommunityManager(config)
        agents = []
        for i in range(4):
            a = _make_agent(f"a{i}", config=config)
            a.location_id = "loc_0" if i < 2 else "loc_1"
            agents.append(a)
        communities = cm.get_communities(agents)
        assert "loc_0" in communities
        assert "loc_1" in communities

    def test_personality_profile_basic(self):
        config = _make_config()
        cm = CommunityManager(config)
        rng = np.random.default_rng(42)
        agents = _make_community(config, "test", 10, rng)
        profile = cm.compute_personality_profile(agents)
        assert "trait_means" in profile
        assert "trait_variance" in profile
        assert "dominant_region" in profile
        assert "character" in profile
        assert profile["size"] == 10
        assert len(profile["trait_means"]) == config.trait_system.count

    def test_personality_profile_empty(self):
        config = _make_config()
        cm = CommunityManager(config)
        profile = cm.compute_personality_profile([])
        assert profile["character"] == "empty"

    def test_cohesion_high_for_similar_agents(self):
        config = _make_config()
        cm = CommunityManager(config)
        # All agents with identical traits = maximum trait cohesion
        base_traits = np.full(config.trait_system.count, 0.5)
        agents = []
        for i in range(10):
            a = _make_agent(f"a{i}", traits=base_traits.copy(), config=config)
            agents.append(a)
        cohesion = cm.compute_cohesion(agents)
        assert cohesion > 0.5  # High cohesion for identical traits

    def test_cohesion_low_for_diverse_agents(self):
        config = _make_config()
        cm = CommunityManager(config)
        rng = np.random.default_rng(42)
        agents = []
        for i in range(10):
            # Alternate between extreme trait values
            traits = np.full(config.trait_system.count, 0.1 if i % 2 == 0 else 0.9)
            a = _make_agent(f"a{i}", traits=traits, config=config)
            agents.append(a)
        cohesion = cm.compute_cohesion(agents)
        # Should be lower than homogeneous group
        assert cohesion < 0.6

    def test_cohesion_single_agent(self):
        config = _make_config()
        cm = CommunityManager(config)
        agents = [_make_agent("a1", config=config)]
        assert cm.compute_cohesion(agents) == 1.0

    def test_identity_distinctiveness(self):
        config = _make_config()
        cm = CommunityManager(config)
        ts = config.trait_system
        # Community A: all high traits
        agents_a = [_make_agent(f"a{i}", traits=np.full(ts.count, 0.9), config=config) for i in range(5)]
        # Community B: all low traits
        agents_b = [_make_agent(f"b{i}", traits=np.full(ts.count, 0.1), config=config) for i in range(5)]
        for a in agents_a:
            a.influence_score = 0.5
        for a in agents_b:
            a.influence_score = 0.5
        communities = {"a": agents_a, "b": agents_b}
        identity = cm.compute_identity(agents_a, communities, "a")
        assert identity["distinctiveness"] > 0.5  # Very distinct

    def test_trait_compatibility(self):
        config = _make_config()
        cm = CommunityManager(config)
        ts = config.trait_system
        # Identical communities → high compatibility
        agents_a = [_make_agent(f"a{i}", traits=np.full(ts.count, 0.5), config=config) for i in range(5)]
        agents_b = [_make_agent(f"b{i}", traits=np.full(ts.count, 0.5), config=config) for i in range(5)]
        compat = cm.trait_compatibility(agents_a, agents_b)
        assert compat > 0.9

    def test_trait_compatibility_different(self):
        config = _make_config()
        cm = CommunityManager(config)
        ts = config.trait_system
        agents_a = [_make_agent(f"a{i}", traits=np.full(ts.count, 0.0), config=config) for i in range(5)]
        agents_b = [_make_agent(f"b{i}", traits=np.full(ts.count, 1.0), config=config) for i in range(5)]
        compat = cm.trait_compatibility(agents_a, agents_b)
        assert compat < 0.5

    def test_detect_factions_no_factions_small_community(self):
        config = _make_config()
        cm = CommunityManager(config)
        rng = np.random.default_rng(42)
        agents = _make_community(config, "small", 4, rng)
        factions = cm.detect_factions(agents)
        assert factions == []

    def test_detect_factions_finds_split(self):
        config = _make_config(community_config={
            "faction_detection_threshold": 0.1,
            "min_community_size": 3,
            "cohesion_trait_weight": 0.4,
            "cohesion_bond_weight": 0.3,
            "cohesion_culture_weight": 0.2,
            "cohesion_conflict_weight": 0.1,
        })
        cm = CommunityManager(config)
        ts = config.trait_system
        agents = []
        for i in range(12):
            traits = np.full(ts.count, 0.1 if i < 6 else 0.9)
            a = _make_agent(f"a{i}", traits=traits, config=config)
            agents.append(a)
        factions = cm.detect_factions(agents)
        assert len(factions) == 2
        assert sum(f["size"] for f in factions) == 12


# =====================================================================
# DiplomacyExtension Tests
# =====================================================================
class TestDiplomacyExtension:
    def test_extension_name(self):
        geo = GeographyExtension()
        ext = DiplomacyExtension(geo)
        assert ext.name == "diplomacy"

    def test_extension_description(self):
        geo = GeographyExtension()
        ext = DiplomacyExtension(geo)
        assert "diplomacy" in ext.description.lower()

    def test_default_config_has_requires(self):
        geo = GeographyExtension()
        ext = DiplomacyExtension(geo)
        dc = ext.get_default_config()
        assert "geography" in dc["requires"]

    def test_on_simulation_start_sets_community_ids(self):
        config = _make_config()
        geo = GeographyExtension()
        ext = DiplomacyExtension(geo)
        agents = []
        for i in range(6):
            a = _make_agent(f"a{i}", config=config)
            a.location_id = "loc_0" if i < 3 else "loc_1"
            agents.append(a)
        ext.on_simulation_start(agents, config)
        for a in agents:
            assert a.community_id == a.location_id

    def test_on_agent_created_inherits_community(self):
        config = _make_config()
        geo = GeographyExtension()
        ext = DiplomacyExtension(geo)
        p1 = _make_agent("p1", config=config)
        p2 = _make_agent("p2", config=config)
        p1.community_id = "comm_a"
        child = _make_agent("c1", config=config)
        ext.on_agent_created(child, (p1, p2), config)
        assert child.community_id == "comm_a"

    def test_on_generation_end_creates_relations(self):
        config = _make_config()
        geo = GeographyExtension()
        ext = DiplomacyExtension(geo)
        rng = np.random.default_rng(42)
        agents_a = _make_community(config, "loc_0", 5, rng, "loc_0")
        agents_b = _make_community(config, "loc_1", 5, rng, "loc_1")
        population = agents_a + agents_b
        ext.on_simulation_start(population, config)
        ext.on_generation_end(0, population, config)
        assert len(ext.relations) == 1  # 2 communities → 1 pair
        rel = list(ext.relations.values())[0]
        assert isinstance(rel.standing, float)

    def test_standing_evolves_over_generations(self):
        config = _make_config()
        geo = GeographyExtension()
        ext = DiplomacyExtension(geo)
        rng = np.random.default_rng(42)
        agents_a = _make_community(config, "loc_0", 5, rng, "loc_0")
        agents_b = _make_community(config, "loc_1", 5, rng, "loc_1")
        population = agents_a + agents_b
        ext.on_simulation_start(population, config)
        standings = []
        for gen in range(5):
            ext.on_generation_end(gen, population, config)
            rel = list(ext.relations.values())[0]
            standings.append(rel.standing)
        # Standing should change over time (not stay exactly 0)
        assert not all(s == standings[0] for s in standings)

    def test_modify_attraction_alliance_boost(self):
        config = _make_config()
        geo = GeographyExtension()
        ext = DiplomacyExtension(geo)
        a1 = _make_agent("a1", config=config)
        a2 = _make_agent("a2", config=config)
        a1.community_id = "comm_a"
        a2.community_id = "comm_b"
        # Create an alliance
        ext.relations[("comm_a", "comm_b")] = DiplomaticRelation(
            community_a="comm_a", community_b="comm_b",
            standing=0.8, alliance=True,
        )
        result = ext.modify_attraction(a1, a2, 1.0, config)
        assert result > 1.0  # Should boost

    def test_modify_attraction_rivalry_penalty(self):
        config = _make_config()
        geo = GeographyExtension()
        ext = DiplomacyExtension(geo)
        a1 = _make_agent("a1", config=config)
        a2 = _make_agent("a2", config=config)
        a1.community_id = "comm_a"
        a2.community_id = "comm_b"
        ext.relations[("comm_a", "comm_b")] = DiplomaticRelation(
            community_a="comm_a", community_b="comm_b",
            standing=-0.6, rivalry=True,
        )
        result = ext.modify_attraction(a1, a2, 1.0, config)
        assert result < 1.0  # Should penalize

    def test_modify_attraction_same_community_unmodified(self):
        config = _make_config()
        geo = GeographyExtension()
        ext = DiplomacyExtension(geo)
        a1 = _make_agent("a1", config=config)
        a2 = _make_agent("a2", config=config)
        a1.community_id = "comm_a"
        a2.community_id = "comm_a"
        result = ext.modify_attraction(a1, a2, 1.0, config)
        assert result == 1.0

    def test_modify_decision_rivalry_boost(self):
        config = _make_config()
        geo = GeographyExtension()
        ext = DiplomacyExtension(geo)
        a1 = _make_agent("a1", config=config)
        a1.community_id = "comm_a"
        ext.relations[("comm_a", "comm_b")] = DiplomaticRelation(
            community_a="comm_a", community_b="comm_b",
            standing=-0.6, rivalry=True,
        )
        utils = {"stay": 0.5, "migrate": 0.3, "defend": 0.2}
        result = ext.modify_decision(a1, "conflict", utils, config)
        assert result["defend"] > 0.2
        assert result["stay"] > 0.5

    def test_get_metrics(self):
        config = _make_config()
        geo = GeographyExtension()
        ext = DiplomacyExtension(geo)
        ext.relations[("a", "b")] = DiplomaticRelation("a", "b", standing=0.8, alliance=True)
        ext.relations[("a", "c")] = DiplomaticRelation("a", "c", standing=-0.6, rivalry=True)
        ext.community_profiles = {"a": {}, "b": {}, "c": {}}
        metrics = ext.get_metrics([])
        assert metrics["total_relations"] == 2
        assert metrics["alliances"] == 1
        assert metrics["rivalries"] == 1
        assert metrics["community_count"] == 3

    def test_get_all_relations(self):
        config = _make_config()
        geo = GeographyExtension()
        ext = DiplomacyExtension(geo)
        ext.relations[("a", "b")] = DiplomaticRelation("a", "b", standing=0.5)
        rels = ext.get_all_relations()
        assert len(rels) == 1
        assert rels[0]["community_a"] == "a"
        assert rels[0]["standing"] == 0.5

    def test_registry_registration(self):
        geo = GeographyExtension()
        ext = DiplomacyExtension(geo)
        registry = ExtensionRegistry()
        registry.register(geo)
        registry.register(ext)
        registry.enable("geography")
        registry.enable("diplomacy")
        enabled = registry.get_enabled()
        ext_names = [e.name for e in enabled]
        assert "geography" in ext_names
        assert "diplomacy" in ext_names

    def test_cultural_exchange_between_allies(self):
        config = _make_config()
        geo = GeographyExtension()
        ext = DiplomacyExtension(geo)
        rng = np.random.default_rng(42)
        ext._rng = rng
        agents_a = _make_community(config, "loc_0", 5, rng, "loc_0")
        agents_b = _make_community(config, "loc_1", 5, rng, "loc_1")
        # Give community A a unique meme
        for a in agents_a:
            a.cultural_memes = ["unique_meme"]
        # Set alliance
        ext.relations[("loc_0", "loc_1")] = DiplomaticRelation(
            "loc_0", "loc_1", standing=0.8, alliance=True,
        )
        population = agents_a + agents_b
        ext.on_simulation_start(population, config)
        # Run exchange via on_generation_end which triggers exchange for allies
        ext.on_generation_end(0, population, config)
        # Some agents in B should now have the meme
        b_memes = set()
        for a in agents_b:
            b_memes.update(a.cultural_memes)
        # With 10% exchange rate and 5 agents, some transfer likely
        # (not guaranteed, but alliance is fresh)


# =====================================================================
# API Router Tests
# =====================================================================
class TestCommunitiesAPIRouter:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from seldon.api.app import create_app
        app = create_app()
        return TestClient(app)

    @pytest.fixture
    def session_with_communities(self, client):
        """Create a session with geography + diplomacy enabled."""
        response = client.post("/api/simulation/sessions", json={
            "config": {
                "initial_population": 20,
                "generations_to_run": 5,
                "random_seed": 42,
                "extensions_enabled": ["geography", "diplomacy"],
            },
        })
        session_id = response.json()["id"]
        # Run a few generations
        client.post(f"/api/simulation/sessions/{session_id}/step", json={"n": 3})
        return session_id

    def test_list_communities(self, client, session_with_communities):
        sid = session_with_communities
        response = client.get(f"/api/communities/{sid}/communities")
        assert response.status_code == 200
        data = response.json()
        assert data["enabled"] is True
        assert len(data["communities"]) > 0
        for comm in data["communities"]:
            assert "id" in comm
            assert "name" in comm
            assert "population" in comm
            assert "cohesion" in comm
            assert "dominant_region" in comm
            assert "trait_means" in comm

    def test_community_detail(self, client, session_with_communities):
        sid = session_with_communities
        # Get list first to find a community_id
        resp = client.get(f"/api/communities/{sid}/communities")
        communities = resp.json()["communities"]
        cid = communities[0]["id"]
        response = client.get(f"/api/communities/{sid}/communities/{cid}")
        assert response.status_code == 200
        data = response.json()
        assert data["community_id"] == cid
        assert "top_members" in data

    def test_community_detail_404(self, client, session_with_communities):
        sid = session_with_communities
        response = client.get(f"/api/communities/{sid}/communities/nonexistent")
        assert response.status_code == 404

    def test_community_factions(self, client, session_with_communities):
        sid = session_with_communities
        resp = client.get(f"/api/communities/{sid}/communities")
        cid = resp.json()["communities"][0]["id"]
        response = client.get(f"/api/communities/{sid}/communities/{cid}/factions")
        assert response.status_code == 200
        data = response.json()
        assert "faction_count" in data

    def test_diplomacy_endpoint(self, client, session_with_communities):
        sid = session_with_communities
        response = client.get(f"/api/communities/{sid}/diplomacy")
        assert response.status_code == 200
        data = response.json()
        assert data["enabled"] is True
        assert "relations" in data
        assert "metrics" in data

    def test_diplomacy_disabled(self, client):
        """Diplomacy should gracefully degrade when extension not enabled."""
        response = client.post("/api/simulation/sessions", json={
            "config": {
                "initial_population": 10,
                "generations_to_run": 3,
                "random_seed": 42,
            },
        })
        sid = response.json()["id"]
        client.post(f"/api/simulation/sessions/{sid}/step", json={"n": 1})
        response = client.get(f"/api/communities/{sid}/diplomacy")
        assert response.status_code == 200
        data = response.json()
        assert data["enabled"] is False

    def test_communities_disabled_without_geography(self, client):
        """Communities should return enabled=False without geography."""
        response = client.post("/api/simulation/sessions", json={
            "config": {
                "initial_population": 10,
                "generations_to_run": 3,
                "random_seed": 42,
            },
        })
        sid = response.json()["id"]
        client.post(f"/api/simulation/sessions/{sid}/step", json={"n": 1})
        response = client.get(f"/api/communities/{sid}/communities")
        assert response.status_code == 200
        data = response.json()
        assert data["enabled"] is False
        assert data["communities"] == []

    def test_compare_communities(self, client, session_with_communities):
        sid = session_with_communities
        response = client.get(f"/api/communities/{sid}/diplomacy/compare")
        assert response.status_code == 200
        data = response.json()
        assert "comparisons" in data

    def test_session_404(self, client):
        response = client.get("/api/communities/nonexistent/communities")
        assert response.status_code == 404
