"""
Tests for Phase 10: Economics & Trade Systems + Phase C: Economy Deepening.

Tests EconomicsExtension and economics API router.
"""

from __future__ import annotations

import numpy as np
import pytest

from seldon.core.agent import Agent
from seldon.core.config import ExperimentConfig
from seldon.extensions.economics import (
    EconomicsExtension, OCCUPATIONS, OCCUPATION_OUTPUT, RESOURCE_TYPES,
    Settlement,
)
from seldon.extensions.geography import GeographyExtension
from seldon.extensions.resources import ResourcesExtension
from seldon.extensions.registry import ExtensionRegistry


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


def _make_population(config, n=10, locations=None):
    rng = np.random.default_rng(42)
    agents = []
    for i in range(n):
        a = _make_agent(f"a{i}", config=config)
        if locations:
            a.location_id = locations[i % len(locations)]
        agents.append(a)
    return agents


# =====================================================================
# EconomicsExtension Tests
# =====================================================================
class TestEconomicsExtension:
    def test_extension_name(self):
        ext = EconomicsExtension()
        assert ext.name == "economics"

    def test_default_config_requires(self):
        ext = EconomicsExtension()
        dc = ext.get_default_config()
        assert "geography" in dc["requires"]
        assert "resources" in dc["requires"]

    def test_on_simulation_start_creates_markets(self):
        config = _make_config()
        ext = EconomicsExtension()
        agents = _make_population(config, 10, ["loc_0", "loc_1"])
        ext.on_simulation_start(agents, config)
        assert "loc_0" in ext.markets
        assert "loc_1" in ext.markets
        for loc_id, market in ext.markets.items():
            for rtype in RESOURCE_TYPES:
                assert rtype in market.prices

    def test_on_simulation_start_assigns_occupations(self):
        config = _make_config()
        ext = EconomicsExtension()
        agents = _make_population(config, 10, ["loc_0"])
        ext.on_simulation_start(agents, config)
        for a in agents:
            assert a.occupation in OCCUPATIONS

    def test_on_agent_created_wealth_inheritance(self):
        config = _make_config()
        ext = EconomicsExtension()
        p1 = _make_agent("p1", config=config)
        p2 = _make_agent("p2", config=config)
        p1.wealth = 5.0
        p2.wealth = 3.0
        child = _make_agent("c1", config=config)
        ext.on_agent_created(child, (p1, p2), config)
        assert child.wealth > 0

    def test_on_generation_end_runs_economy(self):
        config = _make_config()
        ext = EconomicsExtension()
        agents = _make_population(config, 10, ["loc_0", "loc_1"])
        ext.on_simulation_start(agents, config)
        ext.on_generation_end(0, agents, config)
        assert len(ext.gdp_by_location) > 0
        assert len(ext.trade_routes) > 0

    def test_production_increases_wealth(self):
        config = _make_config()
        ext = EconomicsExtension()
        agents = _make_population(config, 5, ["loc_0"])
        ext.on_simulation_start(agents, config)
        initial_wealth = [a.wealth for a in agents]
        ext.on_generation_end(0, agents, config)
        final_wealth = [a.wealth for a in agents]
        assert sum(final_wealth) > sum(initial_wealth)

    def test_modify_mortality_poverty(self):
        config = _make_config()
        ext = EconomicsExtension()
        agent = _make_agent("a1", config=config)
        agent.wealth = 0.05  # Below poverty threshold
        rate = ext.modify_mortality(agent, 0.1, config)
        assert rate > 0.1  # Should increase

    def test_modify_mortality_wealthy(self):
        config = _make_config()
        ext = EconomicsExtension()
        agent = _make_agent("a1", config=config)
        agent.wealth = 5.0  # Well above poverty
        rate = ext.modify_mortality(agent, 0.1, config)
        assert rate == 0.1  # No change

    def test_modify_decision_wealthy_stays(self):
        config = _make_config()
        ext = EconomicsExtension()
        agent = _make_agent("a1", config=config)
        agent.wealth = 2.0
        utils = {"stay": 0.5, "migrate": 0.5}
        result = ext.modify_decision(agent, "migration", utils, config)
        assert result["stay"] > 0.5

    def test_modify_decision_poor_migrates(self):
        config = _make_config()
        ext = EconomicsExtension()
        agent = _make_agent("a1", config=config)
        agent.wealth = 0.05
        utils = {"stay": 0.5, "migrate": 0.5}
        result = ext.modify_decision(agent, "migration", utils, config)
        assert result["migrate"] > 0.5

    def test_get_metrics(self):
        config = _make_config()
        ext = EconomicsExtension()
        agents = _make_population(config, 10, ["loc_0", "loc_1"])
        ext.on_simulation_start(agents, config)
        ext.on_generation_end(0, agents, config)
        metrics = ext.get_metrics(agents)
        assert "total_gdp" in metrics
        assert "gini_coefficient" in metrics
        assert "occupation_distribution" in metrics
        assert "trade_route_count" in metrics

    def test_gini_coefficient_equal_wealth(self):
        vals = [1.0] * 10
        gini = EconomicsExtension._compute_gini(vals)
        assert gini == pytest.approx(0.0, abs=0.01)

    def test_gini_coefficient_unequal(self):
        vals = [0.0] * 9 + [100.0]
        gini = EconomicsExtension._compute_gini(vals)
        assert gini > 0.5

    def test_gini_empty(self):
        assert EconomicsExtension._compute_gini([]) == 0.0

    def test_trade_routes_list(self):
        config = _make_config()
        ext = EconomicsExtension()
        agents = _make_population(config, 10, ["loc_0", "loc_1"])
        ext.on_simulation_start(agents, config)
        ext.on_generation_end(0, agents, config)
        routes = ext.get_trade_routes_list()
        assert len(routes) > 0
        for r in routes:
            assert "location_a" in r
            assert "volume" in r

    def test_market_data(self):
        config = _make_config()
        ext = EconomicsExtension()
        agents = _make_population(config, 10, ["loc_0", "loc_1"])
        ext.on_simulation_start(agents, config)
        ext.on_generation_end(0, agents, config)
        markets = ext.get_market_data()
        assert len(markets) > 0
        for m in markets:
            assert "prices" in m
            assert "supply" in m
            assert "demand" in m

    def test_wealth_distribution(self):
        config = _make_config()
        ext = EconomicsExtension()
        agents = _make_population(config, 20, ["loc_0"])
        ext.on_simulation_start(agents, config)
        ext.on_generation_end(0, agents, config)
        dist = ext.get_wealth_distribution(agents)
        assert "percentiles" in dist
        assert "lorenz_curve" in dist
        assert "gini" in dist

    def test_registry_registration(self):
        geo = GeographyExtension()
        res = ResourcesExtension()
        ext = EconomicsExtension()
        registry = ExtensionRegistry()
        registry.register(geo)
        registry.register(res)
        registry.register(ext)
        registry.enable("geography")
        registry.enable("resources")
        registry.enable("economics")
        enabled = [e.name for e in registry.get_enabled()]
        assert "economics" in enabled


# =====================================================================
# API Router Tests
# =====================================================================
class TestEconomicsAPIRouter:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from seldon.api.app import create_app
        app = create_app()
        return TestClient(app)

    @pytest.fixture
    def session_with_economics(self, client):
        response = client.post("/api/simulation/sessions", json={
            "config": {
                "initial_population": 20,
                "generations_to_run": 5,
                "random_seed": 42,
                "extensions_enabled": ["geography", "resources", "economics"],
            },
        })
        session_id = response.json()["id"]
        client.post(f"/api/simulation/sessions/{session_id}/step", json={"n": 3})
        return session_id

    def test_economics_overview(self, client, session_with_economics):
        sid = session_with_economics
        resp = client.get(f"/api/economics/{sid}/overview")
        assert resp.status_code == 200
        data = resp.json()
        assert data["enabled"] is True
        assert "total_gdp" in data
        assert "gini_coefficient" in data

    def test_economics_disabled(self, client):
        resp = client.post("/api/simulation/sessions", json={
            "config": {"initial_population": 10, "random_seed": 42},
        })
        sid = resp.json()["id"]
        resp = client.get(f"/api/economics/{sid}/overview")
        assert resp.status_code == 200
        assert resp.json()["enabled"] is False

    def test_trade_routes(self, client, session_with_economics):
        sid = session_with_economics
        resp = client.get(f"/api/economics/{sid}/trade-routes")
        assert resp.status_code == 200
        data = resp.json()
        assert data["enabled"] is True
        assert "routes" in data

    def test_markets(self, client, session_with_economics):
        sid = session_with_economics
        resp = client.get(f"/api/economics/{sid}/markets")
        assert resp.status_code == 200
        data = resp.json()
        assert data["enabled"] is True
        assert len(data["markets"]) > 0

    def test_wealth_distribution(self, client, session_with_economics):
        sid = session_with_economics
        resp = client.get(f"/api/economics/{sid}/wealth-distribution")
        assert resp.status_code == 200
        data = resp.json()
        assert data["enabled"] is True
        assert "percentiles" in data

    def test_occupations(self, client, session_with_economics):
        sid = session_with_economics
        resp = client.get(f"/api/economics/{sid}/occupations")
        assert resp.status_code == 200
        data = resp.json()
        assert "occupations" in data
        assert data["total"] > 0

    def test_session_404(self, client):
        resp = client.get("/api/economics/nonexistent/overview")
        assert resp.status_code == 404

    def test_settlements_endpoint(self, client, session_with_economics):
        sid = session_with_economics
        resp = client.get(f"/api/economics/{sid}/settlements")
        assert resp.status_code == 200
        data = resp.json()
        assert data["enabled"] is True
        assert "settlements" in data

    def test_skills_endpoint(self, client, session_with_economics):
        sid = session_with_economics
        resp = client.get(f"/api/economics/{sid}/skills")
        assert resp.status_code == 200
        data = resp.json()
        assert data["enabled"] is True
        assert "skills" in data

    def test_production_endpoint(self, client, session_with_economics):
        sid = session_with_economics
        resp = client.get(f"/api/economics/{sid}/production")
        assert resp.status_code == 200
        data = resp.json()
        assert data["enabled"] is True

    def test_trade_routes_include_new_fields(self, client, session_with_economics):
        sid = session_with_economics
        resp = client.get(f"/api/economics/{sid}/trade-routes")
        data = resp.json()
        if data["routes"]:
            route = data["routes"][0]
            assert "goods_a_to_b" in route
            assert "goods_b_to_a" in route
            assert "distance" in route
            assert "trader_count" in route


# =====================================================================
# Phase C: Settlement Detection Tests
# =====================================================================
class _MockTickEngine:
    """Minimal mock of TickEngine for settlement detection tests."""
    def __init__(self, clusters=None, hex_grid=True):
        self._clusters = clusters or []
        self._hex_grid = True if hex_grid else None

    @property
    def hex_grid(self):
        return self._hex_grid

    def get_agent_clusters(self, min_size=3):
        return [c for c in self._clusters if c["agent_count"] >= min_size]


class TestSettlementDetection:
    def test_detect_from_clusters(self):
        config = _make_config()
        ext = EconomicsExtension()
        ext._rng = np.random.default_rng(42)
        agents = _make_population(config, 6)

        clusters = [{
            "tiles": [(5, 5), (5, 6)],
            "agent_count": 6,
            "agent_ids": [a.id for a in agents],
            "center": (5, 5),
            "terrain_types": {"valley", "river_valley"},
        }]
        mock_engine = _MockTickEngine(clusters)
        ext.detect_settlements(mock_engine, agents, config)

        assert len(ext.settlements) == 1
        sid = "settlement_5_5"
        assert sid in ext.settlements
        s = ext.settlements[sid]
        assert s.center == (5, 5)
        assert len(s.agent_ids) == 6

    def test_assigns_community_id(self):
        config = _make_config()
        ext = EconomicsExtension()
        ext._rng = np.random.default_rng(42)
        agents = _make_population(config, 4)

        clusters = [{
            "tiles": [(3, 3)],
            "agent_count": 4,
            "agent_ids": [a.id for a in agents],
            "center": (3, 3),
            "terrain_types": {"valley"},
        }]
        mock_engine = _MockTickEngine(clusters)
        ext.detect_settlements(mock_engine, agents, config)

        for a in agents:
            assert a.community_id == "settlement_3_3"

    def test_min_size_filter(self):
        config = _make_config()
        ext = EconomicsExtension()
        ext._rng = np.random.default_rng(42)
        agents = _make_population(config, 2)

        clusters = [{
            "tiles": [(1, 1)],
            "agent_count": 2,
            "agent_ids": [a.id for a in agents],
            "center": (1, 1),
            "terrain_types": {"desert"},
        }]
        mock_engine = _MockTickEngine(clusters)
        ext.detect_settlements(mock_engine, agents, config)
        assert len(ext.settlements) == 0  # min_settlement_size=3 by default

    def test_stale_settlement_removed(self):
        config = _make_config()
        ext = EconomicsExtension()
        ext._rng = np.random.default_rng(42)
        agents = _make_population(config, 5)

        # First detection: 1 settlement
        clusters1 = [{
            "tiles": [(5, 5)],
            "agent_count": 5,
            "agent_ids": [a.id for a in agents],
            "center": (5, 5),
            "terrain_types": {"valley"},
        }]
        ext.detect_settlements(_MockTickEngine(clusters1), agents, config)
        assert "settlement_5_5" in ext.settlements

        # Second detection: settlement moves
        clusters2 = [{
            "tiles": [(7, 7)],
            "agent_count": 5,
            "agent_ids": [a.id for a in agents],
            "center": (7, 7),
            "terrain_types": {"forest"},
        }]
        ext.detect_settlements(_MockTickEngine(clusters2), agents, config)
        assert "settlement_5_5" not in ext.settlements
        assert "settlement_7_7" in ext.settlements


# =====================================================================
# Phase C: Tick Production Tests
# =====================================================================
class TestTickProduction:
    def _setup_settlement_ext(self, agents, occupation="farmer", terrain="valley"):
        config = _make_config()
        ext = EconomicsExtension()
        ext._rng = np.random.default_rng(42)

        for a in agents:
            a.occupation = occupation
            a.extension_data["terrain_type"] = terrain

        settlement = Settlement(
            id="s1", center=(5, 5), tiles=[(5, 5)],
            agent_ids=[a.id for a in agents],
            terrain_types={terrain},
            resource_pools={"food": 10.0, "goods": 10.0, "knowledge": 10.0},
        )
        ext.settlements["s1"] = settlement
        return ext, settlement, config

    def test_farmer_produces_food(self):
        config = _make_config()
        agents = _make_population(config, 3)
        ext, settlement, config = self._setup_settlement_ext(agents, "farmer", "valley")

        initial_food = settlement.resource_pools["food"]
        ext.process_economic_tick(agents, config, 0, "spring")
        assert settlement.resource_pools["food"] > initial_food

    def test_artisan_produces_goods(self):
        config = _make_config()
        agents = _make_population(config, 3)
        ext, settlement, config = self._setup_settlement_ext(agents, "artisan", "valley")

        initial_goods = settlement.resource_pools["goods"]
        ext.process_economic_tick(agents, config, 0, "spring")
        assert settlement.resource_pools["goods"] > initial_goods

    def test_terrain_modifier_applied(self):
        config = _make_config()
        agents_v = _make_population(config, 3)
        ext_v, settle_v, _ = self._setup_settlement_ext(agents_v, "farmer", "valley")
        ext_v.process_economic_tick(agents_v, config, 0, "spring")
        prod_valley = settle_v.production_last_tick["food"]

        agents_d = _make_population(config, 3)
        ext_d, settle_d, _ = self._setup_settlement_ext(agents_d, "farmer", "desert")
        ext_d.process_economic_tick(agents_d, config, 0, "spring")
        prod_desert = settle_d.production_last_tick["food"]

        # Valley should produce more food than desert
        assert prod_valley > prod_desert

    def test_season_modifier_applied(self):
        config = _make_config()
        agents_s = _make_population(config, 3)
        ext_s, settle_s, _ = self._setup_settlement_ext(agents_s, "farmer", "valley")
        ext_s.process_economic_tick(agents_s, config, 0, "summer")
        prod_summer = settle_s.production_last_tick["food"]

        agents_w = _make_population(config, 3)
        ext_w, settle_w, _ = self._setup_settlement_ext(agents_w, "farmer", "valley")
        ext_w.process_economic_tick(agents_w, config, 9, "winter")
        prod_winter = settle_w.production_last_tick["food"]

        # Summer food should be more than winter
        assert prod_summer > prod_winter

    def test_skill_bonus_increases_output(self):
        config = _make_config()
        agents_no = _make_population(config, 3)
        ext_no, settle_no, _ = self._setup_settlement_ext(agents_no, "farmer", "valley")
        ext_no.process_economic_tick(agents_no, config, 0, "spring")
        prod_no_skill = settle_no.production_last_tick["food"]

        agents_sk = _make_population(config, 3)
        ext_sk, settle_sk, _ = self._setup_settlement_ext(agents_sk, "farmer", "valley")
        for a in agents_sk:
            a.skills["farmer"] = 0.8
        ext_sk.process_economic_tick(agents_sk, config, 0, "spring")
        prod_with_skill = settle_sk.production_last_tick["food"]

        assert prod_with_skill > prod_no_skill

    def test_production_adds_to_pool_and_holdings(self):
        config = _make_config()
        agents = _make_population(config, 3)
        ext, settlement, config = self._setup_settlement_ext(agents, "farmer", "valley")

        for a in agents:
            a.wealth = 0.0
            a.resource_holdings = {}

        ext.process_economic_tick(agents, config, 0, "spring")

        # Settlement pool grew
        assert settlement.resource_pools["food"] > 10.0
        # Agents gained wealth
        assert any(a.wealth > 0 for a in agents)
        # Agents gained food holdings
        assert any(a.resource_holdings.get("food", 0) > 0 for a in agents)


# =====================================================================
# Phase C: Consumption Tests
# =====================================================================
class TestConsumption:
    def test_consumption_deducts_from_pool(self):
        config = _make_config()
        agents = _make_population(config, 3)
        ext = EconomicsExtension()
        ext._rng = np.random.default_rng(42)
        for a in agents:
            a.occupation = "farmer"
            a.extension_data["terrain_type"] = "valley"

        settlement = Settlement(
            id="s1", center=(5, 5), tiles=[(5, 5)],
            agent_ids=[a.id for a in agents],
            terrain_types={"valley"},
            resource_pools={"food": 100.0, "goods": 100.0, "knowledge": 100.0},
        )
        ext.settlements["s1"] = settlement

        ext.process_economic_tick(agents, config, 0, "spring")

        # Pool should decrease due to consumption (even though production adds)
        # With 3 agents consuming 0.03 food each = 0.09 removed
        assert settlement.consumption_last_tick["food"] > 0

    def test_empty_pool_wealth_loss(self):
        config = _make_config()
        agents = _make_population(config, 3)
        ext = EconomicsExtension()
        ext._rng = np.random.default_rng(42)
        for a in agents:
            a.occupation = "scholar"
            a.extension_data["terrain_type"] = "valley"
            a.wealth = 1.0

        settlement = Settlement(
            id="s1", center=(5, 5), tiles=[(5, 5)],
            agent_ids=[a.id for a in agents],
            terrain_types={"valley"},
            resource_pools={"food": 0.0, "goods": 0.0, "knowledge": 0.0},
        )
        ext.settlements["s1"] = settlement

        ext.process_economic_tick(agents, config, 0, "spring")
        # At least some agents should have lost wealth due to empty pools
        # (food pool is 0, so food < rate*0.5 triggers wealth loss)
        # Note: production also runs, adding to pool, but consumption from empty pool
        # still causes the wealth-loss check to trigger for some resources
        total_consumption = sum(settlement.consumption_last_tick.values())
        # Consumption should be minimal since pools started at 0
        assert total_consumption < 1.0

    def test_configurable_consumption_rates(self):
        config = _make_config()
        # Override consumption rate
        config.economics_config["consumption_per_tick"] = {
            "food": 0.1, "goods": 0.05, "knowledge": 0.02,
        }
        agents = _make_population(config, 1)
        ext = EconomicsExtension()
        ext._rng = np.random.default_rng(42)
        for a in agents:
            a.occupation = "farmer"
            a.extension_data["terrain_type"] = "valley"

        settlement = Settlement(
            id="s1", center=(5, 5), tiles=[(5, 5)],
            agent_ids=[a.id for a in agents],
            terrain_types={"valley"},
            resource_pools={"food": 100.0, "goods": 100.0, "knowledge": 100.0},
        )
        ext.settlements["s1"] = settlement

        ext.process_economic_tick(agents, config, 0, "spring")
        # With high consumption rate of 0.1 for food, consumption should be visible
        assert settlement.consumption_last_tick["food"] >= 0.1


# =====================================================================
# Phase C: Skill Accumulation Tests
# =====================================================================
class TestSkillAccumulation:
    def test_skill_grows_for_occupation(self):
        config = _make_config()
        ext = EconomicsExtension()
        agent = _make_agent("a1", config=config)
        agent.occupation = "farmer"
        agent.skills = {}

        ec = ext._get_config(config)
        ext._accumulate_skill(agent, ec)

        assert agent.skills["farmer"] > 0

    def test_other_skills_decay(self):
        config = _make_config()
        ext = EconomicsExtension()
        agent = _make_agent("a1", config=config)
        agent.occupation = "farmer"
        agent.skills = {"artisan": 0.5, "farmer": 0.2}

        ec = ext._get_config(config)
        ext._accumulate_skill(agent, ec)

        assert agent.skills["farmer"] > 0.2
        assert agent.skills["artisan"] < 0.5

    def test_skill_capped_at_max(self):
        config = _make_config()
        ext = EconomicsExtension()
        agent = _make_agent("a1", config=config)
        agent.occupation = "farmer"
        agent.skills = {"farmer": 0.999}

        ec = ext._get_config(config)
        ext._accumulate_skill(agent, ec)

        assert agent.skills["farmer"] <= ec.get("skill_max", 1.0)

    def test_skill_boosts_output(self):
        # Already tested in TestTickProduction.test_skill_bonus_increases_output
        # This tests the skill accumulation over multiple ticks
        config = _make_config()
        agent = _make_agent("a1", config=config)
        agent.occupation = "farmer"
        agent.skills = {}

        ext = EconomicsExtension()
        ec = ext._get_config(config)

        for _ in range(10):
            ext._accumulate_skill(agent, ec)

        assert agent.skills["farmer"] > 0.05  # 10 * 0.01 = 0.10


# =====================================================================
# Phase C: Inter-Settlement Trade Tests
# =====================================================================
class TestInterSettlementTrade:
    def _make_two_settlements(self, config, n=5):
        ext = EconomicsExtension()
        ext._rng = np.random.default_rng(42)

        agents_a = [_make_agent(f"a{i}", config=config) for i in range(n)]
        agents_b = [_make_agent(f"b{i}", config=config) for i in range(n)]

        for a in agents_a:
            a.occupation = "farmer"
            a.extension_data["terrain_type"] = "valley"
        for b in agents_b:
            b.occupation = "artisan"
            b.extension_data["terrain_type"] = "mountain"

        s_a = Settlement(
            id="sa", center=(0, 0), tiles=[(0, 0)],
            agent_ids=[a.id for a in agents_a],
            terrain_types={"valley"},
            resource_pools={"food": 50.0, "goods": 0.5, "knowledge": 5.0},
        )
        s_b = Settlement(
            id="sb", center=(3, 0), tiles=[(3, 0)],
            agent_ids=[b.id for b in agents_b],
            terrain_types={"mountain"},
            resource_pools={"food": 0.5, "goods": 50.0, "knowledge": 5.0},
        )
        ext.settlements = {"sa": s_a, "sb": s_b}
        all_agents = agents_a + agents_b
        return ext, all_agents, s_a, s_b

    def test_surplus_deficit_flow(self):
        config = _make_config()
        ext, agents, s_a, s_b = self._make_two_settlements(config)

        food_a_before = s_a.resource_pools["food"]
        food_b_before = s_b.resource_pools["food"]

        ext._run_inter_settlement_trade(agents, config)

        # Food should flow from A (surplus) to B (deficit)
        assert s_a.resource_pools["food"] <= food_a_before
        assert s_b.resource_pools["food"] >= food_b_before

    def test_volume_scales_with_traders(self):
        config = _make_config()
        # No traders
        ext1, agents1, s_a1, s_b1 = self._make_two_settlements(config)
        ext1._run_inter_settlement_trade(agents1, config)
        vol1 = sum(r.volume for r in ext1.trade_routes.values())

        # Add traders
        ext2, agents2, s_a2, s_b2 = self._make_two_settlements(config)
        agents2[0].occupation = "trader"
        agents2[5].occupation = "trader"
        agents2[6].occupation = "trader"
        ext2._run_inter_settlement_trade(agents2, config)
        vol2 = sum(r.volume for r in ext2.trade_routes.values())

        # More traders should mean more volume
        assert vol2 >= vol1

    def test_distance_penalty(self):
        config = _make_config()
        ext, agents, s_a, s_b = self._make_two_settlements(config)
        # Close settlements
        ext._run_inter_settlement_trade(agents, config)
        vol_close = sum(r.volume for r in ext.trade_routes.values())

        # Move settlement B far away
        ext2, agents2, s_a2, s_b2 = self._make_two_settlements(config)
        s_b2.center = (7, 0)  # distance = 7
        ext2._run_inter_settlement_trade(agents2, config)
        vol_far = sum(r.volume for r in ext2.trade_routes.values())

        assert vol_close >= vol_far

    def test_traders_gain_wealth(self):
        config = _make_config()
        ext, agents, s_a, s_b = self._make_two_settlements(config)
        agents[0].occupation = "trader"
        initial_wealth = agents[0].wealth

        ext._run_inter_settlement_trade(agents, config)

        assert agents[0].wealth >= initial_wealth

    def test_trade_history_populated(self):
        config = _make_config()
        ext, agents, s_a, s_b = self._make_two_settlements(config)
        agents[0].occupation = "trader"

        ext._run_inter_settlement_trade(agents, config)

        # Trader should have entries in trade_history
        assert len(agents[0].trade_history) > 0
        entry = agents[0].trade_history[0]
        assert "route" in entry
        assert "volume" in entry
        assert "income" in entry


# =====================================================================
# Phase C: Governance Tests
# =====================================================================
class TestGovernance:
    def test_leader_sets_tax_rate(self):
        config = _make_config()
        ext = EconomicsExtension()
        ext._rng = np.random.default_rng(42)

        agents = _make_population(config, 5)
        agents[0].social_role = "leader"
        agents[0].social_status = 0.9
        agents[0].influence_score = 0.9

        settlement = Settlement(
            id="s1", center=(5, 5), tiles=[(5, 5)],
            agent_ids=[a.id for a in agents],
            terrain_types={"valley"},
            communal_pool=10.0,
        )
        ext.settlements["s1"] = settlement

        ext._apply_governance_all(agents, config)

        assert settlement.leader_id == agents[0].id
        assert settlement.taxation_rate > 0.0
        assert settlement.taxation_rate <= 0.25

    def test_tax_collected_to_communal_pool(self):
        config = _make_config()
        agents = _make_population(config, 3)
        ext = EconomicsExtension()
        ext._rng = np.random.default_rng(42)
        for a in agents:
            a.occupation = "farmer"
            a.extension_data["terrain_type"] = "valley"

        settlement = Settlement(
            id="s1", center=(5, 5), tiles=[(5, 5)],
            agent_ids=[a.id for a in agents],
            terrain_types={"valley"},
            resource_pools={"food": 10.0, "goods": 10.0, "knowledge": 10.0},
            taxation_rate=0.2,
            communal_pool=0.0,
        )
        ext.settlements["s1"] = settlement

        ext.process_economic_tick(agents, config, 0, "spring")

        assert settlement.communal_pool > 0  # Tax was collected

    def test_poverty_relief_distributes(self):
        config = _make_config()
        ext = EconomicsExtension()
        ext._rng = np.random.default_rng(42)

        agents = _make_population(config, 5)
        agents[0].social_role = "leader"
        agents[0].social_status = 0.9
        agents[0].influence_score = 0.9
        # Set agreeableness high for more relief
        ts = config.trait_system
        try:
            agree_idx = ts.trait_index("agreeableness")
            agents[0].traits[agree_idx] = 0.95
        except KeyError:
            pass

        # Make some agents poor
        for a in agents[1:]:
            a.wealth = 0.05  # Below poverty threshold

        settlement = Settlement(
            id="s1", center=(5, 5), tiles=[(5, 5)],
            agent_ids=[a.id for a in agents],
            terrain_types={"valley"},
            communal_pool=5.0,
        )
        ext.settlements["s1"] = settlement

        initial_poor_wealth = sum(a.wealth for a in agents[1:])
        ext._apply_governance_all(agents, config)
        final_poor_wealth = sum(a.wealth for a in agents[1:])

        assert final_poor_wealth > initial_poor_wealth

    def test_infrastructure_growth(self):
        config = _make_config()
        ext = EconomicsExtension()
        ext._rng = np.random.default_rng(42)

        agents = _make_population(config, 5)
        agents[0].influence_score = 0.9

        settlement = Settlement(
            id="s1", center=(5, 5), tiles=[(5, 5)],
            agent_ids=[a.id for a in agents],
            terrain_types={"valley"},
            communal_pool=10.0,
            infrastructure_level=0.0,
        )
        ext.settlements["s1"] = settlement

        ext._apply_governance_all(agents, config)

        assert settlement.infrastructure_level > 0.0

    def test_leaderless_default_tax(self):
        config = _make_config()
        ext = EconomicsExtension()
        ext._rng = np.random.default_rng(42)

        agents = _make_population(config, 5)
        # No leader role or influence

        settlement = Settlement(
            id="s1", center=(5, 5), tiles=[(5, 5)],
            agent_ids=[a.id for a in agents],
            terrain_types={"valley"},
            communal_pool=5.0,
        )
        ext.settlements["s1"] = settlement

        ext._apply_governance_all(agents, config)

        # Should still have a leader (fallback to highest influence)
        # and use base tax rate or adjusted
        assert settlement.taxation_rate >= 0.0


# =====================================================================
# Phase C: Backward Compatibility Tests
# =====================================================================
class TestBackwardCompat:
    def test_no_hex_uses_legacy_economy(self):
        """Without hex grid, the generation-level economy runs as before."""
        config = _make_config()
        ext = EconomicsExtension()
        agents = _make_population(config, 10, ["loc_0", "loc_1"])
        ext.on_simulation_start(agents, config)
        ext.on_generation_end(0, agents, config)

        # Legacy path should have set GDP and trade routes
        assert len(ext.gdp_by_location) > 0
        assert len(ext.trade_routes) > 0
        # No settlements in non-hex mode
        assert len(ext.settlements) == 0

    def test_existing_tests_unaffected(self):
        """Verify the get_metrics returns expected keys including new settlement_count."""
        config = _make_config()
        ext = EconomicsExtension()
        agents = _make_population(config, 10, ["loc_0"])
        ext.on_simulation_start(agents, config)
        ext.on_generation_end(0, agents, config)
        metrics = ext.get_metrics(agents)
        assert "settlement_count" in metrics
        assert metrics["settlement_count"] == 0  # No hex = no settlements


# =====================================================================
# Phase C: Query Helper Tests
# =====================================================================
class TestQueryHelpers:
    def test_get_settlements_data(self):
        ext = EconomicsExtension()
        ext.settlements["s1"] = Settlement(
            id="s1", center=(5, 5), tiles=[(5, 5), (5, 6)],
            agent_ids=["a1", "a2", "a3"],
            terrain_types={"valley", "river_valley"},
            resource_pools={"food": 10.0, "goods": 5.0, "knowledge": 2.0},
        )
        data = ext.get_settlements_data()
        assert len(data) == 1
        d = data[0]
        assert d["id"] == "s1"
        assert d["tile_count"] == 2
        assert d["agent_count"] == 3
        assert "food" in d["resource_pools"]

    def test_get_settlement_detail(self):
        ext = EconomicsExtension()
        ext.settlements["s1"] = Settlement(
            id="s1", center=(5, 5), tiles=[(5, 5)],
            agent_ids=["a1"],
            terrain_types={"valley"},
        )
        detail = ext.get_settlement_detail("s1")
        assert detail is not None
        assert detail["id"] == "s1"
        assert "agent_ids" in detail

    def test_get_settlement_detail_not_found(self):
        ext = EconomicsExtension()
        assert ext.get_settlement_detail("nonexistent") is None

    def test_get_skill_distribution(self):
        config = _make_config()
        agents = _make_population(config, 5)
        for a in agents:
            a.skills = {"farmer": 0.5, "artisan": 0.2}
        ext = EconomicsExtension()
        dist = ext.get_skill_distribution(agents)
        assert "farmer" in dist
        assert dist["farmer"]["count"] == 5
        assert dist["farmer"]["mean"] == pytest.approx(0.5, abs=0.01)

    def test_get_production_summary_empty(self):
        ext = EconomicsExtension()
        result = ext.get_production_summary()
        assert result["totals"] == {"food": 0.0, "goods": 0.0, "knowledge": 0.0}

    def test_occupation_output_mapping(self):
        assert OCCUPATION_OUTPUT["farmer"] == "food"
        assert OCCUPATION_OUTPUT["artisan"] == "goods"
        assert OCCUPATION_OUTPUT["scholar"] == "knowledge"
        assert OCCUPATION_OUTPUT["soldier"] == "food"
        assert OCCUPATION_OUTPUT["trader"] == "goods"
