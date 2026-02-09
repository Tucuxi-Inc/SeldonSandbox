"""
Tests for Phase 10: Economics & Trade Systems.

Tests EconomicsExtension and economics API router.
"""

from __future__ import annotations

import numpy as np
import pytest

from seldon.core.agent import Agent
from seldon.core.config import ExperimentConfig
from seldon.extensions.economics import EconomicsExtension, OCCUPATIONS, RESOURCE_TYPES
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
