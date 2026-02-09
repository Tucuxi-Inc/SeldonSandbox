"""
Tests for Phase 11: Environmental Pressures & Events.

Tests EnvironmentExtension and environment API router.
"""

from __future__ import annotations

import numpy as np
import pytest

from seldon.core.agent import Agent
from seldon.core.config import ExperimentConfig
from seldon.core.processing import ProcessingRegion
from seldon.extensions.environment import (
    EnvironmentExtension,
    EventType,
    Season,
    SEASON_EFFECTS,
)
from seldon.extensions.geography import GeographyExtension
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
    agents = []
    for i in range(n):
        a = _make_agent(f"a{i}", config=config)
        if locations:
            a.location_id = locations[i % len(locations)]
        agents.append(a)
    return agents


# =====================================================================
# EnvironmentExtension Tests
# =====================================================================
class TestEnvironmentExtension:
    def test_extension_name(self):
        ext = EnvironmentExtension()
        assert ext.name == "environment"

    def test_default_config_requires_geography(self):
        ext = EnvironmentExtension()
        dc = ext.get_default_config()
        assert "geography" in dc["requires"]

    def test_on_simulation_start_initializes_climate(self):
        config = _make_config()
        ext = EnvironmentExtension()
        agents = _make_population(config, 10, ["loc_0", "loc_1"])
        ext.on_simulation_start(agents, config)
        assert "loc_0" in ext.climate_states
        assert "loc_1" in ext.climate_states
        assert ext.current_season == Season.SPRING

    def test_season_advances(self):
        config = _make_config(environment_config={
            "enabled": True, "seasons_enabled": True,
            "season_length_generations": 2,
        })
        ext = EnvironmentExtension()
        agents = _make_population(config, 5, ["loc_0"])
        ext.on_simulation_start(agents, config)
        assert ext.current_season == Season.SPRING
        # Advance 2 generations → should switch to SUMMER
        ext.on_generation_start(0, agents, config)
        ext.on_generation_start(1, agents, config)
        assert ext.current_season == Season.SUMMER

    def test_season_cycles_fully(self):
        config = _make_config(environment_config={
            "enabled": True, "seasons_enabled": True,
            "season_length_generations": 1,
        })
        ext = EnvironmentExtension()
        agents = _make_population(config, 5, ["loc_0"])
        ext.on_simulation_start(agents, config)
        seasons = [ext.current_season]
        for gen in range(4):
            ext.on_generation_start(gen, agents, config)
            seasons.append(ext.current_season)
        # Should see: SPRING, SUMMER, AUTUMN, WINTER, SPRING
        assert seasons[1] == Season.SUMMER
        assert seasons[2] == Season.AUTUMN
        assert seasons[3] == Season.WINTER
        assert seasons[4] == Season.SPRING

    def test_climate_drift(self):
        config = _make_config()
        ext = EnvironmentExtension()
        agents = _make_population(config, 5, ["loc_0"])
        ext.on_simulation_start(agents, config)
        initial_temp = ext.climate_states["loc_0"].temperature
        for gen in range(50):
            ext.on_generation_start(gen, agents, config)
        final_temp = ext.climate_states["loc_0"].temperature
        # Temperature should have drifted (may be same due to small drift, but shouldn't crash)
        assert 0.0 <= final_temp <= 1.0

    def test_events_generated_high_probability(self):
        config = _make_config(environment_config={
            "enabled": True,
            "drought_probability": 0.9,
            "flood_probability": 0.9,
            "bountiful_probability": 0.9,
        })
        ext = EnvironmentExtension()
        agents = _make_population(config, 5, ["loc_0"])
        ext.on_simulation_start(agents, config)
        ext.on_generation_start(0, agents, config)
        assert len(ext.event_history) > 0

    def test_events_expire(self):
        config = _make_config()
        ext = EnvironmentExtension()
        agents = _make_population(config, 5, ["loc_0"])
        ext.on_simulation_start(agents, config)
        # Manually add a 1-duration event
        from seldon.extensions.environment import EnvironmentalEvent
        ext.active_events.append(EnvironmentalEvent(
            event_type=EventType.DROUGHT,
            generation=0,
            location_id="loc_0",
            severity=0.8,
            duration=1,
            remaining=1,
        ))
        assert len(ext.active_events) == 1
        ext.on_generation_end(0, agents, config)
        assert len(ext.active_events) == 0

    def test_plague_starts_disease(self):
        config = _make_config(environment_config={
            "enabled": True, "plague_probability": 1.0,
        })
        ext = EnvironmentExtension()
        agents = _make_population(config, 5, ["loc_0"])
        ext.on_simulation_start(agents, config)
        ext.on_generation_start(0, agents, config)
        # With 100% plague probability, a disease should start
        assert len(ext.active_diseases) > 0

    def test_modify_mortality_season_effect(self):
        config = _make_config()
        ext = EnvironmentExtension()
        ext.current_season = Season.WINTER
        agent = _make_agent("a1", config=config)
        rate = ext.modify_mortality(agent, 0.1, config)
        assert rate > 0.1  # Winter increases mortality

    def test_modify_mortality_spring_benefit(self):
        config = _make_config()
        ext = EnvironmentExtension()
        ext.current_season = Season.SPRING
        agent = _make_agent("a1", config=config)
        rate = ext.modify_mortality(agent, 0.1, config)
        assert rate < 0.1  # Spring reduces mortality

    def test_modify_mortality_event_effect(self):
        config = _make_config()
        ext = EnvironmentExtension()
        ext.current_season = Season.SUMMER  # Neutral season
        agent = _make_agent("a1", config=config)
        agent.location_id = "loc_0"
        from seldon.extensions.environment import EnvironmentalEvent
        ext.active_events.append(EnvironmentalEvent(
            event_type=EventType.DROUGHT,
            generation=0,
            location_id="loc_0",
            severity=1.0,
            duration=1,
            remaining=1,
        ))
        rate = ext.modify_mortality(agent, 0.1, config)
        assert rate > 0.1  # Drought increases mortality

    def test_modify_decision_negative_event_migrate(self):
        config = _make_config()
        ext = EnvironmentExtension()
        agent = _make_agent("a1", config=config)
        agent.location_id = "loc_0"
        from seldon.extensions.environment import EnvironmentalEvent
        ext.active_events.append(EnvironmentalEvent(
            event_type=EventType.FLOOD,
            generation=0,
            location_id="loc_0",
            severity=0.8,
        ))
        utils = {"stay": 0.5, "migrate": 0.3}
        result = ext.modify_decision(agent, "migration", utils, config)
        assert result["migrate"] > 0.3

    def test_modify_decision_positive_event_stay(self):
        config = _make_config()
        ext = EnvironmentExtension()
        agent = _make_agent("a1", config=config)
        agent.location_id = "loc_0"
        from seldon.extensions.environment import EnvironmentalEvent
        ext.active_events.append(EnvironmentalEvent(
            event_type=EventType.BOUNTIFUL_HARVEST,
            generation=0,
            location_id="loc_0",
            severity=0.8,
        ))
        utils = {"stay": 0.5, "migrate": 0.3}
        result = ext.modify_decision(agent, "migration", utils, config)
        assert result["stay"] > 0.5

    def test_get_metrics(self):
        config = _make_config()
        ext = EnvironmentExtension()
        agents = _make_population(config, 5, ["loc_0"])
        ext.on_simulation_start(agents, config)
        ext.on_generation_start(0, agents, config)
        metrics = ext.get_metrics(agents)
        assert "current_season" in metrics
        assert "active_events" in metrics
        assert "climate_states" in metrics

    def test_event_history(self):
        config = _make_config(environment_config={
            "enabled": True, "bountiful_probability": 1.0,
        })
        ext = EnvironmentExtension()
        agents = _make_population(config, 5, ["loc_0"])
        ext.on_simulation_start(agents, config)
        ext.on_generation_start(0, agents, config)
        history = ext.get_event_history()
        assert len(history) > 0
        assert history[0]["generation"] == 0

    def test_events_for_generation(self):
        config = _make_config(environment_config={
            "enabled": True, "discovery_probability": 1.0,
        })
        ext = EnvironmentExtension()
        agents = _make_population(config, 5, ["loc_0"])
        ext.on_simulation_start(agents, config)
        ext.on_generation_start(0, agents, config)
        ext.on_generation_start(1, agents, config)
        gen0 = ext.get_events_for_generation(0)
        gen1 = ext.get_events_for_generation(1)
        assert all(e["generation"] == 0 for e in gen0)
        assert all(e["generation"] == 1 for e in gen1)

    def test_registry_registration(self):
        geo = GeographyExtension()
        ext = EnvironmentExtension()
        registry = ExtensionRegistry()
        registry.register(geo)
        registry.register(ext)
        registry.enable("geography")
        registry.enable("environment")
        enabled = [e.name for e in registry.get_enabled()]
        assert "environment" in enabled


# =====================================================================
# API Router Tests
# =====================================================================
class TestEnvironmentAPIRouter:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from seldon.api.app import create_app
        app = create_app()
        return TestClient(app)

    @pytest.fixture
    def session_with_environment(self, client):
        response = client.post("/api/simulation/sessions", json={
            "config": {
                "initial_population": 20,
                "generations_to_run": 10,
                "random_seed": 42,
                "extensions_enabled": ["geography", "environment"],
            },
        })
        session_id = response.json()["id"]
        client.post(f"/api/simulation/sessions/{session_id}/step", json={"n": 5})
        return session_id

    def test_climate_endpoint(self, client, session_with_environment):
        sid = session_with_environment
        resp = client.get(f"/api/environment/{sid}/climate")
        assert resp.status_code == 200
        data = resp.json()
        assert data["enabled"] is True
        assert "season" in data
        assert "climate_states" in data

    def test_events_endpoint(self, client, session_with_environment):
        sid = session_with_environment
        resp = client.get(f"/api/environment/{sid}/events")
        assert resp.status_code == 200
        data = resp.json()
        assert data["enabled"] is True
        assert "events" in data

    def test_events_for_generation(self, client, session_with_environment):
        sid = session_with_environment
        resp = client.get(f"/api/environment/{sid}/events/0")
        assert resp.status_code == 200
        data = resp.json()
        assert data["enabled"] is True
        assert data["generation"] == 0

    def test_disease_endpoint(self, client, session_with_environment):
        sid = session_with_environment
        resp = client.get(f"/api/environment/{sid}/disease")
        assert resp.status_code == 200
        data = resp.json()
        assert data["enabled"] is True
        assert "diseases" in data

    def test_environment_disabled(self, client):
        resp = client.post("/api/simulation/sessions", json={
            "config": {"initial_population": 10, "random_seed": 42},
        })
        sid = resp.json()["id"]
        resp = client.get(f"/api/environment/{sid}/climate")
        assert resp.status_code == 200
        assert resp.json()["enabled"] is False

    def test_session_404(self, client):
        resp = client.get("/api/environment/nonexistent/climate")
        assert resp.status_code == 404
