"""Tests for World View tick activity capture and single-tick stepping."""

from __future__ import annotations

import numpy as np
import pytest

from seldon.core.agent import Agent
from seldon.core.config import ExperimentConfig
from seldon.core.tick_engine import (
    AgentTickActivity,
    TickActivityLog,
    TickEngine,
    _get_season,
    Season,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tick_config(**overrides) -> ExperimentConfig:
    """Create a tick-engine-enabled config."""
    base = {
        "tick_config": {"enabled": True},
        "needs_config": {"enabled": True},
        "hex_grid_config": {"enabled": True, "width": 6, "height": 4},
        "initial_population": 10,
        "generations_to_run": 5,
        "random_seed": 42,
    }
    base.update(overrides)
    return ExperimentConfig(**base)


def _make_tick_engine(config: ExperimentConfig | None = None) -> TickEngine:
    """Create a TickEngine with initial population."""
    config = config or _make_tick_config()
    engine = TickEngine(config)
    engine.population = engine._create_initial_population()
    # Fire extension hooks
    for ext in engine.extensions.get_enabled():
        ext.on_simulation_start(engine.population, config)
    return engine


# ---------------------------------------------------------------------------
# Dataclass tests
# ---------------------------------------------------------------------------

class TestAgentTickActivity:
    def test_defaults(self):
        ata = AgentTickActivity(agent_id="a1")
        assert ata.agent_id == "a1"
        assert ata.location is None
        assert ata.previous_location is None
        assert ata.activity is None
        assert ata.activity_need is None
        assert ata.health == 1.0
        assert ata.suffering == 0.0
        assert ata.is_pregnant is False

    def test_fields_set(self):
        ata = AgentTickActivity(
            agent_id="a2",
            location=(3, 4),
            previous_location=(3, 3),
            activity="forage",
            activity_need="hunger",
            life_phase="mature",
            processing_region="optimal",
            needs_snapshot={"hunger": 0.7},
            health=0.9,
            suffering=0.1,
            is_pregnant=True,
        )
        assert ata.location == (3, 4)
        assert ata.activity == "forage"
        assert ata.is_pregnant is True


class TestTickActivityLog:
    def test_defaults(self):
        log = TickActivityLog(year=0, tick_in_year=3, global_tick=3, season="summer")
        assert log.year == 0
        assert log.tick_in_year == 3
        assert log.season == "summer"
        assert log.agent_activities == {}
        assert log.events == []
        assert log.population_count == 0


# ---------------------------------------------------------------------------
# Single-tick stepping tests
# ---------------------------------------------------------------------------

class TestRunSingleTick:
    def test_advances_global_tick(self):
        engine = _make_tick_engine()
        initial_tick = engine._global_tick
        engine._run_single_tick()
        assert engine._global_tick == initial_tick + 1

    def test_tick_in_year_increments(self):
        engine = _make_tick_engine()
        assert engine._single_tick_in_year == 0
        engine._run_single_tick()
        assert engine._single_tick_in_year == 1

    def test_returns_tick_activity_log(self):
        engine = _make_tick_engine()
        log = engine._run_single_tick()
        assert isinstance(log, TickActivityLog)
        assert log.year == 0
        assert log.tick_in_year == 0
        assert log.season == "spring"
        assert log.population_count > 0

    def test_activity_log_has_agents(self):
        engine = _make_tick_engine()
        log = engine._run_single_tick()
        # All living agents should have an activity entry
        living_count = sum(1 for a in engine.population if a.is_alive)
        assert len(log.agent_activities) == living_count

    def test_agent_activity_has_location(self):
        engine = _make_tick_engine()
        log = engine._run_single_tick()
        for ata in log.agent_activities.values():
            # With hex grid enabled, agents should have locations
            assert ata.location is not None
            assert len(ata.location) == 2

    def test_agent_activity_has_needs_snapshot(self):
        engine = _make_tick_engine()
        log = engine._run_single_tick()
        for ata in log.agent_activities.values():
            assert isinstance(ata.needs_snapshot, dict)
            # Should have the 6 core needs
            assert len(ata.needs_snapshot) >= 6

    def test_agent_activity_has_processing_region(self):
        engine = _make_tick_engine()
        log = engine._run_single_tick()
        for ata in log.agent_activities.values():
            assert ata.processing_region != ""

    def test_gathering_activity_captured(self):
        engine = _make_tick_engine()
        log = engine._run_single_tick()
        # At least some agents should have gathered (those old enough)
        activities = [
            ata.activity for ata in log.agent_activities.values()
            if ata.activity is not None
        ]
        assert len(activities) > 0
        # Check valid activity names
        valid_activities = {
            "forage", "hunt", "fish", "find_water",
            "build_shelter", "seek_warmth", "rest", "seek_safety",
        }
        for act in activities:
            assert act in valid_activities

    def test_tick_log_buffer_populated(self):
        engine = _make_tick_engine()
        engine._run_single_tick()
        assert len(engine._tick_log_buffer) == 1
        engine._run_single_tick()
        assert len(engine._tick_log_buffer) == 2

    def test_tick_log_buffer_maxlen(self):
        engine = _make_tick_engine()
        for _ in range(30):
            engine._run_single_tick()
        # Buffer maxlen is 24
        assert len(engine._tick_log_buffer) == 24

    def test_year_wraps_at_12_ticks(self):
        engine = _make_tick_engine()
        for i in range(12):
            log = engine._run_single_tick()
        # After 12 ticks, year should have incremented
        assert engine._single_tick_year == 1
        assert engine._single_tick_in_year == 0

    def test_year_complete_flag(self):
        engine = _make_tick_engine()
        for i in range(11):
            log = engine._run_single_tick()
            assert not getattr(log, "year_complete", False)
        # 12th tick completes the year
        log = engine._run_single_tick()
        assert getattr(log, "year_complete", False)

    def test_snapshot_created_at_year_end(self):
        engine = _make_tick_engine()
        assert len(engine.history) == 0
        for _ in range(12):
            engine._run_single_tick()
        assert len(engine.history) == 1

    def test_multiple_years(self):
        engine = _make_tick_engine()
        for _ in range(36):  # 3 years
            engine._run_single_tick()
        assert engine._single_tick_year == 3
        assert len(engine.history) == 3

    def test_season_progresses(self):
        engine = _make_tick_engine()
        seasons_seen = set()
        for _ in range(12):
            log = engine._run_single_tick()
            seasons_seen.add(log.season)
        assert seasons_seen == {"spring", "summer", "autumn", "winter"}

    def test_movement_previous_location_captured(self):
        """Some agents should have previous_location set when they move."""
        engine = _make_tick_engine()
        moved = False
        for _ in range(12):  # Run a full year to increase chance of movement
            log = engine._run_single_tick()
            for ata in log.agent_activities.values():
                if ata.previous_location is not None:
                    moved = True
                    # previous_location should differ from current
                    assert ata.previous_location != ata.location
        # With 10 agents and movement enabled, at least one should move
        assert moved, "Expected at least one agent to move during a full year"


# ---------------------------------------------------------------------------
# Backward compatibility tests
# ---------------------------------------------------------------------------

class TestBackwardCompat:
    def test_run_year_still_works(self):
        """_run_year() must still produce a valid snapshot."""
        engine = _make_tick_engine()
        snapshot = engine._run_year(0)
        assert snapshot.generation == 0
        assert snapshot.population_size > 0

    def test_run_full_still_works(self):
        """Full run() method must still produce snapshots."""
        config = _make_tick_config(generations_to_run=3)
        engine = TickEngine(config)
        history = engine.run(3)
        assert len(history) == 3

    def test_generation_stepping_still_works(self):
        """_run_generation() must still work."""
        engine = _make_tick_engine()
        snapshot = engine._run_generation(0)
        assert snapshot.generation == 0


# ---------------------------------------------------------------------------
# No hex grid / no needs fallback
# ---------------------------------------------------------------------------

class TestNoHexGrid:
    def test_single_tick_without_hex(self):
        """Single-tick stepping works without hex grid."""
        config = _make_tick_config(
            hex_grid_config={"enabled": False},
        )
        engine = TickEngine(config)
        engine.population = engine._create_initial_population()
        for ext in engine.extensions.get_enabled():
            ext.on_simulation_start(engine.population, config)

        log = engine._run_single_tick()
        assert isinstance(log, TickActivityLog)
        assert log.population_count > 0
        # Agents should have no location
        for ata in log.agent_activities.values():
            assert ata.location is None

    def test_single_tick_without_needs(self):
        """Single-tick stepping works without needs system."""
        config = _make_tick_config(
            needs_config={"enabled": False},
        )
        engine = TickEngine(config)
        engine.population = engine._create_initial_population()
        for ext in engine.extensions.get_enabled():
            ext.on_simulation_start(engine.population, config)

        log = engine._run_single_tick()
        assert isinstance(log, TickActivityLog)
        # No gathering activities when needs disabled
        for ata in log.agent_activities.values():
            assert ata.activity is None


# ---------------------------------------------------------------------------
# Session manager step_tick tests
# ---------------------------------------------------------------------------

class TestSessionManagerStepTick:
    def _make_session_manager(self, tick_enabled=True):
        from seldon.api.sessions import SessionManager
        mgr = SessionManager(db_path=None)
        config = ExperimentConfig(
            tick_config={"enabled": tick_enabled},
            needs_config={"enabled": True},
            hex_grid_config={"enabled": True, "width": 6, "height": 4},
            initial_population=8,
            generations_to_run=3,
            random_seed=99,
        )
        session = mgr.create_session(config=config, name="test-world")
        return mgr, session.id

    def test_step_tick_returns_dict(self):
        mgr, sid = self._make_session_manager()
        result = mgr.step_tick(sid)
        assert isinstance(result, dict)
        assert result["enabled"] is True
        assert "year" in result
        assert "tick_in_year" in result
        assert "season" in result
        assert "agent_activities" in result
        assert "events" in result
        assert "agent_names" in result

    def test_step_tick_advances(self):
        mgr, sid = self._make_session_manager()
        r1 = mgr.step_tick(sid)
        assert r1["global_tick"] == 0  # First tick is tick 0
        r2 = mgr.step_tick(sid)
        assert r2["global_tick"] == 1  # Second tick is tick 1

    def test_step_tick_year_complete(self):
        mgr, sid = self._make_session_manager()
        for i in range(11):
            r = mgr.step_tick(sid)
            assert r["year_complete"] is False
        r = mgr.step_tick(sid)
        assert r["year_complete"] is True

    def test_step_tick_updates_generation(self):
        mgr, sid = self._make_session_manager()
        session = mgr.get_session(sid)
        assert session.current_generation == 0
        for _ in range(12):
            mgr.step_tick(sid)
        session = mgr.get_session(sid)
        assert session.current_generation == 1

    def test_step_tick_rejects_non_tick_engine(self):
        mgr, sid = self._make_session_manager(tick_enabled=False)
        with pytest.raises(ValueError, match="tick engine"):
            mgr.step_tick(sid)

    def test_step_tick_session_not_found(self):
        mgr, _ = self._make_session_manager()
        with pytest.raises(KeyError):
            mgr.step_tick("nonexistent")

    def test_step_tick_completed_session(self):
        mgr, sid = self._make_session_manager()
        # Run to completion
        session = mgr.get_session(sid)
        session.status = "completed"
        result = mgr.step_tick(sid)
        # Should return current state without advancing
        assert result["enabled"] is True

    def test_agent_activities_have_data(self):
        mgr, sid = self._make_session_manager()
        result = mgr.step_tick(sid)
        for aid, ata in result["agent_activities"].items():
            assert "agent_id" in ata
            assert "location" in ata
            assert "processing_region" in ata
            assert "needs_snapshot" in ata
            assert "health" in ata

    def test_agent_names_populated(self):
        mgr, sid = self._make_session_manager()
        result = mgr.step_tick(sid)
        assert len(result["agent_names"]) > 0
        for aid, name in result["agent_names"].items():
            assert isinstance(name, str)
            assert len(name) > 0

    def test_tick_state_endpoint(self):
        mgr, sid = self._make_session_manager()
        # Before any ticks, should return minimal state
        session = mgr.get_session(sid)
        result = mgr._build_tick_response(session)
        assert result["enabled"] is True

        # After stepping, should return last tick
        mgr.step_tick(sid)
        result = mgr._build_tick_response(session)
        assert result["enabled"] is True


# ---------------------------------------------------------------------------
# API endpoint tests
# ---------------------------------------------------------------------------

class TestHexGridWorldAPI:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from seldon.api.app import app
        return TestClient(app)

    @pytest.fixture
    def tick_session(self, client):
        resp = client.post("/api/simulation/sessions", json={
            "name": "world-test",
            "config": {
                "tick_config": {"enabled": True},
                "needs_config": {"enabled": True},
                "hex_grid_config": {"enabled": True, "width": 6, "height": 4},
                "initial_population": 8,
                "generations_to_run": 3,
                "random_seed": 77,
            },
        })
        assert resp.status_code == 200
        return resp.json()["id"]

    def test_step_tick_endpoint(self, client, tick_session):
        resp = client.post(f"/api/hex/{tick_session}/step-tick")
        assert resp.status_code == 200
        data = resp.json()
        assert data["enabled"] is True
        assert data["year"] == 0
        assert data["tick_in_year"] == 0
        assert "agent_activities" in data

    def test_tick_state_endpoint(self, client, tick_session):
        resp = client.get(f"/api/hex/{tick_session}/tick-state")
        assert resp.status_code == 200
        data = resp.json()
        assert data["enabled"] is True

    def test_step_tick_session_not_found(self, client):
        resp = client.post("/api/hex/nonexistent/step-tick")
        assert resp.status_code == 404

    def test_tick_state_non_tick_session(self, client):
        # Create non-tick session
        resp = client.post("/api/simulation/sessions", json={
            "name": "no-tick",
            "config": {"tick_config": {"enabled": False}},
        })
        sid = resp.json()["id"]
        resp = client.get(f"/api/hex/{sid}/tick-state")
        assert resp.status_code == 200
        assert resp.json()["enabled"] is False

    def test_step_tick_non_tick_session(self, client):
        resp = client.post("/api/simulation/sessions", json={
            "name": "no-tick2",
            "config": {"tick_config": {"enabled": False}},
        })
        sid = resp.json()["id"]
        resp = client.post(f"/api/hex/{sid}/step-tick")
        assert resp.status_code == 400

    def test_step_tick_12_times_completes_year(self, client, tick_session):
        for i in range(12):
            resp = client.post(f"/api/hex/{tick_session}/step-tick")
            assert resp.status_code == 200
            data = resp.json()
            if i < 11:
                assert data["year_complete"] is False
            else:
                assert data["year_complete"] is True
                assert data["current_generation"] == 1
