"""
Tests for SQLite session persistence.

Tests agent roundtrip serialization, SessionStore CRUD, state blob
compress/decompress, and full integration (create → step → "restart" →
verify data intact).
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from seldon.core.agent import Agent
from seldon.core.config import ExperimentConfig
from seldon.core.processing import ProcessingRegion
from seldon.metrics.collector import GenerationMetrics

from seldon.api.persistence import (
    SessionStore,
    build_state_blob,
    compress_state,
    decompress_state,
    deserialize_agent,
    deserialize_metrics,
    restore_state,
    serialize_agent,
    serialize_metrics,
)


# =====================================================================
# Helpers
# =====================================================================

def _make_config(**overrides) -> ExperimentConfig:
    defaults = {"random_seed": 42, "initial_population": 20, "generations_to_run": 5}
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
        traits = ts.random_traits(np.random.default_rng(42))
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


def _make_metrics(generation: int = 0, n_traits: int = 15) -> GenerationMetrics:
    """Create a sample GenerationMetrics for testing."""
    return GenerationMetrics(
        generation=generation,
        population_size=50,
        births=5,
        deaths=3,
        breakthroughs=1,
        pairs_formed=4,
        trait_means=np.random.default_rng(42).random(n_traits),
        trait_stds=np.random.default_rng(43).random(n_traits) * 0.2,
        trait_entropy=2.5,
        region_counts={"optimal": 30, "deep": 10, "sacrificial": 5, "under_processing": 3, "pathological": 2},
        region_fractions={"optimal": 0.6, "deep": 0.2, "sacrificial": 0.1, "under_processing": 0.06, "pathological": 0.04},
        region_transitions={"optimal->deep": 2, "deep->sacrificial": 1},
        total_contribution=125.5,
        mean_contribution=2.51,
        max_contribution=8.3,
        mean_suffering=0.35,
        suffering_by_region={"optimal": 0.1, "deep": 0.4, "sacrificial": 0.8, "under_processing": 0.05, "pathological": 0.9},
        mean_age=28.5,
        age_distribution={"0-15": 10, "16-30": 20, "31-50": 15, "51+": 5},
        birth_order_counts={1: 15, 2: 12, 3: 10, 4: 8, 5: 5},
        total_memories=200,
        societal_memories=15,
        myths_count=3,
        outsider_count=2,
        outsider_descendant_count=5,
        dissolutions=1,
        infidelity_events=0,
        outsiders_injected=1,
        dominant_voice_counts={"analytical": 20, "creative": 15, "empathetic": 10},
        extension_metrics={"geography_settlements": 3, "technology_level": 2.1},
    )


def _tmp_db_path():
    """Return a temporary file path for a test database."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    return path


# =====================================================================
# Agent Serialization Tests
# =====================================================================

class TestAgentSerialization:
    """Test agent serialize/deserialize roundtrip."""

    def test_basic_roundtrip(self):
        """Agent survives serialize→deserialize with all fields intact."""
        agent = _make_agent()
        agent.processing_region = ProcessingRegion.DEEP
        agent.suffering = 0.75
        agent.burnout_level = 0.3
        agent.partner_id = "p1"
        agent.parent1_id = "parent_a"
        agent.parent2_id = "parent_b"
        agent.children_ids = ["c1", "c2"]
        agent.relationship_status = "paired"

        d = serialize_agent(agent)
        restored = deserialize_agent(d)

        assert restored.id == agent.id
        assert restored.name == agent.name
        assert restored.age == agent.age
        assert restored.generation == agent.generation
        assert restored.birth_order == agent.birth_order
        np.testing.assert_array_almost_equal(restored.traits, agent.traits)
        np.testing.assert_array_almost_equal(restored.traits_at_birth, agent.traits_at_birth)
        assert restored.processing_region == ProcessingRegion.DEEP
        assert restored.suffering == pytest.approx(0.75)
        assert restored.burnout_level == pytest.approx(0.3)
        assert restored.partner_id == "p1"
        assert restored.parent1_id == "parent_a"
        assert restored.parent2_id == "parent_b"
        assert restored.children_ids == ["c1", "c2"]
        assert restored.relationship_status == "paired"

    def test_history_roundtrip(self):
        """Agent history (traits, regions, contributions, suffering) survives roundtrip."""
        agent = _make_agent()
        # Simulate a few generations of history
        for _ in range(3):
            agent.record_generation(contribution=float(np.random.default_rng(42).random()))
            agent.age += 1

        d = serialize_agent(agent)
        restored = deserialize_agent(d)

        assert len(restored.trait_history) == 3
        assert len(restored.region_history) == 3
        assert len(restored.contribution_history) == 3
        assert len(restored.suffering_history) == 3
        for orig, rest in zip(agent.trait_history, restored.trait_history):
            np.testing.assert_array_almost_equal(orig, rest)

    def test_genetics_roundtrip(self):
        """Genome and epigenetic state survive roundtrip."""
        agent = _make_agent()
        agent.genome = {"CREA_1": ("A", "a"), "RESI_1": ("a", "a")}
        agent.epigenetic_state = {"stress_resilience": True, "creative_amplification": False}
        agent.genetic_lineage = {"parent1": {"CREA_1": ("A", "A")}, "parent2": {"CREA_1": ("a", "a")}}

        d = serialize_agent(agent)
        restored = deserialize_agent(d)

        assert restored.genome["CREA_1"] == ("A", "a")
        assert restored.genome["RESI_1"] == ("a", "a")
        assert restored.epigenetic_state["stress_resilience"] is True
        assert restored.epigenetic_state["creative_amplification"] is False

    def test_social_fields_roundtrip(self):
        """Social hierarchy fields survive roundtrip."""
        agent = _make_agent()
        agent.social_status = 0.85
        agent.social_role = "leader"
        agent.influence_score = 0.92
        agent.mentor_id = "m1"
        agent.mentee_ids = ["me1", "me2"]
        agent.social_bonds = {"b1": 0.7, "b2": 0.3}

        d = serialize_agent(agent)
        restored = deserialize_agent(d)

        assert restored.social_status == pytest.approx(0.85)
        assert restored.social_role == "leader"
        assert restored.influence_score == pytest.approx(0.92)
        assert restored.mentor_id == "m1"
        assert restored.mentee_ids == ["me1", "me2"]
        assert restored.social_bonds["b1"] == pytest.approx(0.7)

    def test_outsider_fields_roundtrip(self):
        """Outsider tracking fields survive roundtrip."""
        agent = _make_agent()
        agent.is_outsider = True
        agent.outsider_origin = "da_vinci"
        agent.injection_generation = 5
        agent.gender = "female"

        d = serialize_agent(agent)
        restored = deserialize_agent(d)

        assert restored.is_outsider is True
        assert restored.outsider_origin == "da_vinci"
        assert restored.injection_generation == 5
        assert restored.gender == "female"

    def test_economics_fields_roundtrip(self):
        """Economics fields survive roundtrip."""
        agent = _make_agent()
        agent.wealth = 150.5
        agent.occupation = "farmer"
        agent.trade_history = [{"gen": 1, "amount": 10.0}]

        d = serialize_agent(agent)
        restored = deserialize_agent(d)

        assert restored.wealth == pytest.approx(150.5)
        assert restored.occupation == "farmer"
        assert len(restored.trade_history) == 1

    def test_extension_data_roundtrip(self):
        """Extension hooks data survives roundtrip."""
        agent = _make_agent()
        agent.location_id = "settlement_1"
        agent.resource_holdings = {"food": 10.0, "wood": 5.0}
        agent.cultural_memes = ["tortured_genius", "community_spirit"]
        agent.skills = {"farming": 0.8, "crafting": 0.3}
        agent.extension_data = {"custom_key": "custom_value"}

        d = serialize_agent(agent)
        restored = deserialize_agent(d)

        assert restored.location_id == "settlement_1"
        assert restored.resource_holdings["food"] == pytest.approx(10.0)
        assert "tortured_genius" in restored.cultural_memes
        assert restored.skills["farming"] == pytest.approx(0.8)
        assert restored.extension_data["custom_key"] == "custom_value"

    def test_dead_agent_roundtrip(self):
        """Dead agents preserve is_alive=False."""
        agent = _make_agent()
        agent.is_alive = False

        d = serialize_agent(agent)
        restored = deserialize_agent(d)

        assert restored.is_alive is False

    def test_numpy_int_fields(self):
        """numpy int64 fields are properly converted."""
        agent = _make_agent()
        agent.age = np.int64(30)
        agent.generation = np.int64(3)
        agent.birth_order = np.int64(2)
        agent.injection_generation = np.int64(5)

        d = serialize_agent(agent)
        # Should be plain Python ints in the dict
        assert isinstance(d["age"], int)
        assert isinstance(d["generation"], int)
        assert isinstance(d["birth_order"], int)
        assert isinstance(d["injection_generation"], int)

        restored = deserialize_agent(d)
        assert restored.age == 30
        assert restored.injection_generation == 5


# =====================================================================
# Metrics Serialization Tests
# =====================================================================

class TestMetricsSerialization:
    """Test GenerationMetrics serialize/deserialize roundtrip."""

    def test_roundtrip(self):
        """GenerationMetrics survives serialize→deserialize."""
        m = _make_metrics()
        d = serialize_metrics(m)
        restored = deserialize_metrics(d)

        assert restored.generation == m.generation
        assert restored.population_size == m.population_size
        assert restored.births == m.births
        assert restored.deaths == m.deaths
        assert restored.breakthroughs == m.breakthroughs
        np.testing.assert_array_almost_equal(restored.trait_means, m.trait_means)
        np.testing.assert_array_almost_equal(restored.trait_stds, m.trait_stds)
        assert restored.trait_entropy == pytest.approx(m.trait_entropy)
        assert restored.region_counts == m.region_counts
        assert restored.total_contribution == pytest.approx(m.total_contribution)
        assert restored.extension_metrics == m.extension_metrics
        assert restored.dominant_voice_counts == m.dominant_voice_counts

    def test_birth_order_int_keys(self):
        """birth_order_counts keys are restored as ints."""
        m = _make_metrics()
        d = serialize_metrics(m)
        restored = deserialize_metrics(d)

        for k in restored.birth_order_counts:
            assert isinstance(k, int)


# =====================================================================
# State Blob Compression Tests
# =====================================================================

class TestStateBlob:
    """Test state blob compress/decompress."""

    def test_compress_decompress_roundtrip(self):
        """Data survives compress→decompress."""
        data = {"key": "value", "numbers": [1, 2, 3], "nested": {"a": 1.5}}
        blob = compress_state(data)
        assert isinstance(blob, bytes)
        restored = decompress_state(blob)
        assert restored == data

    def test_build_and_restore_state(self):
        """Full state blob with agents and metrics survives roundtrip."""
        config = _make_config()
        agents = {f"a{i}": _make_agent(f"a{i}", config=config) for i in range(5)}
        living_ids = ["a0", "a1", "a2"]
        metrics = [_make_metrics(g) for g in range(3)]
        prev_regions = {"a0": "optimal", "a1": "deep"}

        blob = build_state_blob(agents, living_ids, metrics, 100, prev_regions)
        state = restore_state(blob)

        assert len(state["all_agents"]) == 5
        assert state["living_agent_ids"] == living_ids
        assert len(state["metrics_history"]) == 3
        assert state["next_agent_id"] == 100
        assert state["previous_regions"] == prev_regions

        # Verify agent data integrity
        restored_a0 = state["all_agents"]["a0"]
        np.testing.assert_array_almost_equal(restored_a0.traits, agents["a0"].traits)

    def test_compression_reduces_size(self):
        """Compressed blob is smaller than raw JSON."""
        import json
        from seldon.api.persistence import _json_fallback

        config = _make_config()
        agents = {f"a{i}": _make_agent(f"a{i}", config=config) for i in range(20)}
        state = {
            "all_agents": {aid: serialize_agent(a) for aid, a in agents.items()},
            "living_agent_ids": list(agents.keys()),
            "metrics_history": [],
            "next_agent_id": 20,
            "previous_regions": {},
        }

        raw_json = json.dumps(state, default=_json_fallback).encode("utf-8")
        blob = compress_state(state)

        assert len(blob) < len(raw_json)


# =====================================================================
# SessionStore Tests
# =====================================================================

class TestSessionStore:
    """Test SQLite SessionStore CRUD operations."""

    def test_create_and_list(self):
        """Sessions can be saved and listed."""
        db_path = _tmp_db_path()
        try:
            store = SessionStore(db_path)
            config = _make_config()

            blob = compress_state({"test": True})
            store.save_session(
                session_id="s1", name="Test Session", status="created",
                current_generation=0, max_generations=10,
                population_size=50, config=config, state_blob=blob,
            )

            sessions = store.list_sessions()
            assert len(sessions) == 1
            assert sessions[0]["id"] == "s1"
            assert sessions[0]["name"] == "Test Session"
            assert sessions[0]["status"] == "created"
            assert sessions[0]["population_size"] == 50

            store.close()
        finally:
            os.unlink(db_path)

    def test_save_updates_existing(self):
        """Saving with same ID updates the record."""
        db_path = _tmp_db_path()
        try:
            store = SessionStore(db_path)
            config = _make_config()
            blob = compress_state({"v": 1})

            store.save_session(
                "s1", "Session V1", "created", 0, 10, 50, config, blob,
            )
            blob2 = compress_state({"v": 2})
            store.save_session(
                "s1", "Session V2", "running", 5, 10, 45, config, blob2,
            )

            sessions = store.list_sessions()
            assert len(sessions) == 1
            assert sessions[0]["name"] == "Session V2"
            assert sessions[0]["status"] == "running"
            assert sessions[0]["current_generation"] == 5

            store.close()
        finally:
            os.unlink(db_path)

    def test_load_session(self):
        """Full session can be loaded back with config and state."""
        db_path = _tmp_db_path()
        try:
            store = SessionStore(db_path)
            config = _make_config(experiment_name="load_test")
            blob = compress_state({"agents": {"a1": "data"}})

            store.save_session(
                "s1", "Load Test", "running", 3, 10, 40, config, blob,
            )

            loaded = store.load_session("s1")
            assert loaded is not None
            assert loaded["id"] == "s1"
            assert loaded["status"] == "running"
            assert loaded["current_generation"] == 3

            # Config should be parseable
            restored_config = ExperimentConfig.from_json(loaded["config_json"])
            assert restored_config.experiment_name == "load_test"

            # State blob should be decompressable
            state = decompress_state(loaded["state_blob"])
            assert state["agents"]["a1"] == "data"

            store.close()
        finally:
            os.unlink(db_path)

    def test_load_nonexistent_returns_none(self):
        """Loading a non-existent session returns None."""
        db_path = _tmp_db_path()
        try:
            store = SessionStore(db_path)
            assert store.load_session("nonexistent") is None
            store.close()
        finally:
            os.unlink(db_path)

    def test_delete_session(self):
        """Deleted sessions are removed from the database."""
        db_path = _tmp_db_path()
        try:
            store = SessionStore(db_path)
            config = _make_config()
            blob = compress_state({"test": True})

            store.save_session("s1", "Delete Me", "created", 0, 10, 50, config, blob)
            store.save_session("s2", "Keep Me", "created", 0, 10, 50, config, blob)

            store.delete_session("s1")

            sessions = store.list_sessions()
            assert len(sessions) == 1
            assert sessions[0]["id"] == "s2"

            assert not store.has_session("s1")
            assert store.has_session("s2")

            store.close()
        finally:
            os.unlink(db_path)

    def test_has_session(self):
        """has_session correctly reports existence."""
        db_path = _tmp_db_path()
        try:
            store = SessionStore(db_path)
            config = _make_config()
            blob = compress_state({})

            assert not store.has_session("s1")
            store.save_session("s1", "Test", "created", 0, 10, 50, config, blob)
            assert store.has_session("s1")

            store.close()
        finally:
            os.unlink(db_path)

    def test_multiple_sessions(self):
        """Multiple sessions can coexist."""
        db_path = _tmp_db_path()
        try:
            store = SessionStore(db_path)
            config = _make_config()
            blob = compress_state({})

            for i in range(5):
                store.save_session(f"s{i}", f"Session {i}", "created", 0, 10, 50, config, blob)

            sessions = store.list_sessions()
            assert len(sessions) == 5

            store.close()
        finally:
            os.unlink(db_path)

    def test_persistence_across_connections(self):
        """Data persists when store is closed and reopened."""
        db_path = _tmp_db_path()
        try:
            # Write with first connection
            store1 = SessionStore(db_path)
            config = _make_config()
            blob = compress_state({"key": "value"})
            store1.save_session("s1", "Persist Test", "running", 3, 10, 45, config, blob)
            store1.close()

            # Read with second connection
            store2 = SessionStore(db_path)
            sessions = store2.list_sessions()
            assert len(sessions) == 1
            assert sessions[0]["id"] == "s1"
            assert sessions[0]["current_generation"] == 3

            loaded = store2.load_session("s1")
            assert loaded is not None
            state = decompress_state(loaded["state_blob"])
            assert state["key"] == "value"

            store2.close()
        finally:
            os.unlink(db_path)

    def test_unavailable_store_degrades_gracefully(self):
        """When DB can't be created, all operations return empty/None."""
        store = SessionStore("/nonexistent/path/db.sqlite")
        assert not store.available

        # All operations should work without raising
        assert store.list_sessions() == []
        assert store.load_session("s1") is None
        assert not store.has_session("s1")

        # Write operations should not raise
        config = _make_config()
        store.save_session("s1", "Test", "created", 0, 10, 50, config, b"")
        store.delete_session("s1")
        store.close()


# =====================================================================
# Full Integration Tests
# =====================================================================

class TestPersistenceIntegration:
    """Integration tests using SessionManager with persistence."""

    def test_session_create_step_restart_roundtrip(self):
        """Create a session, step it, 'restart' (new store), verify data intact."""
        from seldon.api.sessions import SessionManager

        db_path = _tmp_db_path()
        try:
            # --- Phase 1: Create and step ---
            mgr1 = SessionManager(db_path=db_path)
            config = _make_config(initial_population=20, generations_to_run=10)
            session = mgr1.create_session(config=config, name="Integration Test")
            sid = session.id

            # Step 3 generations
            mgr1.step(sid, 3)
            session = mgr1.get_session(sid)
            assert session.current_generation == 3
            assert len(session.collector.metrics_history) == 3

            original_agent_count = len(session.all_agents)
            original_pop_size = len(session.engine.population)
            original_first_metric_gen = session.collector.metrics_history[0].generation

            # Get a sample agent for comparison
            sample_agent_id = next(iter(session.all_agents))
            original_agent = session.all_agents[sample_agent_id]
            original_traits = original_agent.traits.copy()

            mgr1.close()

            # --- Phase 2: "Restart" — new manager, same DB ---
            mgr2 = SessionManager(db_path=db_path)

            # Session should appear in listing
            listing = mgr2.list_sessions()
            assert any(s["id"] == sid for s in listing)

            # Lazy load: session not in memory yet
            assert sid not in mgr2.sessions

            # Get session triggers load
            restored = mgr2.get_session(sid)
            assert restored.current_generation == 3
            assert restored.status == session.status
            assert len(restored.all_agents) == original_agent_count
            assert len(restored.engine.population) == original_pop_size
            assert len(restored.collector.metrics_history) == 3
            assert restored.collector.metrics_history[0].generation == original_first_metric_gen

            # Verify agent data
            restored_agent = restored.all_agents[sample_agent_id]
            np.testing.assert_array_almost_equal(restored_agent.traits, original_traits)

            # Can continue stepping
            mgr2.step(sid, 2)
            assert mgr2.get_session(sid).current_generation == 5

            mgr2.close()
        finally:
            os.unlink(db_path)

    def test_delete_persists(self):
        """Deleted sessions don't come back after restart."""
        from seldon.api.sessions import SessionManager

        db_path = _tmp_db_path()
        try:
            mgr1 = SessionManager(db_path=db_path)
            config = _make_config(initial_population=10, generations_to_run=5)
            s1 = mgr1.create_session(config=config, name="Keep")
            s2 = mgr1.create_session(config=config, name="Delete")

            mgr1.delete_session(s2.id)
            mgr1.close()

            # Restart
            mgr2 = SessionManager(db_path=db_path)
            listing = mgr2.list_sessions()
            ids = [s["id"] for s in listing]
            assert s1.id in ids
            assert s2.id not in ids

            mgr2.close()
        finally:
            os.unlink(db_path)

    def test_reset_persists(self):
        """Reset session state persists across restart."""
        from seldon.api.sessions import SessionManager

        db_path = _tmp_db_path()
        try:
            mgr1 = SessionManager(db_path=db_path)
            config = _make_config(initial_population=10, generations_to_run=5)
            session = mgr1.create_session(config=config, name="Reset Test")
            sid = session.id

            mgr1.step(sid, 3)
            mgr1.reset_session(sid)
            mgr1.close()

            mgr2 = SessionManager(db_path=db_path)
            restored = mgr2.get_session(sid)
            assert restored.current_generation == 0
            assert restored.status == "created"

            mgr2.close()
        finally:
            os.unlink(db_path)

    def test_multiple_sessions_persist(self):
        """Multiple sessions all survive restart."""
        from seldon.api.sessions import SessionManager

        db_path = _tmp_db_path()
        try:
            mgr1 = SessionManager(db_path=db_path)
            config = _make_config(initial_population=10, generations_to_run=5)

            sids = []
            for i in range(3):
                s = mgr1.create_session(config=config, name=f"Session {i}")
                mgr1.step(s.id, i + 1)  # Step 1, 2, 3 generations respectively
                sids.append(s.id)

            mgr1.close()

            mgr2 = SessionManager(db_path=db_path)
            listing = mgr2.list_sessions()
            assert len(listing) >= 3

            for i, sid in enumerate(sids):
                restored = mgr2.get_session(sid)
                assert restored.current_generation == i + 1

            mgr2.close()
        finally:
            os.unlink(db_path)

    def test_no_db_fallback(self):
        """SessionManager works without a database (in-memory only)."""
        from seldon.api.sessions import SessionManager

        mgr = SessionManager(db_path=None)
        config = _make_config(initial_population=10, generations_to_run=5)

        session = mgr.create_session(config=config, name="In-Memory")
        mgr.step(session.id, 2)

        listing = mgr.list_sessions()
        assert len(listing) == 1

        restored = mgr.get_session(session.id)
        assert restored.current_generation == 2

        mgr.close()
