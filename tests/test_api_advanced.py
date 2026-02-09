"""Tests for advanced API endpoints: anomaly, lore, settlements, network, sensitivity."""

import pytest
from fastapi.testclient import TestClient

from seldon.api.app import create_app


@pytest.fixture
def client():
    app = create_app()
    return TestClient(app)


def _create_and_step(client, n=5, config=None):
    """Helper: create session, step N generations, return session_id."""
    cfg = config or {"initial_population": 20, "generations_to_run": 20, "random_seed": 42}
    resp = client.post("/api/simulation/sessions", json={"config": cfg})
    sid = resp.json()["id"]
    client.post(f"/api/simulation/sessions/{sid}/step", json={"n": n})
    return sid


def _create_with_extensions(client, n=5):
    """Create a session with geography + migration extensions enabled."""
    cfg = {
        "initial_population": 30,
        "generations_to_run": 20,
        "random_seed": 42,
        "extensions_enabled": ["geography", "migration"],
    }
    resp = client.post("/api/simulation/sessions", json={"config": cfg})
    sid = resp.json()["id"]
    client.post(f"/api/simulation/sessions/{sid}/step", json={"n": n})
    return sid


# ===========================================================================
# Anomaly Detection
# ===========================================================================

class TestAnomalyDetection:
    def test_anomalies_with_data(self, client):
        sid = _create_and_step(client, 10)
        resp = client.get(f"/api/advanced/{sid}/anomalies")
        assert resp.status_code == 200
        data = resp.json()
        assert "anomalies" in data
        assert "generation_scores" in data
        assert "thresholds" in data
        assert len(data["generation_scores"]) == 10
        assert data["thresholds"]["anomaly"] == 2.0

    def test_anomalies_few_generations(self, client):
        sid = _create_and_step(client, 2)
        resp = client.get(f"/api/advanced/{sid}/anomalies")
        assert resp.status_code == 200
        data = resp.json()
        assert data["anomalies"] == []
        assert data["generation_scores"] == []

    def test_anomalies_not_found(self, client):
        resp = client.get("/api/advanced/nonexistent/anomalies")
        assert resp.status_code == 404

    def test_anomaly_fields(self, client):
        sid = _create_and_step(client, 10)
        resp = client.get(f"/api/advanced/{sid}/anomalies")
        data = resp.json()
        for anomaly in data["anomalies"]:
            assert "generation" in anomaly
            assert "severity" in anomaly
            assert anomaly["severity"] in ("medium", "high", "critical")
            assert "z_score" in anomaly
            assert "metric" in anomaly
            assert "category" in anomaly
            assert "description" in anomaly


# ===========================================================================
# Lore Endpoints
# ===========================================================================

class TestLoreEndpoints:
    def test_lore_overview(self, client):
        sid = _create_and_step(client, 5)
        resp = client.get(f"/api/advanced/{sid}/lore/overview")
        assert resp.status_code == 200
        data = resp.json()
        assert "time_series" in data
        assert "current_societal_lore" in data
        assert "memory_type_distribution" in data
        ts = data["time_series"]
        assert len(ts["generations"]) == 5
        assert len(ts["total_memories"]) == 5

    def test_lore_overview_not_found(self, client):
        resp = client.get("/api/advanced/nonexistent/lore/overview")
        assert resp.status_code == 404

    def test_meme_prevalence_disabled(self, client):
        sid = _create_and_step(client, 5)
        resp = client.get(f"/api/advanced/{sid}/lore/meme-prevalence")
        assert resp.status_code == 200
        data = resp.json()
        # Without culture extension, should report disabled
        assert data["enabled"] is False

    def test_meme_prevalence_with_culture(self, client):
        cfg = {
            "initial_population": 20,
            "generations_to_run": 20,
            "random_seed": 42,
            "extensions_enabled": ["culture"],
        }
        resp = client.post("/api/simulation/sessions", json={"config": cfg})
        sid = resp.json()["id"]
        client.post(f"/api/simulation/sessions/{sid}/step", json={"n": 5})

        resp = client.get(f"/api/advanced/{sid}/lore/meme-prevalence")
        assert resp.status_code == 200
        data = resp.json()
        assert data["enabled"] is True
        assert "prevalence_over_time" in data
        assert "memes" in data


# ===========================================================================
# Settlements
# ===========================================================================

class TestSettlements:
    def test_overview_disabled(self, client):
        sid = _create_and_step(client, 3)
        resp = client.get(f"/api/settlements/{sid}/overview")
        assert resp.status_code == 200
        data = resp.json()
        assert data["enabled"] is False

    def test_overview_enabled(self, client):
        sid = _create_with_extensions(client, 3)
        resp = client.get(f"/api/settlements/{sid}/overview")
        assert resp.status_code == 200
        data = resp.json()
        assert data["enabled"] is True
        assert "settlements" in data
        assert len(data["settlements"]) >= 1
        assert "total_capacity" in data
        assert "total_population" in data

    def test_settlement_fields(self, client):
        sid = _create_with_extensions(client, 3)
        resp = client.get(f"/api/settlements/{sid}/overview")
        data = resp.json()
        for s in data["settlements"]:
            assert "id" in s
            assert "name" in s
            assert "population" in s
            assert "carrying_capacity" in s
            assert "occupancy_ratio" in s
            assert "region_counts" in s
            assert "coordinates" in s

    def test_viability(self, client):
        sid = _create_with_extensions(client, 3)
        # Get first location
        overview = client.get(f"/api/settlements/{sid}/overview").json()
        loc_id = overview["settlements"][0]["id"]

        resp = client.get(f"/api/settlements/{sid}/viability/{loc_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert "viability_score" in data
        assert "risk_factors" in data
        assert "checks_passed" in data
        assert "checks_total" in data
        assert data["checks_total"] == 7

    def test_viability_no_migration(self, client):
        # Session with geography only, no migration
        cfg = {
            "initial_population": 20,
            "generations_to_run": 10,
            "random_seed": 42,
            "extensions_enabled": ["geography"],
        }
        resp = client.post("/api/simulation/sessions", json={"config": cfg})
        sid = resp.json()["id"]
        client.post(f"/api/simulation/sessions/{sid}/step", json={"n": 2})

        overview = client.get(f"/api/settlements/{sid}/overview").json()
        loc_id = overview["settlements"][0]["id"]
        resp = client.get(f"/api/settlements/{sid}/viability/{loc_id}")
        assert resp.status_code == 400  # migration not enabled

    def test_migration_history_disabled(self, client):
        sid = _create_and_step(client, 3)
        resp = client.get(f"/api/settlements/{sid}/migration-history")
        assert resp.status_code == 200
        data = resp.json()
        assert data["enabled"] is False

    def test_migration_history_enabled(self, client):
        sid = _create_with_extensions(client, 5)
        resp = client.get(f"/api/settlements/{sid}/migration-history")
        assert resp.status_code == 200
        data = resp.json()
        assert data["enabled"] is True
        assert "events" in data
        assert "timeline" in data

    def test_settlement_composition(self, client):
        sid = _create_with_extensions(client, 3)
        overview = client.get(f"/api/settlements/{sid}/overview").json()
        loc_id = overview["settlements"][0]["id"]

        resp = client.get(f"/api/settlements/{sid}/settlement-composition/{loc_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert "trait_means" in data
        assert "region_percentages" in data
        assert "mean_age" in data
        assert data["population"] > 0

    def test_settlement_not_found(self, client):
        sid = _create_with_extensions(client, 3)
        resp = client.get(f"/api/settlements/{sid}/settlement-composition/nonexistent")
        assert resp.status_code == 404


# ===========================================================================
# Network
# ===========================================================================

class TestNetwork:
    def test_graph_basic(self, client):
        sid = _create_and_step(client, 5)
        resp = client.get(f"/api/network/{sid}/graph")
        assert resp.status_code == 200
        data = resp.json()
        assert "nodes" in data
        assert "edges" in data
        assert "stats" in data
        assert len(data["nodes"]) > 0
        assert data["stats"]["total_nodes"] > 0

    def test_graph_with_threshold(self, client):
        sid = _create_and_step(client, 5)
        resp_low = client.get(f"/api/network/{sid}/graph?bond_threshold=0.0")
        resp_high = client.get(f"/api/network/{sid}/graph?bond_threshold=0.9")
        assert resp_low.status_code == 200
        assert resp_high.status_code == 200
        # Higher threshold should have <= edges
        assert len(resp_high.json()["edges"]) <= len(resp_low.json()["edges"])

    def test_graph_node_fields(self, client):
        sid = _create_and_step(client, 3)
        resp = client.get(f"/api/network/{sid}/graph")
        data = resp.json()
        for node in data["nodes"]:
            assert "id" in node
            assert "name" in node
            assert "region" in node

    def test_graph_stats(self, client):
        sid = _create_and_step(client, 5)
        resp = client.get(f"/api/network/{sid}/graph")
        stats = resp.json()["stats"]
        assert "total_nodes" in stats
        assert "total_edges" in stats
        assert "avg_connections" in stats
        assert "connected_components" in stats
        assert stats["connected_components"] >= 1

    def test_graph_not_found(self, client):
        resp = client.get("/api/network/nonexistent/graph")
        assert resp.status_code == 404


# ===========================================================================
# Sensitivity
# ===========================================================================

class TestSensitivity:
    def test_sensitivity_basic(self, client):
        # Create two sessions with different configs
        r1 = client.post("/api/simulation/sessions", json={
            "config": {"initial_population": 20, "generations_to_run": 5, "random_seed": 42},
        })
        sid1 = r1.json()["id"]
        client.post(f"/api/simulation/sessions/{sid1}/run", json={})

        r2 = client.post("/api/simulation/sessions", json={
            "config": {"initial_population": 20, "generations_to_run": 5, "random_seed": 42, "trait_drift_rate": 0.1},
        })
        sid2 = r2.json()["id"]
        client.post(f"/api/simulation/sessions/{sid2}/run", json={})

        resp = client.post(f"/api/advanced/{sid1}/sensitivity", json={
            "session_ids": [sid1, sid2],
            "target_metric": "mean_contribution",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["target_metric"] == "mean_contribution"
        assert data["session_count"] == 2
        assert "sensitivities" in data
        assert "tornado_data" in data
        # trait_drift_rate differs so should appear
        params = [s["parameter"] for s in data["sensitivities"]]
        assert "trait_drift_rate" in params

    def test_sensitivity_insufficient_sessions(self, client):
        r1 = client.post("/api/simulation/sessions", json={
            "config": {"initial_population": 20, "generations_to_run": 3, "random_seed": 42},
        })
        sid1 = r1.json()["id"]
        client.post(f"/api/simulation/sessions/{sid1}/run", json={})

        resp = client.post(f"/api/advanced/{sid1}/sensitivity", json={
            "session_ids": [sid1],
            "target_metric": "mean_contribution",
        })
        assert resp.status_code == 400

    def test_sensitivity_not_found(self, client):
        resp = client.post("/api/advanced/nonexistent/sensitivity", json={
            "session_ids": ["nonexistent", "also_nonexistent"],
        })
        assert resp.status_code == 404
