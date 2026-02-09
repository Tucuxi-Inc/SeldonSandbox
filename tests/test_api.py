"""Integration tests for the Seldon Sandbox REST API."""

import pytest
from fastapi.testclient import TestClient

from seldon.api.app import create_app


@pytest.fixture
def client():
    app = create_app()
    return TestClient(app)


class TestHealthCheck:
    def test_health(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestSessionLifecycle:
    def test_create_session_defaults(self, client):
        resp = client.post("/api/simulation/sessions", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert "id" in data
        assert data["status"] == "created"
        assert data["current_generation"] == 0
        assert data["population_size"] == 100  # default

    def test_create_session_with_config(self, client):
        resp = client.post("/api/simulation/sessions", json={
            "config": {"initial_population": 30, "generations_to_run": 10, "random_seed": 42},
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["population_size"] == 30
        assert data["max_generations"] == 10

    def test_create_session_from_preset(self, client):
        resp = client.post("/api/simulation/sessions", json={"preset": "high_sacrificial"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["config"]["experiment_name"] == "high_sacrificial"

    def test_list_sessions(self, client):
        client.post("/api/simulation/sessions", json={"config": {"initial_population": 20}})
        client.post("/api/simulation/sessions", json={"config": {"initial_population": 20}})
        resp = client.get("/api/simulation/sessions")
        assert resp.status_code == 200
        assert len(resp.json()) >= 2

    def test_get_session(self, client):
        create_resp = client.post("/api/simulation/sessions", json={
            "config": {"initial_population": 20, "random_seed": 42},
        })
        sid = create_resp.json()["id"]
        resp = client.get(f"/api/simulation/sessions/{sid}")
        assert resp.status_code == 200
        assert resp.json()["id"] == sid

    def test_get_session_not_found(self, client):
        resp = client.get("/api/simulation/sessions/nonexistent")
        assert resp.status_code == 404

    def test_delete_session(self, client):
        create_resp = client.post("/api/simulation/sessions", json={
            "config": {"initial_population": 20},
        })
        sid = create_resp.json()["id"]
        resp = client.delete(f"/api/simulation/sessions/{sid}")
        assert resp.status_code == 200
        assert resp.json()["deleted"] is True

        resp = client.get(f"/api/simulation/sessions/{sid}")
        assert resp.status_code == 404


class TestStepAndRun:
    def test_step_one_generation(self, client):
        create_resp = client.post("/api/simulation/sessions", json={
            "config": {"initial_population": 20, "generations_to_run": 10, "random_seed": 42},
        })
        sid = create_resp.json()["id"]

        resp = client.post(f"/api/simulation/sessions/{sid}/step", json={"n": 1})
        assert resp.status_code == 200
        data = resp.json()
        assert data["current_generation"] == 1
        assert data["status"] == "running"

    def test_step_multiple(self, client):
        create_resp = client.post("/api/simulation/sessions", json={
            "config": {"initial_population": 20, "generations_to_run": 10, "random_seed": 42},
        })
        sid = create_resp.json()["id"]

        resp = client.post(f"/api/simulation/sessions/{sid}/step", json={"n": 5})
        data = resp.json()
        assert data["current_generation"] == 5

    def test_run_full(self, client):
        create_resp = client.post("/api/simulation/sessions", json={
            "config": {"initial_population": 20, "generations_to_run": 5, "random_seed": 42},
        })
        sid = create_resp.json()["id"]

        resp = client.post(f"/api/simulation/sessions/{sid}/run", json={})
        data = resp.json()
        assert data["current_generation"] == 5
        assert data["status"] == "completed"

    def test_reset(self, client):
        create_resp = client.post("/api/simulation/sessions", json={
            "config": {"initial_population": 20, "generations_to_run": 5, "random_seed": 42},
        })
        sid = create_resp.json()["id"]
        client.post(f"/api/simulation/sessions/{sid}/step", json={"n": 3})

        resp = client.post(f"/api/simulation/sessions/{sid}/reset")
        data = resp.json()
        assert data["current_generation"] == 0
        assert data["status"] == "created"


class TestMetrics:
    def _create_and_step(self, client, n=5):
        resp = client.post("/api/simulation/sessions", json={
            "config": {"initial_population": 20, "generations_to_run": 10, "random_seed": 42},
        })
        sid = resp.json()["id"]
        client.post(f"/api/simulation/sessions/{sid}/step", json={"n": n})
        return sid

    def test_get_generations(self, client):
        sid = self._create_and_step(client, 5)
        resp = client.get(f"/api/metrics/{sid}/generations")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 5
        assert data[0]["generation"] == 0
        assert "trait_means" in data[0]
        assert "region_counts" in data[0]

    def test_get_generations_range(self, client):
        sid = self._create_and_step(client, 5)
        resp = client.get(f"/api/metrics/{sid}/generations?from_gen=1&to_gen=3")
        data = resp.json()
        assert len(data) == 2

    def test_time_series(self, client):
        sid = self._create_and_step(client, 5)
        resp = client.get(f"/api/metrics/{sid}/time-series/population_size")
        assert resp.status_code == 200
        data = resp.json()
        assert data["field"] == "population_size"
        assert len(data["generations"]) == 5
        assert len(data["values"]) == 5

    def test_summary(self, client):
        sid = self._create_and_step(client, 5)
        resp = client.get(f"/api/metrics/{sid}/summary")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_generations"] == 5
        assert data["final_population_size"] > 0
        assert "mean_contribution" in data


class TestAgents:
    def _create_and_step(self, client, n=3):
        resp = client.post("/api/simulation/sessions", json={
            "config": {"initial_population": 20, "generations_to_run": 10, "random_seed": 42},
        })
        sid = resp.json()["id"]
        client.post(f"/api/simulation/sessions/{sid}/step", json={"n": n})
        return sid

    def test_list_agents(self, client):
        sid = self._create_and_step(client)
        resp = client.get(f"/api/agents/{sid}")
        assert resp.status_code == 200
        data = resp.json()
        assert "agents" in data
        assert data["total"] > 0
        assert data["page"] == 1

    def test_list_agents_all(self, client):
        sid = self._create_and_step(client, 5)
        resp = client.get(f"/api/agents/{sid}?alive_only=false")
        data = resp.json()
        # all_agents includes dead, so should be >= alive
        alive_resp = client.get(f"/api/agents/{sid}?alive_only=true")
        alive_data = alive_resp.json()
        assert data["total"] >= alive_data["total"]

    def test_get_agent_detail(self, client):
        sid = self._create_and_step(client)
        # Get first agent from list
        agents_resp = client.get(f"/api/agents/{sid}")
        agent_id = agents_resp.json()["agents"][0]["id"]

        resp = client.get(f"/api/agents/{sid}/{agent_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert "traits" in data
        assert "trait_history" in data
        assert "region_history" in data
        assert "contribution_history" in data

    def test_get_agent_not_found(self, client):
        sid = self._create_and_step(client)
        resp = client.get(f"/api/agents/{sid}/nonexistent")
        assert resp.status_code == 404

    def test_family_tree(self, client):
        sid = self._create_and_step(client, 5)
        # Get an agent who likely has parents (born in later gen)
        agents_resp = client.get(f"/api/agents/{sid}?alive_only=false")
        agents = agents_resp.json()["agents"]

        # Pick any agent
        agent_id = agents[0]["id"]
        resp = client.get(f"/api/agents/{sid}/{agent_id}/family-tree")
        assert resp.status_code == 200
        data = resp.json()
        assert "root" in data
        assert "ancestors" in data
        assert "descendants" in data

    def test_search_agents(self, client):
        sid = self._create_and_step(client)
        resp = client.get(f"/api/agents/{sid}?search=G0")
        data = resp.json()
        # All founding agents have "G0" in their name
        assert data["total"] > 0

    def test_filter_by_region(self, client):
        sid = self._create_and_step(client, 5)
        resp = client.get(f"/api/agents/{sid}?region=optimal")
        data = resp.json()
        for agent in data["agents"]:
            assert agent["processing_region"] == "optimal"


class TestExperiments:
    def test_list_presets(self, client):
        resp = client.get("/api/experiments/presets")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 10
        names = [p["name"] for p in data]
        assert "baseline" in names

    def test_list_archetypes(self, client):
        resp = client.get("/api/experiments/archetypes")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 11
        names = [a["name"] for a in data]
        assert "einstein" in names

    def test_get_archetype_detail(self, client):
        resp = client.get("/api/experiments/archetypes/einstein")
        assert resp.status_code == 200
        data = resp.json()
        assert "trait_values" in data
        assert data["display_name"] == "Einstein"

    def test_get_archetype_not_found(self, client):
        resp = client.get("/api/experiments/archetypes/nonexistent")
        assert resp.status_code == 404

    def test_compare_sessions(self, client):
        # Create and run two sessions
        r1 = client.post("/api/simulation/sessions", json={
            "config": {"initial_population": 20, "generations_to_run": 3, "random_seed": 42},
        })
        sid1 = r1.json()["id"]
        client.post(f"/api/simulation/sessions/{sid1}/run", json={})

        r2 = client.post("/api/simulation/sessions", json={
            "config": {"initial_population": 20, "generations_to_run": 3, "random_seed": 42, "trait_drift_rate": 0.1},
        })
        sid2 = r2.json()["id"]
        client.post(f"/api/simulation/sessions/{sid2}/run", json={})

        resp = client.post("/api/experiments/compare", json={"session_ids": [sid1, sid2]})
        assert resp.status_code == 200
        data = resp.json()
        assert sid1 in data["sessions"]
        assert sid2 in data["sessions"]
        diff_key = f"{sid1}_vs_{sid2}"
        assert diff_key in data["config_diffs"]
        assert "trait_drift_rate" in data["config_diffs"][diff_key]

    def test_inject_outsider(self, client):
        r = client.post("/api/simulation/sessions", json={
            "config": {"initial_population": 20, "generations_to_run": 10, "random_seed": 42},
        })
        sid = r.json()["id"]
        client.post(f"/api/simulation/sessions/{sid}/step", json={"n": 2})

        resp = client.post("/api/experiments/inject-outsider", json={
            "session_id": sid,
            "archetype": "einstein",
            "noise_sigma": 0.05,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["is_outsider"] is True
        assert "einstein" in data["name"].lower() or data["is_outsider"]

    def test_ripple_report(self, client):
        r = client.post("/api/simulation/sessions", json={
            "config": {"initial_population": 20, "generations_to_run": 10, "random_seed": 42},
        })
        sid = r.json()["id"]
        client.post(f"/api/simulation/sessions/{sid}/step", json={"n": 2})
        client.post("/api/experiments/inject-outsider", json={
            "session_id": sid,
            "archetype": "curie",
        })
        client.post(f"/api/simulation/sessions/{sid}/step", json={"n": 3})

        resp = client.get(f"/api/experiments/{sid}/ripple")
        assert resp.status_code == 200
        data = resp.json()
        assert "injections" in data
        assert "snapshots" in data
