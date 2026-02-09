"""Tests for the genetics API router and enhanced experiment endpoints."""

import pytest
from fastapi.testclient import TestClient

from seldon.api.app import create_app


@pytest.fixture
def client():
    app = create_app()
    return TestClient(app)


@pytest.fixture
def session_with_genetics(client):
    """Create a session with genetics enabled and run a few generations."""
    resp = client.post("/api/simulation/sessions", json={
        "config": {
            "initial_population": 20,
            "generations_to_run": 10,
            "random_seed": 42,
            "genetics_config": {"genetics_enabled": True},
            "epigenetics_config": {"epigenetics_enabled": True},
        },
    })
    sid = resp.json()["id"]
    client.post(f"/api/simulation/sessions/{sid}/step", json={"n": 3})
    return sid


@pytest.fixture
def session_basic(client):
    """Create a basic session (no genetics)."""
    resp = client.post("/api/simulation/sessions", json={
        "config": {
            "initial_population": 20,
            "generations_to_run": 10,
            "random_seed": 42,
        },
    })
    sid = resp.json()["id"]
    client.post(f"/api/simulation/sessions/{sid}/step", json={"n": 2})
    return sid


class TestTraitNames:
    def test_get_trait_names(self, client):
        resp = client.get("/api/experiments/trait-names")
        assert resp.status_code == 200
        names = resp.json()
        assert isinstance(names, list)
        assert len(names) > 0
        assert "creativity" in names
        assert "resilience" in names

    def test_trait_names_order_consistent(self, client):
        resp1 = client.get("/api/experiments/trait-names")
        resp2 = client.get("/api/experiments/trait-names")
        assert resp1.json() == resp2.json()


class TestEnhancedInjection:
    def test_inject_with_name(self, client, session_basic):
        resp = client.post("/api/experiments/inject-outsider", json={
            "session_id": session_basic,
            "archetype": "da_vinci",
            "name": "Leonardo Test",
        })
        assert resp.status_code == 200
        assert resp.json()["name"] == "Leonardo Test"

    def test_inject_custom_traits(self, client, session_basic):
        resp = client.post("/api/experiments/inject-outsider", json={
            "session_id": session_basic,
            "custom_traits": {"creativity": 0.95, "resilience": 0.8},
            "name": "Custom Agent",
            "gender": "non-binary",
            "age": 30,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "Custom Agent"
        assert data["is_outsider"] is True

    def test_inject_with_injection_generation(self, client, session_basic):
        resp = client.post("/api/experiments/inject-outsider", json={
            "session_id": session_basic,
            "archetype": "einstein",
            "injection_generation": 5,
        })
        assert resp.status_code == 200

    def test_inject_no_archetype_or_traits_fails(self, client, session_basic):
        resp = client.post("/api/experiments/inject-outsider", json={
            "session_id": session_basic,
        })
        assert resp.status_code == 400


class TestOutsiderEndpoints:
    def test_get_outsiders_empty(self, client, session_basic):
        resp = client.get(f"/api/experiments/{session_basic}/outsiders")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_get_outsiders_after_injection(self, client, session_basic):
        client.post("/api/experiments/inject-outsider", json={
            "session_id": session_basic,
            "archetype": "da_vinci",
            "name": "Test Outsider",
        })
        resp = client.get(f"/api/experiments/{session_basic}/outsiders")
        assert resp.status_code == 200
        outsiders = resp.json()
        assert len(outsiders) >= 1
        assert any(o["name"] == "Test Outsider" for o in outsiders)

    def test_get_outsider_impact(self, client, session_basic):
        inject_resp = client.post("/api/experiments/inject-outsider", json={
            "session_id": session_basic,
            "archetype": "einstein",
            "name": "Albert",
        })
        agent_id = inject_resp.json()["id"]

        resp = client.get(f"/api/experiments/{session_basic}/outsiders/{agent_id}/impact")
        assert resp.status_code == 200
        data = resp.json()
        assert data["agent"]["id"] == agent_id
        assert "descendant_count" in data
        assert "trait_distance_from_mean" in data
        assert data["outsider_origin"] is not None

    def test_get_outsider_impact_non_outsider(self, client, session_basic):
        # Get a regular agent
        agents_resp = client.get(f"/api/agents/{session_basic}", params={"page_size": 1})
        agents = agents_resp.json()["agents"]
        if agents and not agents[0]["is_outsider"]:
            resp = client.get(f"/api/experiments/{session_basic}/outsiders/{agents[0]['id']}/impact")
            assert resp.status_code == 400

    def test_get_outsiders_not_found(self, client):
        resp = client.get("/api/experiments/nonexistent/outsiders")
        assert resp.status_code == 404

    def test_get_outsider_impact_not_found(self, client, session_basic):
        resp = client.get(f"/api/experiments/{session_basic}/outsiders/fake_id/impact")
        assert resp.status_code == 404


class TestGeneticsRouter:
    def test_allele_frequencies_no_genetics(self, client, session_basic):
        resp = client.get(f"/api/genetics/{session_basic}/allele-frequencies")
        assert resp.status_code == 200
        data = resp.json()
        assert "loci" in data

    def test_allele_frequencies_with_genetics(self, client, session_with_genetics):
        resp = client.get(f"/api/genetics/{session_with_genetics}/allele-frequencies")
        assert resp.status_code == 200
        data = resp.json()
        assert "loci" in data
        # If genetics is enabled, we should have loci data
        if data["enabled"]:
            assert len(data["loci"]) > 0
            first = list(data["loci"].values())[0]
            assert "dominant_frequency" in first
            assert "recessive_frequency" in first

    def test_epigenetic_prevalence(self, client, session_with_genetics):
        resp = client.get(f"/api/genetics/{session_with_genetics}/epigenetic-prevalence")
        assert resp.status_code == 200
        data = resp.json()
        assert "markers" in data

    def test_trait_gene_correlation(self, client, session_with_genetics):
        resp = client.get(f"/api/genetics/{session_with_genetics}/trait-gene-correlation")
        assert resp.status_code == 200
        data = resp.json()
        assert "correlations" in data

    def test_genetics_session_not_found(self, client):
        resp = client.get("/api/genetics/nonexistent/allele-frequencies")
        assert resp.status_code == 404

    def test_epigenetic_session_not_found(self, client):
        resp = client.get("/api/genetics/nonexistent/epigenetic-prevalence")
        assert resp.status_code == 404

    def test_correlation_session_not_found(self, client):
        resp = client.get("/api/genetics/nonexistent/trait-gene-correlation")
        assert resp.status_code == 404


class TestAgentGender:
    def test_custom_injection_preserves_gender(self, client, session_basic):
        resp = client.post("/api/experiments/inject-outsider", json={
            "session_id": session_basic,
            "custom_traits": {"creativity": 0.9},
            "gender": "female",
            "name": "Marie",
        })
        assert resp.status_code == 200
        agent_id = resp.json()["id"]

        # Get the outsider impact (which includes gender)
        impact = client.get(f"/api/experiments/{session_basic}/outsiders/{agent_id}/impact")
        assert impact.status_code == 200
        assert impact.json()["gender"] == "female"
