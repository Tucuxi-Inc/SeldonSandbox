"""Tests for LLM integration: client, prompts, interviewer, narrator."""

from __future__ import annotations

import os
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from seldon.llm.client import (
    ClaudeClient,
    OllamaClient,
    LLMClient,
    LLMResponse,
    LLMUnavailableError,
    create_client,
)
from seldon.llm.prompts import (
    build_agent_context,
    build_generation_context,
    build_decision_context,
)
from seldon.llm.interviewer import AgentInterviewer
from seldon.llm.narrator import NarrativeGenerator, DecisionNarrator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_client() -> LLMClient:
    """Return a mock LLMClient that returns canned responses."""
    client = MagicMock(spec=LLMClient)
    client.provider = "mock"
    client.complete.return_value = LLMResponse(
        text="Mock response text",
        model="mock-model",
        input_tokens=100,
        output_tokens=50,
    )
    return client


def _sample_agent_data() -> dict:
    return {
        "id": "agent-001",
        "name": "Ada Lovelace",
        "age": 25,
        "generation": 3,
        "birth_order": 1,
        "processing_region": "optimal",
        "suffering": 0.3,
        "is_alive": True,
        "partner_id": "agent-002",
        "is_outsider": False,
        "dominant_voice": "analytical",
        "latest_contribution": 4.2,
        "traits": {
            "openness": 0.9,
            "conscientiousness": 0.7,
            "extraversion": 0.4,
            "agreeableness": 0.6,
            "neuroticism": 0.3,
        },
        "traits_at_birth": {
            "openness": 0.85,
            "conscientiousness": 0.65,
            "extraversion": 0.45,
            "agreeableness": 0.55,
            "neuroticism": 0.35,
        },
        "trait_history": [],
        "region_history": ["under_processing", "optimal"],
        "contribution_history": [1.0, 2.5, 4.2],
        "suffering_history": [0.1, 0.2, 0.3],
        "parent1_id": "agent-010",
        "parent2_id": "agent-011",
        "children_ids": ["agent-020"],
        "relationship_status": "paired",
        "burnout_level": 0.15,
        "personal_memories": [
            {"content": "First breakthrough", "emotional_intensity": 0.9, "type": "achievement"},
            {"content": "Loss of a friend", "emotional_intensity": 0.7, "type": "loss"},
        ],
        "inherited_lore": [
            {"content": "The founders were wise", "fidelity": 0.8},
        ],
        "decision_history": [
            {
                "context": "migration",
                "chosen_action": "stay",
                "actions": ["stay", "migrate"],
                "utilities": {"stay": 0.7, "migrate": 0.3},
                "probabilities": {"stay": 0.8, "migrate": 0.2},
                "trait_contributions": {"openness": 0.15, "conscientiousness": 0.1},
            },
        ],
        "outsider_origin": None,
        "injection_generation": None,
    }


def _sample_metrics() -> dict:
    return {
        "generation": 5,
        "population_size": 100,
        "births": 12,
        "deaths": 5,
        "breakthroughs": 2,
        "pairs_formed": 4,
        "dissolutions": 1,
        "region_fractions": {
            "under_processing": 0.2,
            "optimal": 0.4,
            "deep": 0.2,
            "sacrificial": 0.1,
            "pathological": 0.1,
        },
        "region_transitions": {"optimal_to_deep": 3, "deep_to_sacrificial": 1},
        "mean_contribution": 2.5,
        "max_contribution": 8.1,
        "mean_suffering": 0.4,
        "extension_metrics": {},
    }


# ---------------------------------------------------------------------------
# TestClaudeClient
# ---------------------------------------------------------------------------

class TestClaudeClient:
    def test_raises_without_api_key(self):
        with patch.dict(os.environ, {}, clear=True):
            # Ensure ANTHROPIC_API_KEY is not set
            env = os.environ.copy()
            env.pop("ANTHROPIC_API_KEY", None)
            with patch.dict(os.environ, env, clear=True):
                with pytest.raises(LLMUnavailableError, match="ANTHROPIC_API_KEY"):
                    ClaudeClient()

    def test_raises_without_package(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch.dict("sys.modules", {"anthropic": None}):
                with pytest.raises(LLMUnavailableError, match="not installed"):
                    ClaudeClient()

    def test_is_available_false_without_key(self):
        with patch.dict(os.environ, {}, clear=True):
            env = os.environ.copy()
            env.pop("ANTHROPIC_API_KEY", None)
            with patch.dict(os.environ, env, clear=True):
                assert ClaudeClient.is_available() is False

    def test_is_available_true_with_key_and_package(self):
        mock_anthropic = MagicMock()
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
                assert ClaudeClient.is_available() is True

    def test_complete_returns_llm_response(self):
        """Test that complete() properly wraps the API response."""
        mock_anthropic = MagicMock()
        mock_client_instance = MagicMock()

        # Mock the response structure
        mock_response = SimpleNamespace(
            content=[SimpleNamespace(text="Hello from Claude")],
            model="claude-sonnet-4-20250514",
            usage=SimpleNamespace(input_tokens=50, output_tokens=25),
        )
        mock_client_instance.messages.create.return_value = mock_response
        mock_anthropic.Anthropic.return_value = mock_client_instance

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
                client = ClaudeClient()
                resp = client.complete(
                    system="You are helpful.",
                    messages=[{"role": "user", "content": "Hello"}],
                )

        assert isinstance(resp, LLMResponse)
        assert resp.text == "Hello from Claude"
        assert resp.model == "claude-sonnet-4-20250514"
        assert resp.input_tokens == 50
        assert resp.output_tokens == 25


# ---------------------------------------------------------------------------
# TestOllamaClient
# ---------------------------------------------------------------------------

class TestOllamaClient:
    def test_default_base_url_local(self):
        """Outside Docker, defaults to localhost."""
        with patch.dict(os.environ, {}, clear=True):
            env = os.environ.copy()
            env.pop("OLLAMA_HOST", None)
            env.pop("DOCKER_CONTAINER", None)
            with patch.dict(os.environ, env, clear=True):
                with patch("os.path.exists", return_value=False):
                    client = OllamaClient()
                    assert "localhost" in client.base_url

    def test_docker_base_url(self):
        """Inside Docker, uses host.docker.internal."""
        with patch.dict(os.environ, {"DOCKER_CONTAINER": "1"}, clear=True):
            with patch("os.path.exists", return_value=False):
                client = OllamaClient()
                assert "host.docker.internal" in client.base_url

    def test_explicit_host_env(self):
        """OLLAMA_HOST env var overrides auto-detection."""
        with patch.dict(os.environ, {"OLLAMA_HOST": "http://custom:11434"}):
            client = OllamaClient()
            assert client.base_url == "http://custom:11434"

    def test_is_available_false_when_unreachable(self):
        assert OllamaClient.is_available(base_url="http://127.0.0.1:99999") is False

    def test_list_models_empty_when_unreachable(self):
        assert OllamaClient.list_models(base_url="http://127.0.0.1:99999") == []

    def test_complete_returns_llm_response(self):
        """Test that complete() properly handles Ollama's response format."""
        import json

        response_data = json.dumps({
            "message": {"content": "Hello from Ollama"},
            "model": "llama3.2",
            "prompt_eval_count": 40,
            "eval_count": 20,
        }).encode()

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.read.return_value = response_data
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response):
            client = OllamaClient(model="llama3.2", base_url="http://localhost:11434")
            resp = client.complete(
                system="You are helpful.",
                messages=[{"role": "user", "content": "Hello"}],
            )

        assert isinstance(resp, LLMResponse)
        assert resp.text == "Hello from Ollama"
        assert resp.model == "llama3.2"
        assert resp.input_tokens == 40
        assert resp.output_tokens == 20


# ---------------------------------------------------------------------------
# TestCreateClient factory
# ---------------------------------------------------------------------------

class TestCreateClient:
    def test_create_ollama_client(self):
        client = create_client(provider="ollama", model="test-model", ollama_base_url="http://test:11434")
        assert isinstance(client, OllamaClient)
        assert client.model == "test-model"

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            create_client(provider="openai")


# ---------------------------------------------------------------------------
# TestPromptBuilders
# ---------------------------------------------------------------------------

class TestPromptBuilders:
    def test_agent_context_includes_identity(self):
        agent = _sample_agent_data()
        ctx = build_agent_context(agent, list(agent["traits"].keys()))
        assert "Ada Lovelace" in ctx
        assert "optimal" in ctx
        assert "analytical" in ctx
        assert "PERSONALITY TRAITS" in ctx

    def test_agent_context_includes_sorted_traits(self):
        agent = _sample_agent_data()
        ctx = build_agent_context(agent, list(agent["traits"].keys()))
        # openness (0.9) should appear before neuroticism (0.3)
        assert ctx.index("openness") < ctx.index("neuroticism")

    def test_agent_context_includes_memories(self):
        agent = _sample_agent_data()
        ctx = build_agent_context(agent, list(agent["traits"].keys()))
        assert "First breakthrough" in ctx
        assert "STRONGEST MEMORIES" in ctx

    def test_agent_context_includes_lore(self):
        agent = _sample_agent_data()
        ctx = build_agent_context(agent, list(agent["traits"].keys()))
        assert "founders were wise" in ctx
        assert "INHERITED LORE" in ctx

    def test_agent_context_includes_decisions(self):
        agent = _sample_agent_data()
        ctx = build_agent_context(agent, list(agent["traits"].keys()))
        assert "migration" in ctx
        assert "stay" in ctx

    def test_agent_context_includes_life_arc(self):
        agent = _sample_agent_data()
        ctx = build_agent_context(agent, list(agent["traits"].keys()))
        assert "LIFE ARC" in ctx
        assert "Peak contribution" in ctx

    def test_agent_context_trait_changes(self):
        agent = _sample_agent_data()
        # openness changed from 0.85 to 0.9 (delta = 0.05, at boundary)
        # agreeableness changed from 0.55 to 0.6 (delta = 0.05, at boundary)
        # Make a bigger change
        agent["traits"]["openness"] = 0.95
        ctx = build_agent_context(agent, list(agent["traits"].keys()))
        assert "TRAIT CHANGES SINCE BIRTH" in ctx

    def test_generation_context_includes_stats(self):
        metrics = _sample_metrics()
        ctx = build_generation_context(5, metrics)
        assert "GENERATION 5" in ctx
        assert "Population: 100" in ctx
        assert "Births: 12" in ctx
        assert "Deaths: 5" in ctx

    def test_generation_context_includes_regions(self):
        metrics = _sample_metrics()
        ctx = build_generation_context(5, metrics)
        assert "PROCESSING REGION DISTRIBUTION" in ctx
        assert "optimal" in ctx

    def test_generation_context_with_notable_agents(self):
        metrics = _sample_metrics()
        notable = [{"name": "Einstein", "processing_region": "deep", "latest_contribution": 8.0}]
        ctx = build_generation_context(5, metrics, notable_agents=notable)
        assert "Einstein" in ctx
        assert "NOTABLE AGENTS" in ctx

    def test_generation_context_with_lore(self):
        metrics = _sample_metrics()
        lore = [{"content": "The great flood", "fidelity": 0.6}]
        ctx = build_generation_context(5, metrics, societal_lore=lore)
        assert "great flood" in ctx

    def test_decision_context_marks_chosen_action(self):
        agent = _sample_agent_data()
        decision = agent["decision_history"][0]
        ctx = build_decision_context(agent, decision, list(agent["traits"].keys()))
        assert "<<<CHOSEN" in ctx
        assert "stay" in ctx

    def test_decision_context_includes_traits(self):
        agent = _sample_agent_data()
        decision = agent["decision_history"][0]
        ctx = build_decision_context(agent, decision, list(agent["traits"].keys()))
        assert "TOP CONTRIBUTING TRAITS" in ctx
        assert "openness" in ctx


# ---------------------------------------------------------------------------
# TestAgentInterviewer
# ---------------------------------------------------------------------------

class TestAgentInterviewer:
    def test_interview_returns_response(self):
        client = _mock_client()
        interviewer = AgentInterviewer(client)
        agent = _sample_agent_data()

        resp = interviewer.interview(
            agent_data=agent,
            trait_names=list(agent["traits"].keys()),
            question="Tell me about yourself.",
        )

        assert isinstance(resp, LLMResponse)
        assert resp.text == "Mock response text"
        client.complete.assert_called_once()

    def test_interview_passes_system_prompt_with_context(self):
        client = _mock_client()
        interviewer = AgentInterviewer(client)
        agent = _sample_agent_data()

        interviewer.interview(
            agent_data=agent,
            trait_names=list(agent["traits"].keys()),
            question="Hello?",
        )

        call_kwargs = client.complete.call_args
        system = call_kwargs.kwargs.get("system") or call_kwargs[1].get("system") or call_kwargs[0][0]
        assert "Ada Lovelace" in system
        assert "CHARACTER CONTEXT" in system

    def test_multi_turn_conversation(self):
        client = _mock_client()
        interviewer = AgentInterviewer(client)
        agent = _sample_agent_data()

        history = [
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "I am well."},
        ]

        interviewer.interview(
            agent_data=agent,
            trait_names=list(agent["traits"].keys()),
            question="What do you think about the future?",
            conversation_history=history,
        )

        call_kwargs = client.complete.call_args
        messages = call_kwargs.kwargs.get("messages") or call_kwargs[1].get("messages") or call_kwargs[0][1]
        # Should have history (2 messages) + new question (1 message) = 3
        assert len(messages) == 3
        assert messages[0]["content"] == "How are you?"
        assert messages[-1]["content"] == "What do you think about the future?"


# ---------------------------------------------------------------------------
# TestNarrativeGenerator
# ---------------------------------------------------------------------------

class TestNarrativeGenerator:
    def test_narrate_generation(self):
        client = _mock_client()
        narrator = NarrativeGenerator(client)
        metrics = _sample_metrics()

        resp = narrator.narrate_generation(generation=5, metrics=metrics)

        assert isinstance(resp, LLMResponse)
        assert resp.text == "Mock response text"
        client.complete.assert_called_once()

    def test_narrate_with_previous_narrative(self):
        client = _mock_client()
        narrator = NarrativeGenerator(client)
        metrics = _sample_metrics()

        narrator.narrate_generation(
            generation=5,
            metrics=metrics,
            previous_narrative="Last generation saw growth.",
        )

        call_kwargs = client.complete.call_args
        messages = call_kwargs.kwargs.get("messages") or call_kwargs[1].get("messages") or call_kwargs[0][1]
        user_msg = messages[0]["content"]
        assert "PREVIOUS GENERATION NARRATIVE" in user_msg
        assert "Last generation saw growth" in user_msg

    def test_narrate_uses_temperature_0_8(self):
        client = _mock_client()
        narrator = NarrativeGenerator(client)
        metrics = _sample_metrics()

        narrator.narrate_generation(generation=5, metrics=metrics)

        call_kwargs = client.complete.call_args
        temperature = call_kwargs.kwargs.get("temperature") or call_kwargs[1].get("temperature")
        assert temperature == 0.8


# ---------------------------------------------------------------------------
# TestDecisionNarrator
# ---------------------------------------------------------------------------

class TestDecisionNarrator:
    def test_explain_decision(self):
        client = _mock_client()
        narrator = DecisionNarrator(client)
        agent = _sample_agent_data()
        decision = agent["decision_history"][0]

        resp = narrator.explain_decision(
            agent_data=agent,
            decision=decision,
            trait_names=list(agent["traits"].keys()),
        )

        assert isinstance(resp, LLMResponse)
        assert resp.text == "Mock response text"
        client.complete.assert_called_once()

    def test_explain_uses_temperature_0_6(self):
        client = _mock_client()
        narrator = DecisionNarrator(client)
        agent = _sample_agent_data()
        decision = agent["decision_history"][0]

        narrator.explain_decision(
            agent_data=agent,
            decision=decision,
            trait_names=list(agent["traits"].keys()),
        )

        call_kwargs = client.complete.call_args
        temperature = call_kwargs.kwargs.get("temperature") or call_kwargs[1].get("temperature")
        assert temperature == 0.6


# ---------------------------------------------------------------------------
# TestLLMRouter (API integration tests)
# ---------------------------------------------------------------------------

class TestLLMRouter:
    @pytest.fixture
    def client(self):
        from seldon.api.app import create_app
        from fastapi.testclient import TestClient
        app = create_app()
        return TestClient(app)

    def test_status_endpoint(self, client):
        resp = client.get("/api/llm/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "available" in data
        assert "providers" in data
        assert "anthropic" in data["providers"]
        assert "ollama" in data["providers"]

    def test_interview_session_not_found(self, client):
        resp = client.post(
            "/api/llm/nonexistent/interview/agent-1",
            json={"question": "Hello?"},
        )
        assert resp.status_code == 404

    def test_narrative_session_not_found(self, client):
        resp = client.get("/api/llm/nonexistent/narrative/0")
        assert resp.status_code == 404

    def test_decision_explain_session_not_found(self, client):
        resp = client.post(
            "/api/llm/nonexistent/decision-explain",
            json={"agent_id": "a", "decision_index": 0},
        )
        assert resp.status_code == 404
