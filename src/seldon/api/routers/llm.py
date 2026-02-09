"""LLM endpoints: provider status, agent interview, generation narrative, decision explanation."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from seldon.api.serializers import serialize_agent_detail, serialize_agent_summary, serialize_metrics
from seldon.llm.client import (
    ClaudeClient,
    OllamaClient,
    LLMUnavailableError,
    create_client,
)
from seldon.llm.interviewer import AgentInterviewer
from seldon.llm.narrator import DecisionNarrator, NarrativeGenerator

router = APIRouter()


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class InterviewRequest(BaseModel):
    question: str
    conversation_history: list[dict[str, str]] | None = None
    provider: str = "anthropic"
    model: str | None = None


class DecisionExplainRequest(BaseModel):
    agent_id: str
    decision_index: int
    provider: str = "anthropic"
    model: str | None = None


class ProviderSettings(BaseModel):
    provider: str = "anthropic"
    model: str | None = None


# ---------------------------------------------------------------------------
# Helper: build client on-the-fly from request params
# ---------------------------------------------------------------------------

def _make_client(provider: str, model: str | None = None):
    """Create an LLMClient from provider name, raising HTTP 503 on failure."""
    try:
        return create_client(provider=provider, model=model)
    except LLMUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/status")
def llm_status() -> dict[str, Any]:
    """Check availability of all LLM providers."""
    anthropic_ok = ClaudeClient.is_available()
    ollama_ok = OllamaClient.is_available()
    ollama_models: list[str] = []
    if ollama_ok:
        ollama_models = OllamaClient.list_models()

    available = anthropic_ok or ollama_ok
    if available:
        message = "LLM features available"
    else:
        message = "No LLM provider available. Set ANTHROPIC_API_KEY or start Ollama locally."

    return {
        "available": available,
        "message": message,
        "providers": {
            "anthropic": {"available": anthropic_ok},
            "ollama": {
                "available": ollama_ok,
                "models": ollama_models,
            },
        },
    }


@router.post("/{session_id}/interview/{agent_id}")
def interview_agent(
    session_id: str,
    agent_id: str,
    body: InterviewRequest,
    request: Request,
) -> dict[str, Any]:
    """Conduct an in-character interview with an agent."""
    mgr = request.app.state.session_manager
    try:
        session = mgr.get_session(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    agent = session.all_agents.get(agent_id)
    if agent is None:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")

    ts = session.config.trait_system
    agent_data = serialize_agent_detail(agent, ts)
    trait_names = ts.names()

    client = _make_client(body.provider, body.model)
    interviewer = AgentInterviewer(client)

    try:
        resp = interviewer.interview(
            agent_data=agent_data,
            trait_names=trait_names,
            question=body.question,
            conversation_history=body.conversation_history,
        )
    except LLMUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    return {
        "response": resp.text,
        "model": resp.model,
        "input_tokens": resp.input_tokens,
        "output_tokens": resp.output_tokens,
        "provider": body.provider,
    }


@router.get("/{session_id}/narrative/{generation}")
def generate_narrative(
    session_id: str,
    generation: int,
    request: Request,
    provider: str = "anthropic",
    model: str | None = None,
) -> dict[str, Any]:
    """Generate a prose narrative for a specific generation."""
    mgr = request.app.state.session_manager
    try:
        session = mgr.get_session(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    ts = session.config.trait_system

    # Find the matching generation metrics
    if generation < 0 or generation >= len(session.collector.history):
        raise HTTPException(status_code=404, detail=f"Generation {generation} not found")

    metrics_obj = session.collector.history[generation]
    metrics_dict = serialize_metrics(metrics_obj, ts)

    # Top-5 agents by contribution (from living population at that point)
    living = [a for a in session.engine.population if a.is_alive]
    sorted_agents = sorted(
        living,
        key=lambda a: a.contribution_history[-1] if a.contribution_history else 0,
        reverse=True,
    )[:5]
    notable = [serialize_agent_summary(a, ts) for a in sorted_agents]

    # Societal lore
    societal_lore = None
    if hasattr(session.engine, "lore_engine") and session.engine.lore_engine:
        lore_engine = session.engine.lore_engine
        if hasattr(lore_engine, "societal_memories"):
            societal_lore = [
                m.to_dict() if hasattr(m, "to_dict") else m
                for m in lore_engine.societal_memories[:5]
            ]

    extension_metrics = metrics_dict.get("extension_metrics")

    client = _make_client(provider, model)
    narrator = NarrativeGenerator(client)

    try:
        resp = narrator.narrate_generation(
            generation=generation,
            metrics=metrics_dict,
            notable_agents=notable,
            societal_lore=societal_lore,
            extension_metrics=extension_metrics,
        )
    except LLMUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    return {
        "narrative": resp.text,
        "generation": generation,
        "model": resp.model,
        "input_tokens": resp.input_tokens,
        "output_tokens": resp.output_tokens,
        "provider": provider,
    }


@router.post("/{session_id}/decision-explain")
def explain_decision(
    session_id: str,
    body: DecisionExplainRequest,
    request: Request,
) -> dict[str, Any]:
    """Explain a specific decision from an agent's decision history."""
    mgr = request.app.state.session_manager
    try:
        session = mgr.get_session(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    agent = session.all_agents.get(body.agent_id)
    if agent is None:
        raise HTTPException(status_code=404, detail=f"Agent '{body.agent_id}' not found")

    ts = session.config.trait_system
    agent_data = serialize_agent_detail(agent, ts)
    trait_names = ts.names()

    decisions = agent_data.get("decision_history", [])
    if body.decision_index < 0 or body.decision_index >= len(decisions):
        raise HTTPException(
            status_code=404,
            detail=f"Decision index {body.decision_index} out of range (agent has {len(decisions)} decisions)",
        )

    decision = decisions[body.decision_index]

    client = _make_client(body.provider, body.model)
    narrator = DecisionNarrator(client)

    try:
        resp = narrator.explain_decision(
            agent_data=agent_data,
            decision=decision,
            trait_names=trait_names,
        )
    except LLMUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    return {
        "explanation": resp.text,
        "decision": decision,
        "model": resp.model,
        "input_tokens": resp.input_tokens,
        "output_tokens": resp.output_tokens,
        "provider": body.provider,
    }
