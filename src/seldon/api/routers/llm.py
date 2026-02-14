"""LLM endpoints: provider status, agent interview, generation narrative, decision explanation."""

from __future__ import annotations

import os
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from seldon.api.serializers import serialize_agent_detail, serialize_agent_summary, serialize_agent_at_generation, serialize_metrics
from seldon.llm.client import (
    ANTHROPIC_MODELS,
    ClaudeClient,
    OllamaClient,
    LLMUnavailableError,
    create_client,
    _ollama_base_url,
)
from seldon.llm.interviewer import AgentInterviewer
from seldon.llm.narrator import DecisionNarrator, NarrativeGenerator

router = APIRouter()

# In-memory runtime settings
_runtime_api_key: str | None = None
_runtime_ollama_url: str | None = None


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class InterviewRequest(BaseModel):
    question: str
    conversation_history: list[dict[str, str]] | None = None
    provider: str = "anthropic"
    model: str | None = None


class HistoricalInterviewRequest(BaseModel):
    question: str
    target_generation: int
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


class ApiKeyRequest(BaseModel):
    api_key: str


class OllamaUrlRequest(BaseModel):
    base_url: str


class TestConnectionRequest(BaseModel):
    provider: str
    model: str | None = None
    api_key: str | None = None
    ollama_base_url: str | None = None


# ---------------------------------------------------------------------------
# Helper: build client on-the-fly from request params
# ---------------------------------------------------------------------------

def _get_anthropic_key() -> str | None:
    """Get Anthropic API key: runtime override > environment variable."""
    return _runtime_api_key or os.environ.get("ANTHROPIC_API_KEY")


def _get_ollama_url() -> str | None:
    """Get Ollama base URL: runtime override > auto-detected."""
    return _runtime_ollama_url


def _make_client(provider: str, model: str | None = None):
    """Create an LLMClient from provider name, raising HTTP 503 on failure."""
    try:
        if provider == "anthropic":
            return create_client(provider=provider, model=model, api_key=_get_anthropic_key())
        return create_client(
            provider=provider, model=model, ollama_base_url=_get_ollama_url(),
        )
    except LLMUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/status")
def llm_status() -> dict[str, Any]:
    """Check availability of all LLM providers with model lists."""
    # Check Anthropic: env var OR runtime key
    anthropic_ok = bool(_get_anthropic_key())
    if anthropic_ok:
        try:
            import anthropic  # noqa: F401
        except ImportError:
            anthropic_ok = False

    ollama_url = _get_ollama_url()
    ollama_ok = OllamaClient.is_available(base_url=ollama_url)
    ollama_models: list[str] = []
    if ollama_ok:
        ollama_models = OllamaClient.list_models(base_url=ollama_url)

    available = anthropic_ok or ollama_ok
    if available:
        message = "LLM features available"
    else:
        message = "No LLM provider available. Set an API key for Anthropic or start Ollama."

    return {
        "available": available,
        "message": message,
        "providers": {
            "anthropic": {
                "available": anthropic_ok,
                "has_key": bool(_get_anthropic_key()),
                "models": list(ANTHROPIC_MODELS) if anthropic_ok else [],
            },
            "ollama": {
                "available": ollama_ok,
                "models": ollama_models,
                "base_url": ollama_url or _ollama_base_url(),
            },
        },
    }


@router.post("/api-key")
def set_api_key(body: ApiKeyRequest) -> dict[str, Any]:
    """Set the Anthropic API key at runtime (stored in-memory only)."""
    global _runtime_api_key
    key = body.api_key.strip()
    if not key:
        raise HTTPException(status_code=400, detail="API key cannot be empty")
    _runtime_api_key = key
    try:
        ClaudeClient(api_key=key)
        return {"status": "ok", "message": "API key set successfully"}
    except LLMUnavailableError as exc:
        _runtime_api_key = None
        raise HTTPException(status_code=400, detail=str(exc))


@router.delete("/api-key")
def clear_api_key() -> dict[str, Any]:
    """Clear the runtime API key (falls back to env var)."""
    global _runtime_api_key
    _runtime_api_key = None
    return {"status": "ok", "message": "Runtime API key cleared"}


@router.post("/ollama-url")
def set_ollama_url(body: OllamaUrlRequest) -> dict[str, Any]:
    """Set a custom Ollama base URL at runtime."""
    global _runtime_ollama_url
    url = body.base_url.strip()
    if not url:
        _runtime_ollama_url = None
        return {"status": "ok", "message": "Ollama URL reset to auto-detect"}
    if not url.startswith("http"):
        url = f"http://{url}"
    # Verify connectivity
    if not OllamaClient.is_available(base_url=url):
        raise HTTPException(status_code=400, detail=f"Cannot connect to Ollama at {url}")
    _runtime_ollama_url = url
    models = OllamaClient.list_models(base_url=url)
    return {"status": "ok", "message": f"Connected. {len(models)} models available.", "base_url": url}


@router.post("/test-connection")
def test_connection(body: TestConnectionRequest) -> dict[str, Any]:
    """Test connectivity to a provider. Returns success/failure with details."""
    if body.provider == "anthropic":
        key = body.api_key or _get_anthropic_key()
        if not key:
            return {"success": False, "message": "No API key provided"}
        try:
            import anthropic  # noqa: F401
        except ImportError:
            return {"success": False, "message": "anthropic package not installed"}
        try:
            ClaudeClient(api_key=key)
            return {
                "success": True,
                "message": "Connected to Anthropic API",
                "models": list(ANTHROPIC_MODELS),
            }
        except LLMUnavailableError as exc:
            return {"success": False, "message": str(exc)}

    elif body.provider == "ollama":
        url = body.ollama_base_url or _get_ollama_url()
        ok = OllamaClient.is_available(base_url=url)
        if not ok:
            detected_url = url or _ollama_base_url()
            return {"success": False, "message": f"Cannot reach Ollama at {detected_url}"}
        models = OllamaClient.list_models(base_url=url)
        return {
            "success": True,
            "message": f"Connected. {len(models)} models available.",
            "models": models,
            "base_url": url or _ollama_base_url(),
        }

    return {"success": False, "message": f"Unknown provider: {body.provider}"}


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


@router.post("/{session_id}/interview/{agent_id}/historical")
def interview_agent_historical(
    session_id: str,
    agent_id: str,
    body: HistoricalInterviewRequest,
    request: Request,
) -> dict[str, Any]:
    """Interview an agent at a specific past generation."""
    mgr = request.app.state.session_manager
    try:
        session = mgr.get_session(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    agent = session.all_agents.get(agent_id)
    if agent is None:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")

    # Validate generation range
    birth_gen = int(agent.generation)
    lived_gens = len(agent.contribution_history)
    last_gen = birth_gen + lived_gens - 1 if lived_gens > 0 else birth_gen

    if body.target_generation < birth_gen or body.target_generation > last_gen:
        raise HTTPException(
            status_code=400,
            detail=f"Generation {body.target_generation} is outside agent's lived range [{birth_gen}, {last_gen}]",
        )

    ts = session.config.trait_system
    agent_snapshot = serialize_agent_at_generation(agent, ts, body.target_generation)

    # Build context from historical snapshot
    from seldon.llm.prompts import HISTORICAL_INTERVIEW_SYSTEM_PROMPT, build_agent_context

    age_at_target = body.target_generation - birth_gen
    system_prompt = HISTORICAL_INTERVIEW_SYSTEM_PROMPT.format(
        name=agent.name,
        age=age_at_target,
        generation=body.target_generation,
    )

    # Build agent context from snapshot data
    context_parts = [
        f"NAME: {agent_snapshot['name']}",
        f"AGE: {agent_snapshot['age']}",
        f"GENERATION: {agent_snapshot['target_generation']}",
        f"PROCESSING REGION: {agent_snapshot['processing_region']}",
        f"RELATIONSHIP STATUS: {agent_snapshot['relationship_status']}",
    ]
    if agent_snapshot.get("traits"):
        context_parts.append("\nPERSONALITY TRAITS:")
        sorted_traits = sorted(agent_snapshot["traits"].items(), key=lambda x: x[1], reverse=True)
        for name, val in sorted_traits:
            context_parts.append(f"  {name}: {val:.2f}")

    memories = agent_snapshot.get("personal_memories", [])
    if memories:
        context_parts.append(f"\nMEMORIES ({len(memories)} up to this point):")
        for mem in memories[-5:]:
            content = mem.get("content", mem.get("type", "memory"))
            context_parts.append(f"  - {content}")

    context = "\n".join(context_parts)

    client = _make_client(body.provider, body.model)
    try:
        history = body.conversation_history or []
        messages = [{"role": m["role"], "content": m["content"]} for m in history]
        messages.append({"role": "user", "content": body.question})

        resp = client.generate(
            system_prompt=f"{system_prompt}\n\nCHARACTER CONTEXT:\n{context}",
            user_message=body.question,
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
        "target_generation": body.target_generation,
        "agent_age_at_generation": age_at_target,
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
