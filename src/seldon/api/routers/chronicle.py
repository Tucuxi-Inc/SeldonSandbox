"""Chronicle and biography endpoints.

All endpoints work WITHOUT an LLM. The `use_llm=true` query param
enables optional prose enrichment for biography and chronicle entries.
"""

from __future__ import annotations

import os
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request

from seldon.api.serializers import serialize_agent_summary
from seldon.llm.biography import BiographyGenerator
from seldon.llm.chronicle import EventExtractor

router = APIRouter()

_biography_generator = BiographyGenerator()
_event_extractor = EventExtractor()


def _get_session(request: Request, session_id: str):
    mgr = request.app.state.session_manager
    try:
        return mgr.get_session(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")


# ---------------------------------------------------------------------------
# Biography endpoints
# ---------------------------------------------------------------------------

@router.get("/{session_id}/biography/{agent_id}")
def get_biography(
    session_id: str,
    agent_id: str,
    request: Request,
    use_llm: bool = Query(False),
    provider: str = Query("anthropic"),
    model: str | None = Query(None),
) -> dict[str, Any]:
    """Structured biography with optional LLM prose."""
    session = _get_session(request, session_id)
    agent = session.all_agents.get(agent_id)
    if agent is None:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")

    ts = session.config.trait_system
    bio = _biography_generator.build_biography_data(agent, session.all_agents, ts)

    if use_llm:
        try:
            from seldon.llm.prompts import BIOGRAPHY_SYSTEM_PROMPT, build_biography_context
            from seldon.api.routers.llm import _make_client
            client = _make_client(provider, model)
            context = build_biography_context(bio)
            resp = client.complete(
                system=BIOGRAPHY_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": f"Write a biography based on this data:\n\n{context}"}],
            )
            bio["prose"] = resp.text
        except Exception as exc:
            bio["prose_error"] = str(exc)

    return bio


@router.get("/{session_id}/biography/{agent_id}/timeline")
def get_agent_timeline(
    session_id: str, agent_id: str, request: Request,
) -> dict[str, Any]:
    """Life events timeline (no LLM)."""
    session = _get_session(request, session_id)
    agent = session.all_agents.get(agent_id)
    if agent is None:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")

    ts = session.config.trait_system
    events = _event_extractor.extract_agent_timeline(agent, session.all_agents, ts)

    return {
        "events": [
            {
                "event_type": e.event_type,
                "generation": e.generation,
                "severity": e.severity,
                "headline": e.headline,
                "detail": e.detail,
                "agent_ids": e.agent_ids,
                "metrics_snapshot": e.metrics_snapshot,
            }
            for e in events
        ],
    }


@router.get("/{session_id}/biography/{agent_id}/death-analysis")
def get_death_analysis(
    session_id: str, agent_id: str, request: Request,
) -> dict[str, Any]:
    """Mortality breakdown for a dead agent."""
    session = _get_session(request, session_id)
    agent = session.all_agents.get(agent_id)
    if agent is None:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")

    death_info = agent.extension_data.get("death_info")
    if death_info is None:
        if not agent.is_alive:
            return {"has_death_info": False, "message": "Agent is dead but no detailed death info available"}
        return {"has_death_info": False, "message": "Agent is still alive"}

    return {"has_death_info": True, **death_info}


# ---------------------------------------------------------------------------
# Chronicle endpoints
# ---------------------------------------------------------------------------

@router.get("/{session_id}/chronicle")
def get_chronicle_index(session_id: str, request: Request) -> dict[str, Any]:
    """Index of all generations with event counts."""
    session = _get_session(request, session_id)
    ts = session.config.trait_system

    generations = []
    history = session.collector.metrics_history

    for i, metrics in enumerate(history):
        prev_metrics = history[i - 1] if i > 0 else None
        events = _event_extractor.extract_generation_events(
            generation=metrics.generation,
            metrics=metrics,
            all_agents=session.all_agents,
            trait_system=ts,
            prev_metrics=prev_metrics,
        )

        max_severity = "minor"
        severity_order = {"minor": 0, "notable": 1, "major": 2, "critical": 3}
        for e in events:
            if severity_order.get(e.severity, 0) > severity_order.get(max_severity, 0):
                max_severity = e.severity

        generations.append({
            "generation": metrics.generation,
            "event_count": len(events),
            "max_severity": max_severity,
            "population_size": metrics.population_size,
        })

    return {"generations": generations}


@router.get("/{session_id}/chronicle/{generation}")
def get_chronicle_entry(
    session_id: str, generation: int, request: Request,
) -> dict[str, Any]:
    """Events for a specific generation."""
    session = _get_session(request, session_id)
    ts = session.config.trait_system
    history = session.collector.metrics_history

    if generation < 0 or generation >= len(history):
        raise HTTPException(status_code=404, detail=f"Generation {generation} not found")

    metrics = history[generation]
    prev_metrics = history[generation - 1] if generation > 0 else None

    events = _event_extractor.extract_generation_events(
        generation=generation,
        metrics=metrics,
        all_agents=session.all_agents,
        trait_system=ts,
        prev_metrics=prev_metrics,
    )

    # Collect agent ID → name mapping for all referenced agents
    all_ref_ids: set[str] = set()
    for e in events:
        all_ref_ids.update(e.agent_ids)
    agent_names = {
        aid: session.all_agents[aid].name
        for aid in all_ref_ids
        if aid in session.all_agents
    }

    return {
        "generation": generation,
        "events": [
            {
                "event_type": e.event_type,
                "generation": e.generation,
                "severity": e.severity,
                "headline": e.headline,
                "detail": e.detail,
                "agent_ids": e.agent_ids,
                "metrics_snapshot": e.metrics_snapshot,
            }
            for e in events
        ],
        "agent_names": agent_names,
        "population_size": metrics.population_size,
        "births": metrics.births,
        "deaths": metrics.deaths,
    }


@router.get("/{session_id}/outsider-story/{agent_id}")
def get_outsider_story(
    session_id: str, agent_id: str, request: Request,
) -> dict[str, Any]:
    """Outsider integration narrative."""
    session = _get_session(request, session_id)
    agent = session.all_agents.get(agent_id)
    if agent is None:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")

    if not agent.is_outsider:
        return {"is_outsider": False}

    ts = session.config.trait_system

    # Get timeline
    events = _event_extractor.extract_agent_timeline(agent, session.all_agents, ts)

    # Count descendants
    descendants = []
    for a in session.all_agents.values():
        if a.id == agent.id:
            continue
        if a.parent1_id == agent.id or a.parent2_id == agent.id:
            descendants.append(serialize_agent_summary(a, ts))

    return {
        "is_outsider": True,
        "agent": serialize_agent_summary(agent, ts),
        "outsider_origin": agent.outsider_origin,
        "injection_generation": agent.injection_generation,
        "timeline": [
            {
                "event_type": e.event_type,
                "generation": e.generation,
                "severity": e.severity,
                "headline": e.headline,
                "detail": e.detail,
            }
            for e in events
        ],
        "direct_descendants": descendants,
        "descendant_count": len(descendants),
    }
