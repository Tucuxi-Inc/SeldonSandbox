"""Agent list, detail, and family tree endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request

from seldon.api.schemas import AgentDetailResponse, PaginatedAgentList
from seldon.api.serializers import (
    serialize_agent_detail,
    serialize_agent_summary,
    serialize_family_tree,
)

router = APIRouter()


@router.get("/{session_id}", response_model=PaginatedAgentList)
def list_agents(
    session_id: str,
    request: Request,
    alive_only: bool = Query(True),
    region: str | None = Query(None),
    generation: int | None = Query(None),
    birth_order: int | None = Query(None),
    is_outsider: bool | None = Query(None),
    search: str | None = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
) -> dict[str, Any]:
    mgr = request.app.state.session_manager
    try:
        session = mgr.get_session(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    ts = session.config.trait_system

    # Source: alive population or all ever-seen agents
    if alive_only:
        agents = session.engine.population
    else:
        agents = list(session.all_agents.values())

    # Apply filters
    if region is not None:
        agents = [a for a in agents if a.processing_region.value == region]
    if generation is not None:
        agents = [a for a in agents if a.generation == generation]
    if birth_order is not None:
        agents = [a for a in agents if a.birth_order == birth_order]
    if is_outsider is not None:
        agents = [a for a in agents if a.is_outsider == is_outsider]
    if search:
        q = search.lower()
        agents = [a for a in agents if q in a.name.lower() or q in a.id.lower()]

    total = len(agents)
    start = (page - 1) * page_size
    end = start + page_size
    page_agents = agents[start:end]

    return {
        "agents": [serialize_agent_summary(a, ts) for a in page_agents],
        "total": total,
        "page": page,
        "page_size": page_size,
    }


@router.get("/{session_id}/{agent_id}")
def get_agent_detail(session_id: str, agent_id: str, request: Request) -> dict[str, Any]:
    mgr = request.app.state.session_manager
    try:
        session = mgr.get_session(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    ts = session.config.trait_system

    # Look in all_agents (includes dead)
    agent = session.all_agents.get(agent_id)
    if agent is None:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")

    return serialize_agent_detail(agent, ts)


@router.get("/{session_id}/{agent_id}/family-tree")
def get_family_tree(
    session_id: str,
    agent_id: str,
    request: Request,
    depth_up: int = Query(3, ge=0, le=10),
    depth_down: int = Query(3, ge=0, le=10),
) -> dict[str, Any]:
    mgr = request.app.state.session_manager
    try:
        session = mgr.get_session(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    ts = session.config.trait_system
    return serialize_family_tree(agent_id, session.all_agents, ts, depth_up, depth_down)
