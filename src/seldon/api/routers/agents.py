"""Agent list, detail, family tree, comparison, and generation range endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel

from seldon.api.schemas import AgentDetailResponse, PaginatedAgentList
from seldon.api.serializers import (
    serialize_agent_detail,
    serialize_agent_summary,
    serialize_family_tree,
)


class CompareAgentsRequest(BaseModel):
    agent_ids: list[str]

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


@router.get("/{session_id}/{agent_id}/generation-range")
def get_generation_range(
    session_id: str, agent_id: str, request: Request,
) -> dict[str, Any]:
    """Get the birth and last active generation for an agent."""
    mgr = request.app.state.session_manager
    try:
        session = mgr.get_session(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    agent = session.all_agents.get(agent_id)
    if agent is None:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")

    birth_gen = int(agent.generation)
    lived_gens = len(agent.contribution_history)
    last_gen = birth_gen + lived_gens - 1 if lived_gens > 0 else birth_gen

    return {
        "birth_generation": birth_gen,
        "last_generation": last_gen,
        "is_alive": agent.is_alive,
    }


@router.post("/{session_id}/compare")
def compare_agents(
    session_id: str, body: CompareAgentsRequest, request: Request,
) -> dict[str, Any]:
    """Compare 2-3 agents with full details + relationship detection."""
    mgr = request.app.state.session_manager
    try:
        session = mgr.get_session(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    ts = session.config.trait_system
    agent_ids = body.agent_ids[:3]  # Cap at 3

    agents_data = []
    for aid in agent_ids:
        agent = session.all_agents.get(aid)
        if agent is None:
            raise HTTPException(status_code=404, detail=f"Agent '{aid}' not found")
        agents_data.append(serialize_agent_detail(agent, ts))

    # Detect relationships between the compared agents
    relationships = []
    id_set = set(agent_ids)
    for i, a_data in enumerate(agents_data):
        for j, b_data in enumerate(agents_data):
            if i >= j:
                continue
            a_id, b_id = a_data["id"], b_data["id"]
            # Partner?
            if a_data.get("partner_id") == b_id:
                relationships.append({
                    "type": "partners",
                    "agent_a": a_id,
                    "agent_b": b_id,
                    "detail": "Current partners",
                })
            # Parent-child?
            if a_data.get("parent1_id") == b_id or a_data.get("parent2_id") == b_id:
                relationships.append({
                    "type": "parent_child",
                    "agent_a": b_id,
                    "agent_b": a_id,
                    "detail": f"{b_data['name']} is parent of {a_data['name']}",
                })
            if b_data.get("parent1_id") == a_id or b_data.get("parent2_id") == a_id:
                relationships.append({
                    "type": "parent_child",
                    "agent_a": a_id,
                    "agent_b": b_id,
                    "detail": f"{a_data['name']} is parent of {b_data['name']}",
                })
            # Siblings?
            a_parents = {a_data.get("parent1_id"), a_data.get("parent2_id")} - {None}
            b_parents = {b_data.get("parent1_id"), b_data.get("parent2_id")} - {None}
            if a_parents & b_parents:
                relationships.append({
                    "type": "siblings",
                    "agent_a": a_id,
                    "agent_b": b_id,
                    "detail": f"{a_data['name']} and {b_data['name']} share a parent",
                })

    return {
        "agents": agents_data,
        "relationships": relationships,
    }
