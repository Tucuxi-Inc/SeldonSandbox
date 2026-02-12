"""Belief system API endpoints: overview, per-agent beliefs, distributions, ground truths."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request

router = APIRouter()


def _get_session(session_id: str, request: Request):
    """Get session or raise 404."""
    mgr = request.app.state.session_manager
    try:
        return mgr.get_session(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")


def _get_epistemology_ext(session):
    """Get the epistemology extension if enabled, else None."""
    for ext in session.engine.extensions.get_enabled():
        if ext.name == "epistemology":
            return ext
    return None


@router.get("/{session_id}/overview")
def get_belief_overview(session_id: str, request: Request) -> dict[str, Any]:
    """Return belief system overview: totals, distributions, societal beliefs."""
    session = _get_session(session_id, request)
    ext = _get_epistemology_ext(session)
    if ext is None or ext.belief_system is None:
        return {"enabled": False}

    bs = ext.belief_system
    metrics = bs.get_metrics(session.engine.population)
    societal = [b.to_dict() for b in bs.societal_beliefs]

    return {
        "enabled": True,
        **metrics,
        "societal_beliefs": societal,
    }


@router.get("/{session_id}/agent/{agent_id}")
def get_agent_beliefs(
    session_id: str, agent_id: str, request: Request,
) -> dict[str, Any]:
    """Return all beliefs held by a specific agent."""
    session = _get_session(session_id, request)
    ext = _get_epistemology_ext(session)
    if ext is None or ext.belief_system is None:
        return {"enabled": False}

    agent = session.all_agents.get(agent_id)
    if agent is None:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")

    from seldon.social.beliefs import BeliefSystem
    beliefs = BeliefSystem._get_beliefs(agent)
    return {
        "enabled": True,
        "agent_id": agent_id,
        "beliefs": [b.to_dict() for b in beliefs],
    }


@router.get("/{session_id}/epistemology-distribution")
def get_epistemology_distribution(
    session_id: str, request: Request,
) -> dict[str, Any]:
    """Return count + mean accuracy per epistemology type."""
    session = _get_session(session_id, request)
    ext = _get_epistemology_ext(session)
    if ext is None or ext.belief_system is None:
        return {"enabled": False}

    from seldon.social.beliefs import BeliefSystem, EpistemologyType

    distribution: dict[str, dict[str, Any]] = {
        e.value: {"count": 0, "accuracies": []} for e in EpistemologyType
    }

    for agent in session.engine.population:
        if not agent.is_alive:
            continue
        for belief in BeliefSystem._get_beliefs(agent):
            entry = distribution[belief.epistemology.value]
            entry["count"] += 1
            entry["accuracies"].append(belief.accuracy)

    import numpy as np
    result: dict[str, dict[str, Any]] = {}
    for etype, data in distribution.items():
        accs = data["accuracies"]
        result[etype] = {
            "count": data["count"],
            "mean_accuracy": round(float(np.mean(accs)), 4) if accs else 0.0,
        }

    return {"enabled": True, "distribution": result}


@router.get("/{session_id}/ground-truths")
def get_ground_truths(session_id: str, request: Request) -> dict[str, Any]:
    """Return registered ground truths."""
    session = _get_session(session_id, request)
    ext = _get_epistemology_ext(session)
    if ext is None or ext.belief_system is None:
        return {"enabled": False}

    return {
        "enabled": True,
        "ground_truths": dict(ext.belief_system.ground_truths),
    }


@router.get("/{session_id}/accuracy-by-domain")
def get_accuracy_by_domain(session_id: str, request: Request) -> dict[str, Any]:
    """Return mean accuracy per belief domain."""
    session = _get_session(session_id, request)
    ext = _get_epistemology_ext(session)
    if ext is None or ext.belief_system is None:
        return {"enabled": False}

    from seldon.social.beliefs import BeliefDomain, BeliefSystem

    import numpy as np
    domain_accs: dict[str, list[float]] = {d.value: [] for d in BeliefDomain}

    for agent in session.engine.population:
        if not agent.is_alive:
            continue
        for belief in BeliefSystem._get_beliefs(agent):
            domain_accs[belief.domain.value].append(belief.accuracy)

    result: dict[str, dict[str, Any]] = {}
    for domain, accs in domain_accs.items():
        result[domain] = {
            "count": len(accs),
            "mean_accuracy": round(float(np.mean(accs)), 4) if accs else 0.0,
        }

    return {"enabled": True, "domains": result}
