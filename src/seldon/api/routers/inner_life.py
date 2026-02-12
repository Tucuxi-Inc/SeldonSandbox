"""Inner life API endpoints: PQ overview, per-agent experiences, mood map, drift."""

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


def _get_inner_life_ext(session):
    """Get the inner_life extension if enabled, else None."""
    for ext in session.engine.extensions.get_enabled():
        if ext.name == "inner_life":
            return ext
    return None


@router.get("/{session_id}/overview")
def get_overview(session_id: str, request: Request) -> dict[str, Any]:
    """Return mean PQ, PQ distribution, event counts, population mood."""
    session = _get_session(session_id, request)
    ext = _get_inner_life_ext(session)
    if ext is None or ext.experiential_engine is None:
        return {"enabled": False}

    metrics = ext.experiential_engine.get_metrics(session.engine.population)
    return {"enabled": True, **metrics}


@router.get("/{session_id}/agent/{agent_id}")
def get_agent_experiential(
    session_id: str, agent_id: str, request: Request,
) -> dict[str, Any]:
    """Return full experiential state for an agent."""
    session = _get_session(session_id, request)
    ext = _get_inner_life_ext(session)
    if ext is None or ext.experiential_engine is None:
        return {"enabled": False}

    agent = session.all_agents.get(agent_id)
    if agent is None:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")

    state = agent.extension_data.get("inner_life")
    if state is None:
        return {"enabled": True, "agent_id": agent_id, "state": None}

    # Return a safe copy (exclude _prev_state internal cache)
    safe_state = {
        k: v for k, v in state.items() if not k.startswith("_")
    }
    return {"enabled": True, "agent_id": agent_id, "state": safe_state}


@router.get("/{session_id}/phenomenal-quality-distribution")
def get_pq_distribution(session_id: str, request: Request) -> dict[str, Any]:
    """Return bucketed PQ distribution + stats."""
    session = _get_session(session_id, request)
    ext = _get_inner_life_ext(session)
    if ext is None or ext.experiential_engine is None:
        return {"enabled": False}

    import numpy as np

    pqs: list[float] = []
    for agent in session.engine.population:
        if not agent.is_alive:
            continue
        state = agent.extension_data.get("inner_life")
        if state is not None:
            pqs.append(state.get("phenomenal_quality", 0.5))

    if not pqs:
        return {"enabled": True, "distribution": {}, "stats": {}}

    buckets = {"0.0-0.2": 0, "0.2-0.4": 0, "0.4-0.6": 0, "0.6-0.8": 0, "0.8-1.0": 0}
    for pq in pqs:
        if pq < 0.2:
            buckets["0.0-0.2"] += 1
        elif pq < 0.4:
            buckets["0.2-0.4"] += 1
        elif pq < 0.6:
            buckets["0.4-0.6"] += 1
        elif pq < 0.8:
            buckets["0.6-0.8"] += 1
        else:
            buckets["0.8-1.0"] += 1

    return {
        "enabled": True,
        "distribution": buckets,
        "stats": {
            "mean": round(float(np.mean(pqs)), 4),
            "std": round(float(np.std(pqs)), 4),
            "min": round(float(np.min(pqs)), 4),
            "max": round(float(np.max(pqs)), 4),
            "count": len(pqs),
        },
    }


@router.get("/{session_id}/mood-map")
def get_mood_map(session_id: str, request: Request) -> dict[str, Any]:
    """Return per-agent mood vectors."""
    session = _get_session(session_id, request)
    ext = _get_inner_life_ext(session)
    if ext is None or ext.experiential_engine is None:
        return {"enabled": False}

    from seldon.core.experiential import EXPERIENCE_LABELS

    agents_data: list[dict[str, Any]] = []
    for agent in session.engine.population:
        if not agent.is_alive:
            continue
        state = agent.extension_data.get("inner_life")
        if state is None:
            continue
        mood = state.get("mood", [0.0] * 6)
        agents_data.append({
            "agent_id": agent.id,
            "mood": {label: round(mood[i], 4) for i, label in enumerate(EXPERIENCE_LABELS)},
            "phenomenal_quality": round(state.get("phenomenal_quality", 0.5), 4),
        })

    return {"enabled": True, "agents": agents_data}


@router.get("/{session_id}/experiential-drift")
def get_experiential_drift(session_id: str, request: Request) -> dict[str, Any]:
    """Return aggregated trait drift from experiences."""
    session = _get_session(session_id, request)
    ext = _get_inner_life_ext(session)
    if ext is None or ext.experiential_engine is None:
        return {"enabled": False}

    import numpy as np

    aggregate_drift: dict[str, list[float]] = {}
    agent_count = 0

    for agent in session.engine.population:
        if not agent.is_alive:
            continue
        state = agent.extension_data.get("inner_life")
        if state is None:
            continue
        drift = state.get("experiential_drift_applied", {})
        if not drift:
            continue
        agent_count += 1
        for trait_name, delta in drift.items():
            if trait_name not in aggregate_drift:
                aggregate_drift[trait_name] = []
            aggregate_drift[trait_name].append(delta)

    result: dict[str, dict[str, float]] = {}
    for trait_name, deltas in aggregate_drift.items():
        result[trait_name] = {
            "mean_drift": round(float(np.mean(deltas)), 6),
            "max_drift": round(float(np.max(np.abs(deltas))), 6),
            "agents_affected": len(deltas),
        }

    return {
        "enabled": True,
        "drift_by_trait": result,
        "agents_with_drift": agent_count,
    }
