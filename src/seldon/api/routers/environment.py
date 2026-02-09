"""
Environment API router — climate state, event history, disease tracking.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

router = APIRouter()


def _get_session(request: Request, session_id: str):
    sm = request.app.state.session_manager
    try:
        return sm.get_session(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")


def _get_environment_extension(session):
    for ext in session.engine.extensions.get_enabled():
        if ext.name == "environment":
            return ext
    return None


@router.get("/{session_id}/climate")
def get_climate(request: Request, session_id: str):
    """Current climate state per location + season."""
    session = _get_session(request, session_id)
    ext = _get_environment_extension(session)
    if ext is None:
        return {"enabled": False, "message": "Environment extension not enabled"}
    metrics = ext.get_metrics(session.engine.population)
    return {
        "enabled": True,
        "season": metrics["current_season"],
        "climate_states": metrics["climate_states"],
    }


@router.get("/{session_id}/events")
def get_event_history(request: Request, session_id: str):
    """Full event history timeline."""
    session = _get_session(request, session_id)
    ext = _get_environment_extension(session)
    if ext is None:
        return {"enabled": False, "events": []}
    return {
        "enabled": True,
        "events": ext.get_event_history(),
        "total": len(ext.get_event_history()),
    }


@router.get("/{session_id}/events/{generation}")
def get_events_for_generation(request: Request, session_id: str, generation: int):
    """Events for a specific generation."""
    session = _get_session(request, session_id)
    ext = _get_environment_extension(session)
    if ext is None:
        return {"enabled": False, "events": []}
    events = ext.get_events_for_generation(generation)
    return {"enabled": True, "generation": generation, "events": events}


@router.get("/{session_id}/disease")
def get_disease_status(request: Request, session_id: str):
    """Active diseases, infection rates, resistance stats."""
    session = _get_session(request, session_id)
    ext = _get_environment_extension(session)
    if ext is None:
        return {"enabled": False, "diseases": []}
    diseases = ext.get_disease_status(session.engine.population)
    return {
        "enabled": True,
        "active_diseases": len(diseases),
        "diseases": diseases,
    }
