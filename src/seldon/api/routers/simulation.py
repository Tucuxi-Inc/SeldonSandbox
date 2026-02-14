"""Simulation session management endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from seldon.api.schemas import (
    CloneRequest,
    CreateSessionRequest,
    RunRequest,
    SessionResponse,
    SessionSummary,
    StepRequest,
)
from seldon.core.config import ExperimentConfig
from seldon.experiment.presets import get_preset

router = APIRouter()


def _session_response(session) -> dict:
    return {
        "id": session.id,
        "name": session.name,
        "status": session.status,
        "current_generation": session.current_generation,
        "max_generations": session.max_generations,
        "population_size": len(session.engine.population),
        "config": session.config.to_dict(),
    }


@router.post("/sessions", response_model=SessionResponse)
def create_session(req: CreateSessionRequest, request: Request):
    mgr = request.app.state.session_manager

    config = None
    if req.preset:
        config = get_preset(req.preset)
    elif req.config:
        config = ExperimentConfig.from_dict(req.config)

    session = mgr.create_session(config=config, name=req.name)
    return _session_response(session)


@router.get("/sessions", response_model=list[SessionSummary])
def list_sessions(request: Request):
    mgr = request.app.state.session_manager
    return mgr.list_sessions()


@router.get("/sessions/{session_id}", response_model=SessionResponse)
def get_session(session_id: str, request: Request):
    mgr = request.app.state.session_manager
    try:
        session = mgr.get_session(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return _session_response(session)


@router.delete("/sessions/{session_id}")
def delete_session(session_id: str, request: Request):
    mgr = request.app.state.session_manager
    try:
        mgr.delete_session(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return {"deleted": True}


@router.post("/sessions/{session_id}/run", response_model=SessionResponse)
def run_session(session_id: str, req: RunRequest, request: Request):
    mgr = request.app.state.session_manager
    try:
        session = mgr.get_session(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    if req.generations is not None:
        session.max_generations = min(
            session.current_generation + req.generations,
            session.config.generations_to_run,
        )
    mgr.run_full_async(session_id)
    return _session_response(session)


@router.post("/sessions/{session_id}/step", response_model=SessionResponse)
def step_session(session_id: str, req: StepRequest, request: Request):
    mgr = request.app.state.session_manager
    try:
        mgr.step(session_id, req.n)
        session = mgr.get_session(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return _session_response(session)


@router.post("/sessions/{session_id}/clone", response_model=SessionResponse)
def clone_session(session_id: str, body: CloneRequest, request: Request):
    """Clone a session to create a what-if branch with optional config overrides."""
    mgr = request.app.state.session_manager
    try:
        session = mgr.clone_session(
            source_id=session_id,
            config_overrides=body.config_overrides,
            name=body.name,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return _session_response(session)


@router.post("/sessions/{session_id}/reset", response_model=SessionResponse)
def reset_session(session_id: str, request: Request):
    mgr = request.app.state.session_manager
    if mgr.is_running(session_id):
        raise HTTPException(status_code=409, detail="Cannot reset while running")
    try:
        session = mgr.reset_session(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return _session_response(session)
