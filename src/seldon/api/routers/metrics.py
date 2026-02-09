"""Generation metrics endpoints."""

from __future__ import annotations

from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException, Query, Request

from seldon.api.schemas import SummaryResponse, TimeSeriesResponse
from seldon.api.serializers import serialize_metrics

router = APIRouter()


@router.get("/{session_id}/generations")
def get_generations(
    session_id: str,
    request: Request,
    from_gen: int = Query(0, ge=0),
    to_gen: int | None = Query(None),
) -> list[dict[str, Any]]:
    mgr = request.app.state.session_manager
    try:
        session = mgr.get_session(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    metrics = session.collector.metrics_history
    end = to_gen if to_gen is not None else len(metrics)
    ts = session.config.trait_system

    return [serialize_metrics(m, ts) for m in metrics[from_gen:end]]


@router.get("/{session_id}/time-series/{field_name}", response_model=TimeSeriesResponse)
def get_time_series(session_id: str, field_name: str, request: Request):
    mgr = request.app.state.session_manager
    try:
        session = mgr.get_session(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    collector = session.collector
    try:
        values = collector.get_time_series(field_name)
    except AttributeError:
        raise HTTPException(status_code=400, detail=f"Unknown metric field: '{field_name}'")

    # Convert numpy types to Python scalars
    safe_values = []
    for v in values:
        if isinstance(v, np.ndarray):
            safe_values.append(v.tolist())
        elif isinstance(v, (np.integer, np.floating)):
            safe_values.append(v.item())
        else:
            safe_values.append(v)

    return {
        "field": field_name,
        "generations": list(range(len(values))),
        "values": safe_values,
    }


@router.get("/{session_id}/summary", response_model=SummaryResponse)
def get_summary(session_id: str, request: Request):
    mgr = request.app.state.session_manager
    try:
        session = mgr.get_session(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    history = session.collector.metrics_history
    if not history:
        return {
            "total_generations": 0,
            "final_population_size": len(session.engine.population),
            "total_breakthroughs": 0,
            "mean_contribution": 0.0,
            "mean_suffering": 0.0,
            "peak_population": len(session.engine.population),
            "total_births": 0,
            "total_deaths": 0,
        }

    return {
        "total_generations": len(history),
        "final_population_size": history[-1].population_size,
        "total_breakthroughs": sum(m.breakthroughs for m in history),
        "mean_contribution": round(
            float(np.mean([m.mean_contribution for m in history if m.population_size > 0])), 4
        ),
        "mean_suffering": round(
            float(np.mean([m.mean_suffering for m in history if m.population_size > 0])), 4
        ),
        "peak_population": max(m.population_size for m in history),
        "total_births": sum(m.births for m in history),
        "total_deaths": sum(m.deaths for m in history),
    }
