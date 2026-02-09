"""
Economics API router — trade routes, markets, wealth distribution.
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


def _get_economics_extension(session):
    for ext in session.engine.extensions.get_enabled():
        if ext.name == "economics":
            return ext
    return None


@router.get("/{session_id}/overview")
def economics_overview(request: Request, session_id: str):
    """GDP, trade volume, Gini, occupation distribution."""
    session = _get_session(request, session_id)
    ext = _get_economics_extension(session)
    if ext is None:
        return {"enabled": False, "message": "Economics extension not enabled"}
    metrics = ext.get_metrics(session.engine.population)
    return {"enabled": True, **metrics}


@router.get("/{session_id}/trade-routes")
def get_trade_routes(request: Request, session_id: str):
    """All active trade routes with volumes."""
    session = _get_session(request, session_id)
    ext = _get_economics_extension(session)
    if ext is None:
        return {"enabled": False, "routes": []}
    return {"enabled": True, "routes": ext.get_trade_routes_list()}


@router.get("/{session_id}/markets")
def get_markets(request: Request, session_id: str):
    """Supply/demand/prices per community."""
    session = _get_session(request, session_id)
    ext = _get_economics_extension(session)
    if ext is None:
        return {"enabled": False, "markets": []}
    return {"enabled": True, "markets": ext.get_market_data()}


@router.get("/{session_id}/wealth-distribution")
def get_wealth_distribution(request: Request, session_id: str):
    """Population-wide wealth percentiles + Lorenz curve."""
    session = _get_session(request, session_id)
    ext = _get_economics_extension(session)
    if ext is None:
        return {"enabled": False, "distribution": {}}
    dist = ext.get_wealth_distribution(session.engine.population)
    return {"enabled": True, **dist}


@router.get("/{session_id}/occupations")
def get_occupations(request: Request, session_id: str):
    """Occupation distribution across population."""
    session = _get_session(request, session_id)
    population = session.engine.population
    occ_counts: dict[str, int] = {}
    for a in population:
        if a.is_alive:
            occ = a.occupation or "unassigned"
            occ_counts[occ] = occ_counts.get(occ, 0) + 1
    return {"occupations": occ_counts, "total": sum(occ_counts.values())}
