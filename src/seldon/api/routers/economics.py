"""
Economics API router — trade routes, markets, wealth distribution, settlements.
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
    """All active trade routes with volumes and goods flow."""
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


# ------------------------------------------------------------------
# Phase C: Settlement economy endpoints
# ------------------------------------------------------------------

@router.get("/{session_id}/settlements")
def get_settlements(request: Request, session_id: str):
    """Per-settlement economy data: resources, GDP, governance, infrastructure."""
    session = _get_session(request, session_id)
    ext = _get_economics_extension(session)
    if ext is None:
        return {"enabled": False, "settlements": []}
    return {"enabled": True, "settlements": ext.get_settlements_data()}


@router.get("/{session_id}/settlements/{settlement_id}")
def get_settlement_detail(
    request: Request, session_id: str, settlement_id: str,
):
    """Detailed settlement data including agent list and tile coords."""
    session = _get_session(request, session_id)
    ext = _get_economics_extension(session)
    if ext is None:
        return {"enabled": False, "message": "Economics extension not enabled"}
    detail = ext.get_settlement_detail(settlement_id)
    if detail is None:
        raise HTTPException(
            status_code=404,
            detail=f"Settlement '{settlement_id}' not found",
        )
    return {"enabled": True, **detail}


@router.get("/{session_id}/skills")
def get_skills(request: Request, session_id: str):
    """Population skill distribution by occupation."""
    session = _get_session(request, session_id)
    ext = _get_economics_extension(session)
    if ext is None:
        return {"enabled": False, "skills": {}}
    skills = ext.get_skill_distribution(session.engine.population)
    return {"enabled": True, "skills": skills}


@router.get("/{session_id}/production")
def get_production(request: Request, session_id: str):
    """Total production by resource type and settlement."""
    session = _get_session(request, session_id)
    ext = _get_economics_extension(session)
    if ext is None:
        return {"enabled": False, "production": {}}
    production = ext.get_production_summary()
    return {"enabled": True, **production}
