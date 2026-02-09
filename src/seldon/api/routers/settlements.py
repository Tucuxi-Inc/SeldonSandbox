"""Settlement diagnostics and migration flow endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request

from seldon.api.serializers import _int

router = APIRouter()


def _get_geography(session):
    """Get geography extension if enabled, or None."""
    if session.engine.extensions is None:
        return None
    if not session.engine.extensions.is_enabled("geography"):
        return None
    return session.engine.extensions.get("geography")


def _get_migration(session):
    """Get migration extension if enabled, or None."""
    if session.engine.extensions is None:
        return None
    if not session.engine.extensions.is_enabled("migration"):
        return None
    return session.engine.extensions.get("migration")


@router.get("/{session_id}/overview")
def get_settlements_overview(session_id: str, request: Request) -> dict[str, Any]:
    """Overview of all settlements with population and region breakdown."""
    mgr = request.app.state.session_manager
    try:
        session = mgr.get_session(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    geo = _get_geography(session)
    if geo is None:
        return {"enabled": False}

    settlements: list[dict[str, Any]] = []
    for loc_id, loc in geo.locations.items():
        # Get agents at this location
        agents_here = [a for a in session.engine.population if a.location_id == loc_id]

        # Region breakdown
        region_counts: dict[str, int] = {}
        for a in agents_here:
            r = a.processing_region.value
            region_counts[r] = region_counts.get(r, 0) + 1

        settlements.append({
            "id": loc_id,
            "name": loc.name,
            "coordinates": list(loc.coordinates),
            "population": len(agents_here),
            "carrying_capacity": loc.carrying_capacity,
            "occupancy_ratio": round(len(agents_here) / max(loc.carrying_capacity, 1), 4),
            "resource_richness": round(loc.resource_richness, 4),
            "region_counts": region_counts,
        })

    total_capacity = sum(loc.carrying_capacity for loc in geo.locations.values())
    total_population = len(session.engine.population)

    return {
        "enabled": True,
        "settlements": settlements,
        "total_capacity": total_capacity,
        "total_population": total_population,
    }


@router.get("/{session_id}/viability/{location_id}")
def get_settlement_viability(
    session_id: str, location_id: str, request: Request,
) -> dict[str, Any]:
    """Evaluate viability of agents at a specific settlement."""
    mgr = request.app.state.session_manager
    try:
        session = mgr.get_session(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    migration = _get_migration(session)
    if migration is None:
        raise HTTPException(status_code=400, detail="Migration extension not enabled")

    geo = _get_geography(session)
    if geo is None or location_id not in geo.locations:
        raise HTTPException(status_code=404, detail=f"Location '{location_id}' not found")

    agents_here = [a for a in session.engine.population if a.location_id == location_id]
    if not agents_here:
        return {
            "location_id": location_id,
            "viability_score": 0.0,
            "risk_factors": ["empty_settlement"],
            "checks_passed": 0,
            "checks_total": 7,
            "group_size": 0,
        }

    viability, risks = migration.evaluate_settlement_viability(agents_here, session.config)

    return {
        "location_id": location_id,
        "viability_score": round(viability, 4),
        "risk_factors": risks,
        "checks_passed": int(round(viability * 7)),
        "checks_total": 7,
        "group_size": len(agents_here),
    }


@router.get("/{session_id}/migration-history")
def get_migration_history(session_id: str, request: Request) -> dict[str, Any]:
    """Migration event timeline across generations."""
    mgr = request.app.state.session_manager
    try:
        session = mgr.get_session(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    geo = _get_geography(session)
    if geo is None:
        return {"enabled": False}

    history = session.collector.metrics_history
    all_events: list[dict[str, Any]] = []
    settlement_count_by_gen: list[int] = []
    migrations_by_gen: list[int] = []

    for i, m in enumerate(history):
        mig_data = m.extension_metrics.get("migration", {})
        events = mig_data.get("events", [])
        for event in events:
            all_events.append({**event, "generation": i})

        geo_data = m.extension_metrics.get("geography", {})
        settlement_count_by_gen.append(geo_data.get("settlement_count", 0))
        migrations_by_gen.append(mig_data.get("migration_events", 0))

    return {
        "enabled": True,
        "events": all_events,
        "timeline": {
            "settlement_count_by_gen": settlement_count_by_gen,
            "migrations_by_gen": migrations_by_gen,
        },
    }


@router.get("/{session_id}/settlement-composition/{location_id}")
def get_settlement_composition(
    session_id: str, location_id: str, request: Request,
) -> dict[str, Any]:
    """Detailed composition of a single settlement."""
    mgr = request.app.state.session_manager
    try:
        session = mgr.get_session(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    geo = _get_geography(session)
    if geo is None or location_id not in geo.locations:
        raise HTTPException(status_code=404, detail=f"Location '{location_id}' not found")

    loc = geo.locations[location_id]
    agents_here = [a for a in session.engine.population if a.location_id == location_id]
    ts = session.config.trait_system
    trait_names = ts.names()

    if not agents_here:
        return {
            "location_id": location_id,
            "name": loc.name,
            "population": 0,
            "trait_means": {},
            "trait_stds": {},
            "region_percentages": {},
            "mean_age": 0.0,
            "paired_count": 0,
            "single_count": 0,
        }

    import numpy as np

    traits_matrix = np.array([a.traits for a in agents_here])
    trait_means = {name: round(float(traits_matrix[:, i].mean()), 4) for i, name in enumerate(trait_names)}
    trait_stds = {name: round(float(traits_matrix[:, i].std()), 4) for i, name in enumerate(trait_names)}

    # Region percentages
    region_counts: dict[str, int] = {}
    for a in agents_here:
        r = a.processing_region.value
        region_counts[r] = region_counts.get(r, 0) + 1
    n = len(agents_here)
    region_pcts = {r: round(c / n, 4) for r, c in region_counts.items()}

    mean_age = round(float(np.mean([_int(a.age) for a in agents_here])), 2)
    paired = sum(1 for a in agents_here if a.partner_id is not None)

    return {
        "location_id": location_id,
        "name": loc.name,
        "population": n,
        "carrying_capacity": loc.carrying_capacity,
        "trait_means": trait_means,
        "trait_stds": trait_stds,
        "region_percentages": region_pcts,
        "mean_age": mean_age,
        "paired_count": paired,
        "single_count": n - paired,
    }
