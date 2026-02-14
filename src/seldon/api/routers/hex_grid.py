"""Hex grid map endpoints: full grid view, tile detail, and tick stepping."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request

from seldon.api.serializers import serialize_agent_summary

router = APIRouter()


# ---------------------------------------------------------------------------
# World View: Tick-level stepping
# ---------------------------------------------------------------------------

@router.post("/{session_id}/step-tick")
def step_tick(session_id: str, request: Request) -> dict[str, Any]:
    """Advance the simulation by one tick (1/12 of a year).

    Only works with tick-engine sessions (tick_config.enabled=True).
    Returns the activity log for world-view visualization.
    """
    mgr = request.app.state.session_manager
    try:
        return mgr.step_tick(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/{session_id}/tick-state")
def get_tick_state(session_id: str, request: Request) -> dict[str, Any]:
    """Current tick state without advancing (for page load / reconnect).

    Returns ``{"enabled": false}`` if the session does not use the tick engine.
    """
    mgr = request.app.state.session_manager
    try:
        session = mgr.get_session(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    if not hasattr(session.engine, "_run_single_tick"):
        return {"enabled": False}

    return mgr._build_tick_response(session)


def _get_hex_grid(session):
    """Get the hex grid from a session's engine, or None."""
    return getattr(session.engine, "hex_grid", None)


@router.get("/{session_id}/grid")
def get_hex_grid(session_id: str, request: Request) -> dict[str, Any]:
    """Full grid with tiles, agent counts, and cluster data."""
    mgr = request.app.state.session_manager
    try:
        session = mgr.get_session(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    grid = _get_hex_grid(session)
    if grid is None:
        return {"enabled": False}

    ts = session.config.trait_system

    tiles = []
    for tile in grid.tiles.values():
        # Get agent region counts for this tile
        region_counts: dict[str, int] = {}
        agent_summaries = []
        for aid in tile.current_agents:
            agent = session.all_agents.get(aid)
            if agent and agent.is_alive:
                region = agent.processing_region.value
                region_counts[region] = region_counts.get(region, 0) + 1
                if len(agent_summaries) < 10:  # Cap summaries per tile
                    agent_summaries.append({
                        "id": agent.id,
                        "name": agent.name,
                        "processing_region": region,
                    })

        tiles.append({
            "q": tile.q,
            "r": tile.r,
            "terrain_type": tile.terrain_type.value,
            "elevation": round(tile.elevation, 2),
            "water_access": round(tile.water_access, 2),
            "soil_quality": round(tile.soil_quality, 2),
            "natural_resources": round(tile.natural_resources, 2),
            "vegetation": round(tile.vegetation, 2),
            "wildlife": round(tile.wildlife, 2),
            "habitability": round(tile.habitability, 2),
            "capacity": tile.capacity,
            "agent_count": len(tile.current_agents),
            "agent_ids": list(tile.current_agents),
            "region_counts": region_counts,
            "agents": agent_summaries,
        })

    # Cluster data
    clusters = []
    if hasattr(grid, "get_agent_clusters"):
        raw_clusters = grid.get_agent_clusters(min_size=2)
        for c in raw_clusters:
            clusters.append({
                "tiles": [list(t) for t in c.get("tiles", [])],
                "agent_count": c.get("agent_count", 0),
                "agent_ids": c.get("agent_ids", []),
                "center": list(c.get("center", (0, 0))),
                "terrain_types": c.get("terrain_types", []),
            })

    # Grid stats
    all_tiles = list(grid.tiles.values())
    habitable_count = sum(1 for t in all_tiles if t.is_habitable)
    occupied_count = sum(1 for t in all_tiles if len(t.current_agents) > 0)
    total_agents = sum(len(t.current_agents) for t in all_tiles)

    return {
        "enabled": True,
        "tiles": tiles,
        "clusters": clusters,
        "stats": {
            "total_tiles": len(all_tiles),
            "habitable_tiles": habitable_count,
            "occupied_tiles": occupied_count,
            "total_agents_on_grid": total_agents,
            "width": grid.width,
            "height": grid.height,
        },
    }


@router.get("/{session_id}/tile/{q}/{r}")
def get_tile_detail(
    session_id: str, q: int, r: int, request: Request,
) -> dict[str, Any]:
    """Detailed view of a single tile including full agent summaries."""
    mgr = request.app.state.session_manager
    try:
        session = mgr.get_session(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    grid = _get_hex_grid(session)
    if grid is None:
        return {"enabled": False}

    tile = grid.tiles.get((q, r))
    if tile is None:
        raise HTTPException(status_code=404, detail=f"Tile ({q}, {r}) not found")

    ts = session.config.trait_system

    # Full agent summaries
    agents = []
    for aid in tile.current_agents:
        agent = session.all_agents.get(aid)
        if agent:
            agents.append(serialize_agent_summary(agent, ts))

    # Neighbor tiles
    neighbors = []
    for nq, nr in grid.neighbors(q, r):
        ntile = grid.tiles.get((nq, nr))
        if ntile:
            neighbors.append({
                "q": nq,
                "r": nr,
                "terrain_type": ntile.terrain_type.value,
                "agent_count": len(ntile.current_agents),
                "habitability": round(ntile.habitability, 2),
            })

    return {
        "enabled": True,
        "tile": {
            "q": tile.q,
            "r": tile.r,
            "terrain_type": tile.terrain_type.value,
            "elevation": round(tile.elevation, 2),
            "water_access": round(tile.water_access, 2),
            "soil_quality": round(tile.soil_quality, 2),
            "natural_resources": round(tile.natural_resources, 2),
            "vegetation": round(tile.vegetation, 2),
            "wildlife": round(tile.wildlife, 2),
            "habitability": round(tile.habitability, 2),
            "capacity": tile.capacity,
            "agent_count": len(tile.current_agents),
            "is_habitable": tile.is_habitable,
        },
        "agents": agents,
        "neighbors": neighbors,
    }
