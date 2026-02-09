"""Social network graph endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request

router = APIRouter()


@router.get("/{session_id}/graph")
def get_network_graph(
    session_id: str,
    request: Request,
    bond_threshold: float = Query(0.1, ge=0.0, le=1.0),
) -> dict[str, Any]:
    """Build a social network graph from agent relationships and bonds."""
    mgr = request.app.state.session_manager
    try:
        session = mgr.get_session(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    population = session.engine.population
    alive_ids = {a.id for a in population}

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    edge_set: set[tuple[str, str, str]] = set()  # (source, target, type) to dedupe

    for agent in population:
        nodes.append({
            "id": agent.id,
            "name": agent.name,
            "region": agent.processing_region.value,
            "location_id": agent.location_id,
        })

        # Partner edges
        if agent.partner_id and agent.partner_id in alive_ids:
            edge_key = tuple(sorted([agent.id, agent.partner_id])) + ("partner",)
            if edge_key not in edge_set:
                edge_set.add(edge_key)
                edges.append({
                    "source": agent.id,
                    "target": agent.partner_id,
                    "type": "partner",
                    "strength": 1.0,
                })

        # Social bond edges
        for bond_id, strength in agent.social_bonds.items():
            if bond_id not in alive_ids:
                continue
            if strength < bond_threshold:
                continue
            edge_key = tuple(sorted([agent.id, bond_id])) + ("social",)
            if edge_key not in edge_set:
                edge_set.add(edge_key)
                edges.append({
                    "source": agent.id,
                    "target": bond_id,
                    "type": "social",
                    "strength": round(float(strength), 4),
                })

        # Parent-child edges (only if child is alive)
        for child_id in agent.children_ids:
            if child_id not in alive_ids:
                continue
            edge_key = (agent.id, child_id, "parent")
            if edge_key not in edge_set:
                edge_set.add(edge_key)
                edges.append({
                    "source": agent.id,
                    "target": child_id,
                    "type": "parent",
                    "strength": 0.8,
                })

    # Compute stats
    connection_counts: dict[str, int] = {n["id"]: 0 for n in nodes}
    for e in edges:
        connection_counts[e["source"]] = connection_counts.get(e["source"], 0) + 1
        connection_counts[e["target"]] = connection_counts.get(e["target"], 0) + 1

    counts = list(connection_counts.values())
    avg_connections = round(sum(counts) / max(len(counts), 1), 2)

    # Connected components via union-find
    parent_map: dict[str, str] = {n["id"]: n["id"] for n in nodes}

    def find(x: str) -> str:
        while parent_map[x] != x:
            parent_map[x] = parent_map[parent_map[x]]
            x = parent_map[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent_map[ra] = rb

    for e in edges:
        union(e["source"], e["target"])

    components = len(set(find(nid) for nid in parent_map))

    return {
        "nodes": nodes,
        "edges": edges,
        "stats": {
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "avg_connections": avg_connections,
            "connected_components": components,
        },
    }
