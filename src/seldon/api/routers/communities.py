"""
Communities API router — community profiles, diplomacy, and comparison.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from seldon.social.community import CommunityManager

router = APIRouter()


def _get_session(request: Request, session_id: str):
    """Helper to get a session or raise 404."""
    sm = request.app.state.session_manager
    try:
        return sm.get_session(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")


def _get_diplomacy_extension(session):
    """Get the diplomacy extension from the session's engine, or None."""
    for ext in session.engine.extensions.get_enabled():
        if ext.name == "diplomacy":
            return ext
    return None


@router.get("/{session_id}/communities")
def list_communities(request: Request, session_id: str):
    """List all communities with personality profiles."""
    session = _get_session(request, session_id)
    config = session.config
    cm = CommunityManager(config)
    population = session.engine.population
    communities = cm.get_communities(population)

    result = []
    for cid, agents in communities.items():
        profile = cm.compute_personality_profile(agents)
        cohesion = cm.compute_cohesion(agents)
        identity = cm.compute_identity(agents, communities, cid)
        result.append({
            "community_id": cid,
            "profile": profile,
            "cohesion": round(cohesion, 4),
            "identity": identity,
        })

    return {
        "community_count": len(result),
        "communities": result,
    }


@router.get("/{session_id}/communities/{community_id}")
def get_community_detail(request: Request, session_id: str, community_id: str):
    """Detailed community profile with member stats."""
    session = _get_session(request, session_id)
    config = session.config
    cm = CommunityManager(config)
    population = session.engine.population
    communities = cm.get_communities(population)

    if community_id not in communities:
        raise HTTPException(status_code=404, detail=f"Community '{community_id}' not found")

    agents = communities[community_id]
    profile = cm.compute_personality_profile(agents)
    cohesion = cm.compute_cohesion(agents)
    identity = cm.compute_identity(agents, communities, community_id)

    # Top members by status
    sorted_agents = sorted(agents, key=lambda a: a.social_status, reverse=True)
    top_members = [
        {
            "id": a.id,
            "name": a.name,
            "social_status": round(float(a.social_status), 4),
            "social_role": a.social_role,
            "influence_score": round(float(a.influence_score), 4),
        }
        for a in sorted_agents[:10]
    ]

    return {
        "community_id": community_id,
        "profile": profile,
        "cohesion": round(cohesion, 4),
        "identity": identity,
        "top_members": top_members,
    }


@router.get("/{session_id}/communities/{community_id}/factions")
def get_community_factions(request: Request, session_id: str, community_id: str):
    """Internal faction analysis for a community."""
    session = _get_session(request, session_id)
    config = session.config
    cm = CommunityManager(config)
    population = session.engine.population
    communities = cm.get_communities(population)

    if community_id not in communities:
        raise HTTPException(status_code=404, detail=f"Community '{community_id}' not found")

    agents = communities[community_id]
    factions = cm.detect_factions(agents)

    return {
        "community_id": community_id,
        "faction_count": len(factions),
        "factions": factions,
    }


@router.get("/{session_id}/diplomacy")
def get_diplomacy(request: Request, session_id: str):
    """Full diplomatic relations graph."""
    session = _get_session(request, session_id)
    ext = _get_diplomacy_extension(session)

    if ext is None:
        return {
            "enabled": False,
            "message": "Diplomacy extension not enabled",
            "relations": [],
        }

    return {
        "enabled": True,
        "relations": ext.get_all_relations(),
        "metrics": ext.get_metrics(session.engine.population),
    }


@router.get("/{session_id}/diplomacy/compare")
def compare_communities(request: Request, session_id: str):
    """Side-by-side community personality comparison."""
    session = _get_session(request, session_id)
    config = session.config
    cm = CommunityManager(config)
    population = session.engine.population
    communities = cm.get_communities(population)

    comparisons = []
    community_ids = list(communities.keys())
    for i, cid_a in enumerate(community_ids):
        for cid_b in community_ids[i + 1:]:
            compat = cm.trait_compatibility(communities[cid_a], communities[cid_b])
            comparisons.append({
                "community_a": cid_a,
                "community_b": cid_b,
                "trait_compatibility": round(compat, 4),
            })

    return {
        "community_count": len(community_ids),
        "comparisons": comparisons,
    }
