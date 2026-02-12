"""
Social dynamics API router — hierarchy, mentorship, and influence endpoints.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from seldon.api.serializers import _int

router = APIRouter()


def _get_session(request: Request, session_id: str):
    """Helper to get a session or raise 404."""
    sm = request.app.state.session_manager
    try:
        return sm.get_session(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")


def _get_social_dynamics_ext(session):
    """Get the social_dynamics extension or None."""
    for ext in session.engine.extensions.get_enabled():
        if ext.name == "social_dynamics":
            return ext
    return None


def _is_social_dynamics_enabled(session) -> bool:
    """Check if the social_dynamics extension is active."""
    return _get_social_dynamics_ext(session) is not None


@router.get("/{session_id}/hierarchy")
def get_hierarchy(request: Request, session_id: str):
    """Get social hierarchy overview: status distribution and mean status."""
    session = _get_session(request, session_id)

    if not _is_social_dynamics_enabled(session):
        return {"enabled": False}

    population = session.engine.population

    # Compute status distribution (bucketed)
    buckets = {"0.0-0.2": 0, "0.2-0.4": 0, "0.4-0.6": 0, "0.6-0.8": 0, "0.8-1.0": 0}
    total_status = 0.0
    for a in population:
        s = float(a.social_status)
        total_status += s
        if s < 0.2:
            buckets["0.0-0.2"] += 1
        elif s < 0.4:
            buckets["0.2-0.4"] += 1
        elif s < 0.6:
            buckets["0.4-0.6"] += 1
        elif s < 0.8:
            buckets["0.6-0.8"] += 1
        else:
            buckets["0.8-1.0"] += 1

    mean_status = total_status / max(len(population), 1)

    return {
        "enabled": True,
        "status_distribution": buckets,
        "mean_status": round(mean_status, 4),
    }


@router.get("/{session_id}/roles")
def get_role_distribution(request: Request, session_id: str):
    """Get role distribution counts."""
    session = _get_session(request, session_id)

    if not _is_social_dynamics_enabled(session):
        return {"enabled": False, "roles": {}}

    population = session.engine.population
    roles: dict[str, int] = {}
    for agent in population:
        role = agent.social_role or "unassigned"
        roles[role] = roles.get(role, 0) + 1

    return {
        "enabled": True,
        "roles": roles,
    }


@router.get("/{session_id}/mentorship")
def get_mentorship(request: Request, session_id: str):
    """Get all active mentorship pairs with chain info."""
    session = _get_session(request, session_id)

    if not _is_social_dynamics_enabled(session):
        return {"enabled": False, "chains": []}

    population = session.engine.population

    # Build mentor → mentees mapping
    mentor_map: dict[str, list[dict]] = {}
    agent_lookup = {a.id: a for a in population}
    for a in population:
        if a.mentor_id is not None and a.mentor_id in agent_lookup:
            if a.mentor_id not in mentor_map:
                mentor_map[a.mentor_id] = []
            mentor_map[a.mentor_id].append({"id": a.id, "name": a.name})

    chains = [
        {
            "mentor_id": mid,
            "mentor_name": agent_lookup[mid].name,
            "mentees": mentees,
        }
        for mid, mentees in mentor_map.items()
    ]

    return {
        "enabled": True,
        "chains": chains,
    }


@router.get("/{session_id}/influence-map")
def get_influence_map(request: Request, session_id: str):
    """Get influence scores and top agents."""
    session = _get_session(request, session_id)

    if not _is_social_dynamics_enabled(session):
        return {"enabled": False, "agents": []}

    population = session.engine.population

    sorted_by_influence = sorted(
        population, key=lambda a: a.influence_score, reverse=True,
    )

    return {
        "enabled": True,
        "agents": [
            {
                "agent_id": a.id,
                "agent_name": a.name,
                "influence_score": round(float(a.influence_score), 4),
                "social_role": a.social_role,
            }
            for a in sorted_by_influence[:30]
        ],
    }


@router.get("/{session_id}/marriages")
def get_marriages(request: Request, session_id: str):
    """Get marriage statistics and political marriages."""
    session = _get_session(request, session_id)
    ext = _get_social_dynamics_ext(session)

    if ext is None or ext._marriage is None:
        return {"enabled": False}

    population = session.engine.population
    generation = session.current_generation

    stats = ext._marriage.get_marriage_stats(population, generation)
    political = ext._marriage.get_political_marriages(population)

    return {
        "enabled": True,
        "married_count": stats["married_count"],
        "avg_duration": round(stats["avg_duration"], 2),
        "total_divorces": stats["total_divorces"],
        "political_marriages": political,
    }


@router.get("/{session_id}/clans")
def get_clans(request: Request, session_id: str):
    """Get clan data: membership, honor, and rivalries."""
    session = _get_session(request, session_id)
    ext = _get_social_dynamics_ext(session)

    if ext is None or ext._clans is None:
        return {"enabled": False}

    clans = ext._clans.get_clan_data()

    return {
        "enabled": True,
        "clans": clans,
    }


@router.get("/{session_id}/institutions")
def get_institutions(request: Request, session_id: str):
    """Get institution data: councils, guilds, leaders, and prestige."""
    session = _get_session(request, session_id)
    ext = _get_social_dynamics_ext(session)

    if ext is None or ext._institutions is None:
        return {"enabled": False}

    institutions = ext._institutions.get_institution_data()

    return {
        "enabled": True,
        "institutions": institutions,
    }
