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


@router.get("/{session_id}/hierarchy")
def get_hierarchy(request: Request, session_id: str, page: int = 1, page_size: int = 50):
    """Get all agents with their social status, role, and influence."""
    session = _get_session(request, session_id)
    ts = session.config.trait_system
    population = session.engine.population

    # Sort by status descending
    sorted_agents = sorted(population, key=lambda a: a.social_status, reverse=True)

    # Paginate
    start = (page - 1) * page_size
    end = start + page_size
    page_agents = sorted_agents[start:end]

    return {
        "total": len(population),
        "page": page,
        "page_size": page_size,
        "agents": [
            {
                "id": a.id,
                "name": a.name,
                "age": _int(a.age),
                "social_status": round(float(a.social_status), 4),
                "social_role": a.social_role,
                "influence_score": round(float(a.influence_score), 4),
                "processing_region": a.processing_region.value,
                "mentor_id": a.mentor_id,
                "mentee_count": len(a.mentee_ids),
                "bond_count": len(a.social_bonds),
            }
            for a in page_agents
        ],
    }


@router.get("/{session_id}/hierarchy/roles")
def get_role_distribution(request: Request, session_id: str):
    """Get role distribution counts and agent lists per role."""
    session = _get_session(request, session_id)
    population = session.engine.population

    roles: dict[str, list[dict]] = {}
    for agent in population:
        role = agent.social_role or "unassigned"
        if role not in roles:
            roles[role] = []
        roles[role].append({
            "id": agent.id,
            "name": agent.name,
            "social_status": round(float(agent.social_status), 4),
        })

    return {
        "roles": {
            role: {
                "count": len(agents),
                "agents": agents[:10],  # Top 10 per role
            }
            for role, agents in sorted(roles.items())
        },
    }


@router.get("/{session_id}/mentorship")
def get_mentorship(request: Request, session_id: str):
    """Get all active mentorship pairs with chain info."""
    session = _get_session(request, session_id)
    population = session.engine.population

    from seldon.social.mentorship import MentorshipManager
    mm = MentorshipManager(session.config)
    chains = mm.get_mentorship_chains(population)

    # Count active mentorships
    active_pairs = [
        {
            "mentor_id": a.mentor_id,
            "mentee_id": a.id,
            "mentee_name": a.name,
        }
        for a in population
        if a.mentor_id is not None
    ]

    return {
        "active_count": len(active_pairs),
        "pairs": active_pairs,
        "chains": chains,
    }


@router.get("/{session_id}/influence-map")
def get_influence_map(request: Request, session_id: str):
    """Get influence scores and top-10 most influential agents."""
    session = _get_session(request, session_id)
    population = session.engine.population

    sorted_by_influence = sorted(
        population, key=lambda a: a.influence_score, reverse=True,
    )

    top_10 = sorted_by_influence[:10]

    return {
        "total_agents": len(population),
        "mean_influence": round(
            sum(a.influence_score for a in population) / max(len(population), 1), 4,
        ),
        "top_agents": [
            {
                "id": a.id,
                "name": a.name,
                "influence_score": round(float(a.influence_score), 4),
                "social_status": round(float(a.social_status), 4),
                "social_role": a.social_role,
                "bond_count": len(a.social_bonds),
                "mentee_count": len(a.mentee_ids),
            }
            for a in top_10
        ],
    }
