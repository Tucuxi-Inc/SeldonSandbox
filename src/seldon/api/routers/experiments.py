"""Experiment-related endpoints: presets, archetypes, comparison, injection."""

from __future__ import annotations

from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException, Request

from seldon.api.schemas import (
    ArchetypeDetailResponse,
    ArchetypeInfo,
    CompareRequest,
    ComparisonResponse,
    InjectRequest,
    PresetInfo,
    SummaryResponse,
)
from seldon.api.serializers import serialize_agent_summary
from seldon.experiment.archetypes import ARCHETYPES, get_archetype, list_archetypes
from seldon.experiment.presets import get_preset, list_presets

router = APIRouter()


@router.get("/presets", response_model=list[PresetInfo])
def get_presets():
    return [
        {"name": name, "config": get_preset(name).to_dict()}
        for name in list_presets()
    ]


@router.get("/archetypes", response_model=list[ArchetypeInfo])
def get_archetypes_list():
    result = []
    for name in list_archetypes():
        arch = get_archetype(name)
        result.append({
            "name": name,
            "display_name": arch.name,
            "description": arch.description,
            "key_traits": arch.key_traits,
            "use_case": arch.use_case,
        })
    return result


@router.get("/archetypes/{name}", response_model=ArchetypeDetailResponse)
def get_archetype_detail(name: str):
    try:
        arch = get_archetype(name)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Archetype '{name}' not found")
    return {
        "name": name,
        "display_name": arch.name,
        "description": arch.description,
        "key_traits": arch.key_traits,
        "use_case": arch.use_case,
        "trait_values": arch.trait_values,
    }


@router.post("/compare", response_model=ComparisonResponse)
def compare_sessions(req: CompareRequest, request: Request):
    mgr = request.app.state.session_manager

    sessions_data: dict[str, SummaryResponse] = {}
    configs: dict[str, dict] = {}

    for sid in req.session_ids:
        try:
            session = mgr.get_session(sid)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Session '{sid}' not found")

        history = session.collector.metrics_history
        if history:
            sessions_data[sid] = {
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
        else:
            sessions_data[sid] = {
                "total_generations": 0,
                "final_population_size": len(session.engine.population),
                "total_breakthroughs": 0,
                "mean_contribution": 0.0,
                "mean_suffering": 0.0,
                "peak_population": len(session.engine.population),
                "total_births": 0,
                "total_deaths": 0,
            }
        configs[sid] = session.config.to_dict()

    # Compute config diffs between all pairs
    config_diffs: dict[str, dict] = {}
    sids = req.session_ids
    for i in range(len(sids)):
        for j in range(i + 1, len(sids)):
            key = f"{sids[i]}_vs_{sids[j]}"
            diffs = {}
            for param in configs[sids[i]]:
                v1 = configs[sids[i]][param]
                v2 = configs[sids[j]][param]
                if v1 != v2:
                    diffs[param] = [v1, v2]
            config_diffs[key] = diffs

    return {
        "sessions": sessions_data,
        "config_diffs": config_diffs,
    }


@router.get("/trait-names")
def get_trait_names(request: Request) -> list[str]:
    """Return ordered list of trait names from the default trait system."""
    from seldon.core.traits import TraitSystem
    from seldon.core.config import ExperimentConfig
    ts = ExperimentConfig().trait_system
    return ts.names()


@router.post("/inject-outsider")
def inject_outsider(req: InjectRequest, request: Request) -> dict[str, Any]:
    mgr = request.app.state.session_manager
    try:
        session = mgr.get_session(req.session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session '{req.session_id}' not found")

    engine = session.engine
    ts = session.config.trait_system
    gen = req.injection_generation if req.injection_generation is not None else session.current_generation

    if req.archetype:
        agent = engine.outsider_interface.inject_archetype(
            req.archetype,
            gen,
            noise_sigma=req.noise_sigma,
            rng=engine.rng,
        )
    elif req.custom_traits:
        traits = np.full(ts.count, 0.5)
        for trait_name, value in req.custom_traits.items():
            try:
                idx = ts.trait_index(trait_name)
                traits[idx] = value
            except KeyError:
                pass
        agent = engine.outsider_interface.inject_outsider(
            traits, gen, origin="api_custom",
        )
    else:
        raise HTTPException(status_code=400, detail="Must provide archetype or custom_traits")

    # Apply optional overrides
    if req.name:
        agent.name = req.name
    if req.gender:
        agent.gender = req.gender
    if req.age is not None:
        agent.age = req.age

    engine.population.append(agent)
    session.all_agents[agent.id] = agent

    # Track in ripple tracker
    for rec in engine.outsider_interface.injections:
        if rec.agent_id == agent.id:
            engine.ripple_tracker.track_injection(rec)
            break

    return serialize_agent_summary(agent, ts)


@router.get("/{session_id}/outsiders")
def get_outsiders(session_id: str, request: Request) -> list[dict[str, Any]]:
    """Return all outsider agents in a session."""
    mgr = request.app.state.session_manager
    try:
        session = mgr.get_session(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    ts = session.config.trait_system
    outsiders = [
        a for a in session.all_agents.values() if a.is_outsider
    ]
    return [serialize_agent_summary(a, ts) for a in outsiders]


@router.get("/{session_id}/outsiders/{agent_id}/impact")
def get_outsider_impact(
    session_id: str, agent_id: str, request: Request,
) -> dict[str, Any]:
    """Return impact data for a specific outsider."""
    mgr = request.app.state.session_manager
    try:
        session = mgr.get_session(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    if agent_id not in session.all_agents:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")

    agent = session.all_agents[agent_id]
    if not agent.is_outsider:
        raise HTTPException(status_code=400, detail="Agent is not an outsider")

    ts = session.config.trait_system

    # Collect descendants recursively
    def _collect_descendants(aid: str) -> list[str]:
        a = session.all_agents.get(aid)
        if not a:
            return []
        result = []
        for cid in a.children_ids:
            result.append(cid)
            result.extend(_collect_descendants(cid))
        return result

    descendant_ids = _collect_descendants(agent_id)
    descendants = [
        serialize_agent_summary(session.all_agents[d], ts)
        for d in descendant_ids if d in session.all_agents
    ]

    # Compute trait distance from population mean
    pop_traits = np.array([a.traits for a in session.engine.population if a.is_alive])
    trait_distance = 0.0
    if len(pop_traits) > 0 and agent.is_alive:
        pop_mean = pop_traits.mean(axis=0)
        trait_distance = float(np.linalg.norm(agent.traits - pop_mean))

    return {
        "agent": serialize_agent_summary(agent, ts),
        "descendant_count": len(descendant_ids),
        "descendants": descendants,
        "trait_distance_from_mean": round(trait_distance, 4),
        "injection_generation": agent.injection_generation,
        "outsider_origin": agent.outsider_origin,
        "gender": agent.gender,
    }


@router.get("/{session_id}/ripple")
def get_ripple_report(session_id: str, request: Request) -> dict[str, Any]:
    mgr = request.app.state.session_manager
    try:
        session = mgr.get_session(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    return session.engine.ripple_tracker.get_diffusion_report()
