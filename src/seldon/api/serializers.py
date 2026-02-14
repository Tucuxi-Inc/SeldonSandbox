"""
Serializers for converting simulation objects to JSON-safe dicts.

Handles numpy arrays, enums, and Agent dataclass fields.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from seldon.core.agent import Agent
from seldon.core.traits import TraitSystem


def _int(v) -> int:
    """Safely convert numpy int to Python int."""
    return int(v) if v is not None else 0


def _sanitize_lineage(lineage: dict) -> dict:
    """Convert genetic lineage to JSON-safe format (tuples→lists)."""
    result = {}
    for k, v in lineage.items():
        if isinstance(v, tuple):
            result[k] = list(v)
        elif isinstance(v, dict):
            result[k] = _sanitize_lineage(v)
        else:
            result[k] = v
    return result


def serialize_agent_summary(agent: Agent, trait_system: TraitSystem) -> dict[str, Any]:
    """Lightweight agent summary for list views."""
    return {
        "id": agent.id,
        "name": agent.name,
        "age": _int(agent.age),
        "generation": _int(agent.generation),
        "birth_order": _int(agent.birth_order),
        "processing_region": agent.processing_region.value,
        "suffering": round(float(agent.suffering), 4),
        "is_alive": agent.is_alive,
        "partner_id": agent.partner_id,
        "is_outsider": agent.is_outsider,
        "dominant_voice": agent.dominant_voice,
        "latest_contribution": round(
            float(agent.contribution_history[-1]) if agent.contribution_history else 0.0, 4
        ),
        "has_decisions": bool(agent.decision_history),
        "location": list(agent.location) if agent.location else None,
    }


def serialize_agent_detail(agent: Agent, trait_system: TraitSystem) -> dict[str, Any]:
    """Full agent detail for detail panel."""
    trait_names = trait_system.names()
    return {
        **serialize_agent_summary(agent, trait_system),
        "traits": {
            name: round(float(agent.traits[i]), 4)
            for i, name in enumerate(trait_names)
        },
        "traits_at_birth": {
            name: round(float(agent.traits_at_birth[i]), 4)
            for i, name in enumerate(trait_names)
        },
        "trait_history": [
            {
                name: round(float(th[i]), 4)
                for i, name in enumerate(trait_names)
            }
            for th in agent.trait_history
        ],
        "region_history": [r.value for r in agent.region_history],
        "contribution_history": [round(float(c), 4) for c in agent.contribution_history],
        "suffering_history": [round(float(s), 4) for s in agent.suffering_history],
        "parent1_id": agent.parent1_id,
        "parent2_id": agent.parent2_id,
        "children_ids": agent.children_ids,
        "relationship_status": agent.relationship_status,
        "burnout_level": round(float(agent.burnout_level), 4),
        "personal_memories": agent.personal_memories,
        "inherited_lore": agent.inherited_lore,
        "decision_history": agent.decision_history,
        "outsider_origin": agent.outsider_origin,
        "injection_generation": agent.injection_generation,
        "location_id": agent.location_id,
        "resource_holdings": dict(agent.resource_holdings),
        "cultural_memes": list(agent.cultural_memes),
        "skills": dict(agent.skills),
        # Social hierarchy (Phase 7)
        "social_status": round(float(agent.social_status), 4),
        "social_role": agent.social_role,
        "influence_score": round(float(agent.influence_score), 4),
        "mentor_id": agent.mentor_id,
        "mentee_ids": list(agent.mentee_ids),
        "social_bonds": {k: round(float(v), 4) for k, v in agent.social_bonds.items()},
        # Genetics (Phase 8)
        "genome": {k: list(v) for k, v in agent.genome.items()} if agent.genome else {},
        "epigenetic_state": dict(agent.epigenetic_state) if agent.epigenetic_state else {},
        "genetic_lineage": _sanitize_lineage(agent.genetic_lineage) if agent.genetic_lineage else {},
        # Tick-based / Needs (Phase A)
        "life_phase": agent.life_phase,
        "location": list(agent.location) if agent.location else None,
        "needs": dict(agent.needs),
        "health": round(float(agent.health), 4),
        "needs_history": agent.needs_history,
        "health_history": [round(float(h), 4) for h in agent.health_history],
        # Death info (Chunk 1)
        "death_info": agent.extension_data.get("death_info"),
    }


def serialize_agent_at_generation(
    agent: Agent, trait_system: TraitSystem, target_generation: int,
) -> dict[str, Any]:
    """Slice agent history to reconstruct state at a past generation.

    History arrays are indexed relative to birth generation:
      idx = target_generation - agent.generation
    """
    birth_gen = int(agent.generation)
    idx = target_generation - birth_gen
    trait_names = trait_system.names()

    # Clamp index
    if idx < 0:
        idx = 0
    max_idx = len(agent.trait_history) - 1 if agent.trait_history else 0
    if idx > max_idx:
        idx = max_idx

    # Traits at target generation
    if agent.trait_history and idx < len(agent.trait_history):
        traits = {
            name: round(float(agent.trait_history[idx][i]), 4)
            for i, name in enumerate(trait_names)
        }
    else:
        traits = {
            name: round(float(agent.traits[i]), 4)
            for i, name in enumerate(trait_names)
        }

    # Processing region at target
    region = agent.processing_region.value
    if agent.region_history and idx < len(agent.region_history):
        region = agent.region_history[idx].value

    # Contribution and suffering at target
    contribution = 0.0
    if agent.contribution_history and idx < len(agent.contribution_history):
        contribution = round(float(agent.contribution_history[idx]), 4)

    suffering = 0.0
    if agent.suffering_history and idx < len(agent.suffering_history):
        suffering = round(float(agent.suffering_history[idx]), 4)

    # Filter memories and decisions to target generation
    memories = [
        m for m in agent.personal_memories
        if m.get("created_generation", 0) <= target_generation
    ]
    decisions = [
        d for d in agent.decision_history
        if d.get("generation", 0) <= target_generation
    ]

    age_at_target = target_generation - birth_gen

    return {
        "id": agent.id,
        "name": agent.name,
        "age": age_at_target,
        "generation": birth_gen,
        "target_generation": target_generation,
        "birth_order": _int(agent.birth_order),
        "processing_region": region,
        "traits": traits,
        "contribution": contribution,
        "suffering": suffering,
        "is_alive": agent.is_alive or target_generation < birth_gen + len(agent.contribution_history),
        "partner_id": agent.partner_id,
        "is_outsider": agent.is_outsider,
        "dominant_voice": agent.dominant_voice,
        "personal_memories": memories,
        "decision_history": decisions,
        "relationship_status": agent.relationship_status,
    }


def serialize_family_tree(
    agent_id: str,
    all_agents: dict[str, Agent],
    trait_system: TraitSystem,
    depth_up: int = 3,
    depth_down: int = 3,
) -> dict[str, Any]:
    """Build a family tree rooted at the given agent."""
    if agent_id not in all_agents:
        return {"root": None, "ancestors": [], "descendants": []}

    root = all_agents[agent_id]

    def _node(a: Agent) -> dict[str, Any]:
        # Top 3 traits by value
        trait_names_list = trait_system.names()
        top_traits = []
        if len(a.traits) > 0:
            indexed = [(trait_names_list[i], round(float(a.traits[i]), 4)) for i in range(len(trait_names_list))]
            indexed.sort(key=lambda x: x[1], reverse=True)
            top_traits = [{"name": n, "value": v} for n, v in indexed[:3]]

        peak_contribution = round(float(max(a.contribution_history)), 4) if a.contribution_history else 0.0
        has_breakthrough = any(
            "breakthrough" in str(m.get("type", "")).lower()
            for m in a.personal_memories
        )

        return {
            "id": a.id,
            "name": a.name,
            "age": _int(a.age),
            "generation": _int(a.generation),
            "birth_order": _int(a.birth_order),
            "processing_region": a.processing_region.value,
            "is_alive": a.is_alive,
            "is_outsider": a.is_outsider,
            "gender": getattr(a, "gender", None),
            "peak_contribution": peak_contribution,
            "has_breakthrough": has_breakthrough,
            "social_role": a.social_role,
            "top_traits": top_traits,
        }

    # Walk up (ancestors)
    ancestors = []
    _collect_ancestors(root, all_agents, ancestors, depth_up)

    # Walk down (descendants)
    descendants = _collect_descendants(root, all_agents, depth_down)

    return {
        "root": _node(root),
        "ancestors": ancestors,
        "descendants": descendants,
    }


def _collect_ancestors(
    agent: Agent,
    all_agents: dict[str, Agent],
    result: list[dict],
    depth: int,
) -> None:
    """Recursively collect ancestor nodes."""
    if depth <= 0:
        return
    for pid in [agent.parent1_id, agent.parent2_id]:
        if pid and pid in all_agents:
            parent = all_agents[pid]
            node = {
                "id": parent.id,
                "name": parent.name,
                "generation": _int(parent.generation),
                "birth_order": _int(parent.birth_order),
                "processing_region": parent.processing_region.value,
                "is_alive": parent.is_alive,
                "is_outsider": parent.is_outsider,
                "child_id": agent.id,
            }
            result.append(node)
            _collect_ancestors(parent, all_agents, result, depth - 1)


def _collect_descendants(
    agent: Agent,
    all_agents: dict[str, Agent],
    depth: int,
) -> list[dict]:
    """Recursively collect descendant nodes."""
    if depth <= 0:
        return []

    result = []
    for cid in agent.children_ids:
        if cid in all_agents:
            child = all_agents[cid]
            node = {
                "id": child.id,
                "name": child.name,
                "generation": _int(child.generation),
                "birth_order": _int(child.birth_order),
                "processing_region": child.processing_region.value,
                "is_alive": child.is_alive,
                "is_outsider": child.is_outsider,
                "parent_id": agent.id,
                "children": _collect_descendants(child, all_agents, depth - 1),
            }
            result.append(node)
    return result


def serialize_metrics(metrics_obj, trait_system: TraitSystem) -> dict[str, Any]:
    """Convert GenerationMetrics to a JSON-serializable dict with named traits."""
    trait_names = trait_system.names()
    return {
        "generation": metrics_obj.generation,
        "population_size": metrics_obj.population_size,
        "births": metrics_obj.births,
        "deaths": metrics_obj.deaths,
        "breakthroughs": metrics_obj.breakthroughs,
        "pairs_formed": metrics_obj.pairs_formed,
        "trait_means": {
            name: round(float(metrics_obj.trait_means[i]), 4)
            for i, name in enumerate(trait_names)
        } if metrics_obj.trait_means is not None and len(metrics_obj.trait_means) > 0 else {},
        "trait_stds": {
            name: round(float(metrics_obj.trait_stds[i]), 4)
            for i, name in enumerate(trait_names)
        } if metrics_obj.trait_stds is not None and len(metrics_obj.trait_stds) > 0 else {},
        "trait_entropy": round(float(metrics_obj.trait_entropy), 4),
        "region_counts": metrics_obj.region_counts,
        "region_fractions": {
            k: round(float(v), 4) for k, v in metrics_obj.region_fractions.items()
        },
        "region_transitions": metrics_obj.region_transitions,
        "total_contribution": round(float(metrics_obj.total_contribution), 4),
        "mean_contribution": round(float(metrics_obj.mean_contribution), 4),
        "max_contribution": round(float(metrics_obj.max_contribution), 4),
        "mean_suffering": round(float(metrics_obj.mean_suffering), 4),
        "suffering_by_region": {
            k: round(float(v), 4) for k, v in metrics_obj.suffering_by_region.items()
        },
        "mean_age": round(float(metrics_obj.mean_age), 2),
        "age_distribution": metrics_obj.age_distribution,
        "birth_order_counts": {str(k): v for k, v in metrics_obj.birth_order_counts.items()},
        "total_memories": metrics_obj.total_memories,
        "societal_memories": metrics_obj.societal_memories,
        "myths_count": metrics_obj.myths_count,
        "outsider_count": metrics_obj.outsider_count,
        "outsider_descendant_count": metrics_obj.outsider_descendant_count,
        "dissolutions": metrics_obj.dissolutions,
        "infidelity_events": metrics_obj.infidelity_events,
        "outsiders_injected": metrics_obj.outsiders_injected,
        "dominant_voice_counts": metrics_obj.dominant_voice_counts,
        "extension_metrics": getattr(metrics_obj, "extension_metrics", {}),
    }
