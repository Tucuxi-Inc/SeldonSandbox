"""
Biography data assembly and optional LLM prose generation.

Assembles structured biography data from agent history arrays.
Optionally enriches with LLM-generated prose.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from seldon.core.agent import Agent
from seldon.core.traits import TraitSystem
from seldon.llm.chronicle import EventExtractor


class BiographyGenerator:
    """Assembles structured biography data for an agent."""

    def __init__(self) -> None:
        self._extractor = EventExtractor()

    def build_biography_data(
        self,
        agent: Agent,
        all_agents: dict[str, Agent],
        trait_system: TraitSystem,
    ) -> dict[str, Any]:
        """Build a structured biography dict with no LLM calls."""
        trait_names = trait_system.names()

        # Identity
        identity = {
            "name": agent.name,
            "age": int(agent.age),
            "generation": int(agent.generation),
            "birth_order": int(agent.birth_order),
            "is_outsider": agent.is_outsider,
            "outsider_origin": agent.outsider_origin,
            "gender": getattr(agent, "gender", None),
        }

        # Personality profile — top traits
        trait_values = {
            name: round(float(agent.traits[i]), 4)
            for i, name in enumerate(trait_names)
        }
        sorted_traits = sorted(trait_values.items(), key=lambda x: x[1], reverse=True)
        top_traits = [{"name": n, "value": v} for n, v in sorted_traits[:8]]

        region_journey = [r.value for r in agent.region_history]

        personality_profile = {
            "top_traits": top_traits,
            "processing_region": agent.processing_region.value,
            "dominant_voice": agent.dominant_voice,
            "region_journey": region_journey,
        }

        # Life timeline from event extractor
        life_timeline = self._extractor.extract_agent_timeline(
            agent, all_agents, trait_system,
        )
        timeline_dicts = [
            {
                "event_type": e.event_type,
                "generation": e.generation,
                "severity": e.severity,
                "headline": e.headline,
                "detail": e.detail,
                "agent_ids": e.agent_ids,
                "metrics_snapshot": e.metrics_snapshot,
            }
            for e in life_timeline
        ]

        # Relationships
        partner = None
        if agent.partner_id and agent.partner_id in all_agents:
            p = all_agents[agent.partner_id]
            partner = {"id": p.id, "name": p.name}

        parents = []
        for pid in [agent.parent1_id, agent.parent2_id]:
            if pid and pid in all_agents:
                pa = all_agents[pid]
                parents.append({"id": pa.id, "name": pa.name})

        children = []
        for cid in agent.children_ids:
            if cid in all_agents:
                ch = all_agents[cid]
                children.append({"id": ch.id, "name": ch.name})

        relationships = {
            "partner": partner,
            "parents": parents,
            "children": children,
        }

        # Contribution summary
        contribs = agent.contribution_history
        has_breakthrough = any(
            "breakthrough" in str(m.get("type", "")).lower()
            for m in agent.personal_memories
        )
        contribution_summary = {
            "peak": round(float(max(contribs)), 4) if contribs else 0.0,
            "mean": round(float(np.mean(contribs)), 4) if contribs else 0.0,
            "total_generations": len(contribs),
            "has_breakthrough": has_breakthrough,
        }

        # Death analysis
        death_analysis = agent.extension_data.get("death_info")

        # Agent summary for the response
        agent_summary = {
            "id": agent.id,
            "name": agent.name,
            "age": int(agent.age),
            "generation": int(agent.generation),
            "birth_order": int(agent.birth_order),
            "processing_region": agent.processing_region.value,
            "suffering": round(float(agent.suffering), 4),
            "is_alive": agent.is_alive,
            "partner_id": agent.partner_id,
            "is_outsider": agent.is_outsider,
            "dominant_voice": agent.dominant_voice,
            "latest_contribution": round(
                float(contribs[-1]) if contribs else 0.0, 4
            ),
            "has_decisions": bool(agent.decision_history),
            "location": list(agent.location) if agent.location else None,
        }

        return {
            "agent": agent_summary,
            "identity": identity,
            "personality_profile": personality_profile,
            "life_timeline": timeline_dicts,
            "relationships": relationships,
            "contribution_summary": contribution_summary,
            "death_analysis": death_analysis,
            "prose": None,
        }
