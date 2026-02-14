"""
Event extraction system for structured narratives.

Detects notable events from simulation data without requiring LLM calls.
Events are used by the Chronicle and Biography views.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from seldon.core.agent import Agent
from seldon.core.traits import TraitSystem
from seldon.metrics.collector import GenerationMetrics


@dataclass
class NotableEvent:
    """A notable event extracted from simulation data."""

    event_type: str  # "premature_death", "breakthrough", "population_boom", etc.
    generation: int
    severity: str  # "minor" | "notable" | "major" | "critical"
    headline: str  # Structured text, no LLM needed
    detail: str
    agent_ids: list[str] = field(default_factory=list)
    metrics_snapshot: dict[str, Any] = field(default_factory=dict)


# Severity ordering for comparison
SEVERITY_ORDER = {"minor": 0, "notable": 1, "major": 2, "critical": 3}


class EventExtractor:
    """Extracts notable events from simulation data.

    All detection is rule-based — no LLM calls needed.
    """

    def extract_agent_timeline(
        self,
        agent: Agent,
        all_agents: dict[str, Agent],
        trait_system: TraitSystem,
    ) -> list[NotableEvent]:
        """Extract a chronological timeline of notable events for an agent."""
        events: list[NotableEvent] = []

        # Birth event
        birth_gen = int(agent.generation)
        parent_names = []
        if agent.parent1_id and agent.parent1_id in all_agents:
            parent_names.append(all_agents[agent.parent1_id].name)
        if agent.parent2_id and agent.parent2_id in all_agents:
            parent_names.append(all_agents[agent.parent2_id].name)

        if agent.is_outsider:
            events.append(NotableEvent(
                event_type="outsider_arrival",
                generation=agent.injection_generation or birth_gen,
                severity="major",
                headline=f"{agent.name} arrived as an outsider",
                detail=f"Origin: {agent.outsider_origin or 'unknown'}. "
                       f"Injected into the population with foreign traits.",
                agent_ids=[agent.id],
            ))
        elif parent_names:
            events.append(NotableEvent(
                event_type="birth",
                generation=birth_gen,
                severity="minor",
                headline=f"{agent.name} was born",
                detail=f"Birth order: {int(agent.birth_order)}. "
                       f"Parents: {' & '.join(parent_names)}.",
                agent_ids=[agent.id],
            ))

        # Breakthroughs from personal memories
        for mem in agent.personal_memories:
            mem_type = mem.get("type", mem.get("content", ""))
            if "breakthrough" in str(mem_type).lower():
                gen = mem.get("created_generation", birth_gen)
                events.append(NotableEvent(
                    event_type="breakthrough",
                    generation=gen,
                    severity="notable",
                    headline=f"{agent.name} achieved a breakthrough",
                    detail=f"A notable achievement during generation {gen}.",
                    agent_ids=[agent.id],
                ))

        # Region transitions — detect R2→R4/R5 descent
        for i in range(1, len(agent.region_history)):
            prev = agent.region_history[i - 1].value
            curr = agent.region_history[i].value
            gen = birth_gen + i
            if prev in ("optimal", "deep") and curr == "sacrificial":
                events.append(NotableEvent(
                    event_type="descent_to_sacrificial",
                    generation=gen,
                    severity="notable",
                    headline=f"{agent.name} descended into sacrificial processing",
                    detail=f"Transitioned from {prev} to sacrificial at age {i}. "
                           f"Suffering intensified but productivity may follow.",
                    agent_ids=[agent.id],
                ))
            elif prev in ("optimal", "deep") and curr == "pathological":
                events.append(NotableEvent(
                    event_type="descent_to_pathological",
                    generation=gen,
                    severity="major",
                    headline=f"{agent.name} fell into pathological processing",
                    detail=f"Transitioned from {prev} to pathological at age {i}. "
                           f"Unproductive suffering with no output.",
                    agent_ids=[agent.id],
                ))
            elif prev in ("sacrificial", "pathological") and curr == "optimal":
                events.append(NotableEvent(
                    event_type="recovery",
                    generation=gen,
                    severity="notable",
                    headline=f"{agent.name} recovered to optimal processing",
                    detail=f"Recovered from {prev} to optimal at age {i}.",
                    agent_ids=[agent.id],
                ))

        # Relationship events from decision history
        for dec in agent.decision_history:
            context = dec.get("context", "")
            chosen = dec.get("chosen_action", "")
            gen = dec.get("generation", birth_gen)
            if "pairing" in str(context).lower() and "form" in str(chosen).lower():
                events.append(NotableEvent(
                    event_type="pair_formed",
                    generation=gen,
                    severity="minor",
                    headline=f"{agent.name} formed a partnership",
                    detail=f"Chose to form a pair in generation {gen}.",
                    agent_ids=[agent.id],
                ))

        # Children born
        for child_id in agent.children_ids:
            if child_id in all_agents:
                child = all_agents[child_id]
                events.append(NotableEvent(
                    event_type="child_born",
                    generation=int(child.generation),
                    severity="minor",
                    headline=f"{agent.name} had a child: {child.name}",
                    detail=f"{child.name} was born (birth order {int(child.birth_order)}).",
                    agent_ids=[agent.id, child_id],
                ))

        # Death event
        death_info = agent.extension_data.get("death_info")
        if death_info:
            # Compute mean age for premature death detection
            all_death_ages = [
                a.extension_data["death_info"]["age_at_death"]
                for a in all_agents.values()
                if "death_info" in a.extension_data
            ]
            mean_death_age = float(np.mean(all_death_ages)) if all_death_ages else 40.0
            age_at_death = death_info["age_at_death"]
            is_premature = age_at_death < mean_death_age * 0.5

            primary_cause = death_info.get("primary_cause", "unknown")
            breakdown = death_info.get("mortality_breakdown", {})
            breakdown_str = ", ".join(
                f"{k}: {v:.4f}" for k, v in sorted(
                    breakdown.items(), key=lambda x: x[1], reverse=True
                )
            )

            severity = "critical" if is_premature else "notable"
            headline = (
                f"{agent.name} died prematurely at age {age_at_death}"
                if is_premature
                else f"{agent.name} died at age {age_at_death}"
            )

            events.append(NotableEvent(
                event_type="premature_death" if is_premature else "death",
                generation=death_info["generation"],
                severity=severity,
                headline=headline,
                detail=f"Primary cause: {primary_cause}. "
                       f"Region at death: {death_info.get('processing_region_at_death', '?')}. "
                       f"Suffering: {death_info.get('suffering_at_death', 0):.2f}, "
                       f"Burnout: {death_info.get('burnout_at_death', 0):.2f}. "
                       f"Breakdown: {breakdown_str}.",
                agent_ids=[agent.id],
                metrics_snapshot=death_info,
            ))
        elif not agent.is_alive:
            # Dead but no death_info (pre-Chunk 1 agents)
            events.append(NotableEvent(
                event_type="death",
                generation=birth_gen + len(agent.contribution_history),
                severity="notable",
                headline=f"{agent.name} died",
                detail="No detailed death information available.",
                agent_ids=[agent.id],
            ))

        # Sort by generation
        events.sort(key=lambda e: (e.generation, SEVERITY_ORDER.get(e.severity, 0)))
        return events

    def extract_generation_events(
        self,
        generation: int,
        metrics: GenerationMetrics | dict[str, Any],
        all_agents: dict[str, Agent],
        trait_system: TraitSystem,
        prev_metrics: GenerationMetrics | dict[str, Any] | None = None,
    ) -> list[NotableEvent]:
        """Extract notable events for a specific generation."""
        events: list[NotableEvent] = []

        # Normalize metrics access
        def _get(m: Any, key: str, default: Any = 0) -> Any:
            if isinstance(m, dict):
                return m.get(key, default)
            return getattr(m, key, default)

        pop = _get(metrics, "population_size", 0)
        births = _get(metrics, "births", 0)
        deaths = _get(metrics, "deaths", 0)
        breakthroughs = _get(metrics, "breakthroughs", 0)
        outsiders_injected = _get(metrics, "outsiders_injected", 0)

        # Population change
        if prev_metrics is not None:
            prev_pop = _get(prev_metrics, "population_size", pop)
            if prev_pop > 0:
                change_pct = (pop - prev_pop) / prev_pop
                if change_pct > 0.2:
                    events.append(NotableEvent(
                        event_type="population_boom",
                        generation=generation,
                        severity="critical",
                        headline=f"Population boom: {prev_pop} → {pop} (+{change_pct:.0%})",
                        detail=f"{births} births, {deaths} deaths this generation.",
                        metrics_snapshot={"prev_pop": prev_pop, "new_pop": pop, "change_pct": round(change_pct, 3)},
                    ))
                elif change_pct < -0.2:
                    events.append(NotableEvent(
                        event_type="population_crash",
                        generation=generation,
                        severity="critical",
                        headline=f"Population crash: {prev_pop} → {pop} ({change_pct:.0%})",
                        detail=f"{births} births, {deaths} deaths this generation.",
                        metrics_snapshot={"prev_pop": prev_pop, "new_pop": pop, "change_pct": round(change_pct, 3)},
                    ))

        # Breakthroughs — identify WHO achieved them
        if breakthroughs > 0:
            severity = "major" if breakthroughs >= 3 else "notable"
            breakthrough_agents = []
            for agent in all_agents.values():
                if not agent.personal_memories:
                    continue
                for mem in agent.personal_memories:
                    mem_gen = mem.get("created_generation")
                    mem_type = str(mem.get("type", mem.get("content", "")))
                    if mem_gen == generation and "breakthrough" in mem_type.lower():
                        region = agent.processing_region.value if hasattr(agent.processing_region, "value") else str(agent.processing_region)
                        breakthrough_agents.append({
                            "id": agent.id,
                            "name": agent.name,
                            "region": region,
                        })
                        break  # one entry per agent

            agent_ids = [a["id"] for a in breakthrough_agents]
            if breakthrough_agents:
                names = ", ".join(a["name"] for a in breakthrough_agents[:8])
                if len(breakthrough_agents) > 8:
                    names += f" and {len(breakthrough_agents) - 8} others"
                region_summary = ", ".join(
                    f"{a['name']} ({a['region']})" for a in breakthrough_agents[:5]
                )
                detail = f"Achieved by: {region_summary}."
            else:
                names = f"{breakthroughs} agents"
                detail = f"Notable achievements amid a population of {pop}."

            events.append(NotableEvent(
                event_type="breakthroughs",
                generation=generation,
                severity=severity,
                headline=f"{breakthroughs} breakthrough{'s' if breakthroughs > 1 else ''}: {names}",
                detail=detail,
                agent_ids=agent_ids,
                metrics_snapshot={"breakthroughs": breakthroughs},
            ))

        # Outsider injections — identify who arrived
        if outsiders_injected > 0:
            outsider_agents = [
                a for a in all_agents.values()
                if a.is_outsider and (a.injection_generation or a.generation) == generation
            ]
            outsider_ids = [a.id for a in outsider_agents]
            outsider_names = ", ".join(a.name for a in outsider_agents[:5])
            if not outsider_names:
                outsider_names = f"{outsiders_injected} unknown"
            detail_parts = []
            for a in outsider_agents[:5]:
                origin = a.outsider_origin or "unknown origin"
                region = a.processing_region.value if hasattr(a.processing_region, "value") else str(a.processing_region)
                detail_parts.append(f"{a.name} ({origin}, {region})")
            detail = "Arrivals: " + ", ".join(detail_parts) + "." if detail_parts else "Foreign agents injected."

            events.append(NotableEvent(
                event_type="outsider_arrival",
                generation=generation,
                severity="major",
                headline=f"Outsider{'s' if outsiders_injected > 1 else ''} arrived: {outsider_names}",
                detail=detail,
                agent_ids=outsider_ids,
                metrics_snapshot={"outsiders_injected": outsiders_injected},
            ))

        # High death rate — identify who died and why
        if pop > 0 and deaths > pop * 0.3:
            dead_this_gen = [
                a for a in all_agents.values()
                if not a.is_alive and a.extension_data.get("death_info", {}).get("generation") == generation
            ]
            cause_counts: dict[str, int] = {}
            dead_names = []
            for a in dead_this_gen:
                di = a.extension_data.get("death_info", {})
                cause = di.get("primary_cause", "unknown")
                cause_counts[cause] = cause_counts.get(cause, 0) + 1
                dead_names.append(a.name)

            cause_str = ", ".join(f"{c}: {n}" for c, n in sorted(cause_counts.items(), key=lambda x: x[1], reverse=True))
            names_str = ", ".join(dead_names[:8])
            if len(dead_names) > 8:
                names_str += f" and {len(dead_names) - 8} others"

            events.append(NotableEvent(
                event_type="mass_death",
                generation=generation,
                severity="critical",
                headline=f"Mass casualties: {deaths} deaths ({deaths/pop:.0%} of population)",
                detail=f"Causes: {cause_str}. Deceased: {names_str}.",
                agent_ids=[a.id for a in dead_this_gen[:10]],
                metrics_snapshot={"deaths": deaths, "pop": pop},
            ))

        # Premature deaths this generation — consolidated into one event
        all_death_ages = [
            a.extension_data["death_info"]["age_at_death"]
            for a in all_agents.values()
            if "death_info" in a.extension_data
        ]
        mean_death_age = float(np.mean(all_death_ages)) if all_death_ages else 40.0

        premature_deaths = []
        seen_ids: set[str] = set()
        for agent in all_agents.values():
            if agent.id in seen_ids:
                continue
            death_info = agent.extension_data.get("death_info")
            if death_info and death_info.get("generation") == generation:
                if death_info["age_at_death"] < mean_death_age * 0.5:
                    premature_deaths.append(agent)
                    seen_ids.add(agent.id)

        if premature_deaths:
            detail_lines = []
            for agent in premature_deaths[:8]:
                di = agent.extension_data["death_info"]
                cause = di.get("primary_cause", "unknown")
                region = di.get("processing_region_at_death", "?")
                detail_lines.append(f"{agent.name} (age {di['age_at_death']}, {cause}, {region})")

            names = ", ".join(a.name for a in premature_deaths[:8])
            if len(premature_deaths) > 8:
                names += f" and {len(premature_deaths) - 8} others"

            severity = "critical" if len(premature_deaths) >= 3 else "major"
            events.append(NotableEvent(
                event_type="premature_death",
                generation=generation,
                severity=severity,
                headline=f"{len(premature_deaths)} premature death{'s' if len(premature_deaths) > 1 else ''}: {names}",
                detail=f"Mean death age: {mean_death_age:.1f}. " + "; ".join(detail_lines) + ".",
                agent_ids=[a.id for a in premature_deaths[:10]],
                metrics_snapshot={"count": len(premature_deaths), "mean_death_age": round(mean_death_age, 1)},
            ))

        # Region distribution shifts — identify suffering agents
        region_fractions = _get(metrics, "region_fractions", {})
        if isinstance(region_fractions, dict):
            r4_frac = region_fractions.get("sacrificial", 0)
            r5_frac = region_fractions.get("pathological", 0)
            if r4_frac + r5_frac > 0.3:
                # Find agents currently in R4/R5
                suffering_agents = []
                for agent in all_agents.values():
                    if not agent.is_alive:
                        continue
                    region_val = agent.processing_region.value if hasattr(agent.processing_region, "value") else str(agent.processing_region)
                    if region_val in ("sacrificial", "pathological"):
                        suffering_agents.append({
                            "id": agent.id,
                            "name": agent.name,
                            "region": region_val,
                            "suffering": round(float(agent.suffering), 2),
                        })
                # Sort by suffering (worst first)
                suffering_agents.sort(key=lambda a: a["suffering"], reverse=True)

                r4_agents = [a for a in suffering_agents if a["region"] == "sacrificial"]
                r5_agents = [a for a in suffering_agents if a["region"] == "pathological"]

                detail_parts = [f"Sacrificial: {r4_frac:.0%} ({len(r4_agents)} agents), Pathological: {r5_frac:.0%} ({len(r5_agents)} agents)."]
                if r5_agents:
                    worst = ", ".join(f"{a['name']} (suf={a['suffering']})" for a in r5_agents[:5])
                    detail_parts.append(f"Worst affected (R5): {worst}.")
                if r4_agents:
                    top_r4 = ", ".join(f"{a['name']} (suf={a['suffering']})" for a in r4_agents[:5])
                    detail_parts.append(f"Sacrificial (R4): {top_r4}.")

                events.append(NotableEvent(
                    event_type="suffering_epidemic",
                    generation=generation,
                    severity="major",
                    headline=f"Suffering epidemic: {(r4_frac + r5_frac):.0%} in R4/R5",
                    detail=" ".join(detail_parts),
                    agent_ids=[a["id"] for a in suffering_agents[:10]],
                    metrics_snapshot={"r4": round(r4_frac, 3), "r5": round(r5_frac, 3)},
                ))

        # Sort by severity (most important first)
        events.sort(key=lambda e: SEVERITY_ORDER.get(e.severity, 0), reverse=True)
        return events
