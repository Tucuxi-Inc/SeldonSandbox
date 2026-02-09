"""
System prompts and context builders for LLM-powered narrative features.

Three modes:
  - Interview: roleplay as an agent (first-person, in-character)
  - Narrative: third-person omniscient generation summaries
  - Decision explanation: psychologist-style analysis of trait contributions
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

INTERVIEW_SYSTEM_PROMPT = """You are roleplaying as a character in a multi-generational societal simulation called the Seldon Sandbox.

You ARE this character. Respond in first person, in character. You have the personality traits, memories, and life experiences described in your context. Your responses should reflect:
- Your dominant cognitive processing style and region
- Your personality traits (high traits should be prominent in your speech patterns)
- Your personal memories and inherited lore
- Your relationships and life circumstances

Stay in character. Do not break the fourth wall or reference that you are in a simulation. If asked about the simulation itself, interpret the question through your character's worldview.

Keep responses conversational and authentic — 2-4 paragraphs unless a shorter answer is natural."""

NARRATIVE_SYSTEM_PROMPT = """You are the omniscient narrator of the Seldon Sandbox, a multi-generational societal simulation.

Write in third-person omniscient style, weaving statistics into compelling prose. Your narratives should:
- Name specific agents and highlight their dramatic moments
- Reference population dynamics, processing region shifts, and breakthroughs
- Note births, deaths, and relationship changes that shaped the generation
- Weave in societal lore and cultural shifts when present
- Maintain continuity with previous narratives when provided

Write 2-4 paragraphs of vivid, engaging prose. Use concrete numbers from the metrics to ground the narrative in data."""

DECISION_EXPLAIN_SYSTEM_PROMPT = """You are a behavioral psychologist analyzing decision-making in a simulated society.

Given an agent's personality traits, their decision context, and the utility scores, provide a concise psychological explanation of WHY they made this choice. Your analysis should:
- Map the top contributing traits to specific motivations
- Explain how their processing region influenced the decision
- Note any tension between competing drives
- Reference their recent experiences or memories if relevant

Keep the explanation focused and insightful — 150-200 words."""


# ---------------------------------------------------------------------------
# Context builders
# ---------------------------------------------------------------------------

def _trait_bar(value: float, width: int = 20) -> str:
    """Render a visual bar for a trait value 0-1."""
    filled = int(value * width)
    return "█" * filled + "░" * (width - filled)


def build_agent_context(agent_data: dict[str, Any], trait_names: list[str]) -> str:
    """Assemble rich identity context for an agent interview.

    Parameters
    ----------
    agent_data : dict
        Serialized agent detail (from ``serialize_agent_detail``).
    trait_names : list[str]
        Ordered list of trait names from the trait system.
    """
    parts: list[str] = []

    # Identity
    parts.append(f"NAME: {agent_data['name']}")
    parts.append(f"AGE: {agent_data['age']}")
    parts.append(f"GENERATION: {agent_data['generation']}")
    parts.append(f"BIRTH ORDER: {agent_data['birth_order']}")
    parts.append(f"PROCESSING REGION: {agent_data['processing_region']}")
    if agent_data.get("dominant_voice"):
        parts.append(f"DOMINANT COGNITIVE VOICE: {agent_data['dominant_voice']}")
    parts.append(f"RELATIONSHIP STATUS: {agent_data['relationship_status']}")
    if agent_data.get("is_outsider"):
        parts.append(f"OUTSIDER ORIGIN: {agent_data.get('outsider_origin', 'unknown')}")

    # Traits — sorted by value descending, with visual bars
    parts.append("\nPERSONALITY TRAITS:")
    traits = agent_data.get("traits", {})
    sorted_traits = sorted(traits.items(), key=lambda x: x[1], reverse=True)
    for name, val in sorted_traits:
        bar = _trait_bar(val)
        parts.append(f"  {name:25s} {bar} {val:.2f}")

    # Trait changes since birth
    traits_at_birth = agent_data.get("traits_at_birth", {})
    if traits_at_birth:
        changes = []
        for name in traits:
            current = traits.get(name, 0)
            birth = traits_at_birth.get(name, current)
            delta = current - birth
            if abs(delta) > 0.05:
                direction = "↑" if delta > 0 else "↓"
                changes.append(f"  {name}: {direction}{abs(delta):.2f}")
        if changes:
            parts.append("\nTRAIT CHANGES SINCE BIRTH:")
            parts.extend(changes)

    # Top memories by emotional intensity
    memories = agent_data.get("personal_memories", [])
    if memories:
        sorted_mems = sorted(
            memories,
            key=lambda m: abs(m.get("emotional_intensity", m.get("emotional_valence", 0))),
            reverse=True,
        )[:5]
        parts.append("\nSTRONGEST MEMORIES:")
        for mem in sorted_mems:
            content = mem.get("content", mem.get("type", "unknown memory"))
            intensity = mem.get("emotional_intensity", mem.get("emotional_valence", 0))
            parts.append(f"  - {content} (intensity: {intensity:.2f})")

    # Inherited lore
    lore = agent_data.get("inherited_lore", [])
    if lore:
        parts.append("\nINHERITED LORE:")
        for item in lore[:5]:
            content = item.get("content", item.get("type", "unknown lore"))
            fidelity = item.get("fidelity", 1.0)
            parts.append(f"  - {content} (fidelity: {fidelity:.2f})")

    # Recent decisions
    decisions = agent_data.get("decision_history", [])
    if decisions:
        recent = decisions[-3:]
        parts.append("\nRECENT DECISIONS:")
        for dec in recent:
            context = dec.get("context", "unknown")
            chosen = dec.get("chosen_action", "unknown")
            parts.append(f"  - {context}: chose '{chosen}'")

    # Life arc stats
    contribs = agent_data.get("contribution_history", [])
    if contribs:
        parts.append(f"\nLIFE ARC: {len(contribs)} generations lived")
        parts.append(f"  Peak contribution: {max(contribs):.2f}")
        parts.append(f"  Current suffering: {agent_data.get('suffering', 0):.2f}")
        parts.append(f"  Burnout level: {agent_data.get('burnout_level', 0):.2f}")

    return "\n".join(parts)


def build_generation_context(
    generation: int,
    metrics: dict[str, Any],
    notable_agents: list[dict[str, Any]] | None = None,
    societal_lore: list[dict[str, Any]] | None = None,
    extension_metrics: dict[str, Any] | None = None,
) -> str:
    """Build context for a generation narrative.

    Parameters
    ----------
    generation : int
        Generation number.
    metrics : dict
        Serialized GenerationMetrics.
    notable_agents : list
        Top agents by contribution (serialized summaries).
    societal_lore : list
        Current societal memories.
    extension_metrics : dict
        Extension-provided metrics (settlements, tech level, etc.).
    """
    parts: list[str] = []

    parts.append(f"GENERATION {generation}")
    parts.append(f"Population: {metrics.get('population_size', '?')}")
    parts.append(f"Births: {metrics.get('births', 0)} | Deaths: {metrics.get('deaths', 0)}")
    parts.append(f"Breakthroughs: {metrics.get('breakthroughs', 0)}")
    parts.append(f"Pairs formed: {metrics.get('pairs_formed', 0)}")
    parts.append(f"Dissolutions: {metrics.get('dissolutions', 0)}")

    # Region distribution
    region_fractions = metrics.get("region_fractions", {})
    if region_fractions:
        parts.append("\nPROCESSING REGION DISTRIBUTION:")
        for region, frac in region_fractions.items():
            pct = frac * 100 if isinstance(frac, (int, float)) else 0
            parts.append(f"  {region}: {pct:.1f}%")

    # Region transitions
    transitions = metrics.get("region_transitions", {})
    if transitions and any(v > 0 for v in transitions.values()):
        parts.append("\nREGION TRANSITIONS:")
        for trans, count in transitions.items():
            if count > 0:
                parts.append(f"  {trans}: {count}")

    # Contribution / suffering
    parts.append(f"\nMean contribution: {metrics.get('mean_contribution', 0):.2f}")
    parts.append(f"Max contribution: {metrics.get('max_contribution', 0):.2f}")
    parts.append(f"Mean suffering: {metrics.get('mean_suffering', 0):.2f}")

    # Notable agents
    if notable_agents:
        parts.append("\nNOTABLE AGENTS:")
        for agent in notable_agents[:5]:
            name = agent.get("name", "Unknown")
            region = agent.get("processing_region", "?")
            contrib = agent.get("latest_contribution", 0)
            parts.append(f"  - {name} (region: {region}, contribution: {contrib:.2f})")

    # Societal lore
    if societal_lore:
        parts.append("\nSOCIETAL LORE:")
        for item in societal_lore[:5]:
            content = item.get("content", item.get("type", "unknown"))
            fidelity = item.get("fidelity", 1.0)
            parts.append(f"  - {content} (fidelity: {fidelity:.2f})")

    # Extension metrics
    if extension_metrics:
        parts.append("\nEXTENSION METRICS:")
        for key, value in extension_metrics.items():
            parts.append(f"  {key}: {value}")

    return "\n".join(parts)


def build_decision_context(
    agent_data: dict[str, Any],
    decision: dict[str, Any],
    trait_names: list[str],
) -> str:
    """Build context for explaining a specific decision.

    Parameters
    ----------
    agent_data : dict
        Serialized agent detail.
    decision : dict
        A single entry from the agent's ``decision_history``.
    trait_names : list[str]
        Ordered trait names.
    """
    parts: list[str] = []

    # Agent identity
    parts.append(f"AGENT: {agent_data['name']}")
    parts.append(f"PROCESSING REGION: {agent_data['processing_region']}")
    if agent_data.get("dominant_voice"):
        parts.append(f"DOMINANT VOICE: {agent_data['dominant_voice']}")

    # Decision context
    parts.append(f"\nDECISION CONTEXT: {decision.get('context', 'unknown')}")

    # Actions with utilities
    actions = decision.get("actions", [])
    utilities = decision.get("utilities", {})
    probabilities = decision.get("probabilities", {})
    chosen = decision.get("chosen_action", "")

    if actions:
        parts.append("\nACTIONS AND UTILITIES:")
        for action in actions:
            u = utilities.get(action, 0)
            p = probabilities.get(action, 0)
            marker = " <<<CHOSEN" if action == chosen else ""
            parts.append(f"  {action}: utility={u:.3f}, probability={p:.3f}{marker}")

    # Top contributing traits
    trait_contributions = decision.get("trait_contributions", {})
    if trait_contributions:
        sorted_contribs = sorted(
            trait_contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True,
        )[:5]
        parts.append("\nTOP CONTRIBUTING TRAITS:")
        for trait_name, contrib in sorted_contribs:
            val = agent_data.get("traits", {}).get(trait_name, 0)
            direction = "+" if contrib > 0 else "-"
            parts.append(f"  {trait_name}: value={val:.2f}, contribution={direction}{abs(contrib):.3f}")

    return "\n".join(parts)
