"""
Archetype definitions — 11 seed personality vectors.

Each archetype defines a trait-name->value dict that works with both compact (15)
and full (50) trait presets. Traits not specified default to 0.5 (neutral).

Archetypes are used to create agents with specific personality profiles for
experiments like "What emerges from a society of Einsteins?"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from seldon.core.agent import Agent
from seldon.core.config import ExperimentConfig
from seldon.core.traits import TraitSystem


@dataclass
class ArchetypeDefinition:
    """Seed personality vector for creating agents."""
    name: str
    description: str
    key_traits: list[str]  # Traits that define this archetype
    use_case: str  # When to use this archetype in experiments
    trait_values: dict[str, float]  # trait_name -> value [0, 1]


# ---------------------------------------------------------------------------
# 11 Archetype definitions
# ---------------------------------------------------------------------------
ARCHETYPES: dict[str, ArchetypeDefinition] = {
    "da_vinci": ArchetypeDefinition(
        name="Da Vinci",
        description="Polymath — boundless curiosity, creativity across domains",
        key_traits=["openness", "creativity", "adaptability"],
        use_case="Testing creative/innovative societies",
        trait_values={
            "openness": 0.95, "creativity": 0.95, "conscientiousness": 0.7,
            "extraversion": 0.6, "agreeableness": 0.5, "neuroticism": 0.3,
            "resilience": 0.7, "ambition": 0.85, "empathy": 0.6,
            "dominance": 0.5, "trust": 0.6, "risk_taking": 0.8,
            "adaptability": 0.9, "self_control": 0.6, "depth_drive": 0.85,
        },
    ),
    "einstein": ArchetypeDefinition(
        name="Einstein",
        description="Deep thinker — intense focus, theoretical breakthroughs",
        key_traits=["depth_drive", "openness", "creativity"],
        use_case="Testing deep-processing dominated societies",
        trait_values={
            "openness": 0.9, "creativity": 0.9, "conscientiousness": 0.6,
            "extraversion": 0.3, "agreeableness": 0.6, "neuroticism": 0.4,
            "resilience": 0.7, "ambition": 0.7, "empathy": 0.6,
            "dominance": 0.3, "trust": 0.7, "risk_taking": 0.6,
            "adaptability": 0.5, "self_control": 0.5, "depth_drive": 0.95,
        },
    ),
    "montessori": ArchetypeDefinition(
        name="Montessori",
        description="Educator — empathetic, structured, patient nurturing",
        key_traits=["empathy", "conscientiousness", "agreeableness"],
        use_case="Testing nurturing/educational societies",
        trait_values={
            "openness": 0.8, "creativity": 0.7, "conscientiousness": 0.85,
            "extraversion": 0.6, "agreeableness": 0.9, "neuroticism": 0.2,
            "resilience": 0.8, "ambition": 0.6, "empathy": 0.95,
            "dominance": 0.3, "trust": 0.85, "risk_taking": 0.3,
            "adaptability": 0.8, "self_control": 0.85, "depth_drive": 0.5,
        },
    ),
    "socrates": ArchetypeDefinition(
        name="Socrates",
        description="Questioner — relentless inquiry, challenging assumptions",
        key_traits=["openness", "depth_drive", "dominance"],
        use_case="Testing societies with strong critical thinking",
        trait_values={
            "openness": 0.95, "creativity": 0.7, "conscientiousness": 0.5,
            "extraversion": 0.7, "agreeableness": 0.3, "neuroticism": 0.3,
            "resilience": 0.85, "ambition": 0.5, "empathy": 0.6,
            "dominance": 0.7, "trust": 0.4, "risk_taking": 0.75,
            "adaptability": 0.6, "self_control": 0.6, "depth_drive": 0.9,
        },
    ),
    "curie": ArchetypeDefinition(
        name="Curie",
        description="Sacrificial genius — relentless pursuit at personal cost",
        key_traits=["depth_drive", "resilience", "conscientiousness"],
        use_case="Testing R4 (sacrificial) processing outcomes",
        trait_values={
            "openness": 0.8, "creativity": 0.8, "conscientiousness": 0.9,
            "extraversion": 0.3, "agreeableness": 0.5, "neuroticism": 0.5,
            "resilience": 0.85, "ambition": 0.8, "empathy": 0.5,
            "dominance": 0.4, "trust": 0.5, "risk_taking": 0.7,
            "adaptability": 0.4, "self_control": 0.7, "depth_drive": 0.95,
        },
    ),
    "fred_rogers": ArchetypeDefinition(
        name="Fred Rogers",
        description="Compassionate connector — unconditional positive regard",
        key_traits=["empathy", "agreeableness", "trust"],
        use_case="Testing highly cooperative/trusting societies",
        trait_values={
            "openness": 0.7, "creativity": 0.6, "conscientiousness": 0.8,
            "extraversion": 0.7, "agreeableness": 0.95, "neuroticism": 0.1,
            "resilience": 0.85, "ambition": 0.3, "empathy": 0.98,
            "dominance": 0.1, "trust": 0.95, "risk_taking": 0.2,
            "adaptability": 0.7, "self_control": 0.9, "depth_drive": 0.4,
        },
    ),
    "john_dewey": ArchetypeDefinition(
        name="John Dewey",
        description="Pragmatic reformer — learning by doing, democratic values",
        key_traits=["adaptability", "openness", "conscientiousness"],
        use_case="Testing adaptive/pragmatic societies",
        trait_values={
            "openness": 0.85, "creativity": 0.7, "conscientiousness": 0.8,
            "extraversion": 0.6, "agreeableness": 0.7, "neuroticism": 0.2,
            "resilience": 0.7, "ambition": 0.6, "empathy": 0.7,
            "dominance": 0.4, "trust": 0.7, "risk_taking": 0.5,
            "adaptability": 0.9, "self_control": 0.7, "depth_drive": 0.6,
        },
    ),
    "dumbledore": ArchetypeDefinition(
        name="Dumbledore",
        description="Wise leader — strategic patience, moral complexity",
        key_traits=["self_control", "empathy", "depth_drive"],
        use_case="Testing wisdom-driven leadership societies",
        trait_values={
            "openness": 0.8, "creativity": 0.7, "conscientiousness": 0.7,
            "extraversion": 0.5, "agreeableness": 0.6, "neuroticism": 0.2,
            "resilience": 0.9, "ambition": 0.5, "empathy": 0.85,
            "dominance": 0.6, "trust": 0.6, "risk_taking": 0.5,
            "adaptability": 0.7, "self_control": 0.9, "depth_drive": 0.8,
        },
    ),
    "yoda": ArchetypeDefinition(
        name="Yoda",
        description="Ancient sage — extreme patience, deep insight, low ego",
        key_traits=["self_control", "depth_drive", "resilience"],
        use_case="Testing contemplative/meditative societies",
        trait_values={
            "openness": 0.7, "creativity": 0.6, "conscientiousness": 0.6,
            "extraversion": 0.2, "agreeableness": 0.7, "neuroticism": 0.1,
            "resilience": 0.95, "ambition": 0.2, "empathy": 0.8,
            "dominance": 0.2, "trust": 0.7, "risk_taking": 0.3,
            "adaptability": 0.8, "self_control": 0.95, "depth_drive": 0.9,
        },
    ),
    "ada_lovelace": ArchetypeDefinition(
        name="Ada Lovelace",
        description="Visionary engineer — mathematical creativity, future-seeing",
        key_traits=["creativity", "conscientiousness", "depth_drive"],
        use_case="Testing engineering/systematic innovation societies",
        trait_values={
            "openness": 0.85, "creativity": 0.9, "conscientiousness": 0.8,
            "extraversion": 0.4, "agreeableness": 0.5, "neuroticism": 0.4,
            "resilience": 0.7, "ambition": 0.8, "empathy": 0.5,
            "dominance": 0.4, "trust": 0.5, "risk_taking": 0.6,
            "adaptability": 0.6, "self_control": 0.7, "depth_drive": 0.85,
        },
    ),
    "carl_sagan": ArchetypeDefinition(
        name="Carl Sagan",
        description="Science communicator — wonder, accessibility, cosmic perspective",
        key_traits=["openness", "extraversion", "creativity"],
        use_case="Testing knowledge-sharing/communicative societies",
        trait_values={
            "openness": 0.95, "creativity": 0.8, "conscientiousness": 0.7,
            "extraversion": 0.8, "agreeableness": 0.8, "neuroticism": 0.2,
            "resilience": 0.7, "ambition": 0.6, "empathy": 0.75,
            "dominance": 0.4, "trust": 0.75, "risk_taking": 0.5,
            "adaptability": 0.7, "self_control": 0.7, "depth_drive": 0.7,
        },
    ),
}


def get_archetype(name: str) -> ArchetypeDefinition:
    """Get an archetype definition by name (case-insensitive, underscore-separated)."""
    key = name.lower().replace(" ", "_")
    if key not in ARCHETYPES:
        raise KeyError(
            f"Unknown archetype: '{name}'. Available: {list(ARCHETYPES.keys())}"
        )
    return ARCHETYPES[key]


def list_archetypes() -> list[str]:
    """Return list of available archetype names."""
    return list(ARCHETYPES.keys())


def create_agent_from_archetype(
    archetype_name: str,
    config: ExperimentConfig,
    agent_id: str,
    agent_name: str | None = None,
    generation: int = 0,
    age: int | None = None,
    noise_sigma: float = 0.0,
    rng: np.random.Generator | None = None,
) -> Agent:
    """
    Create an agent from an archetype definition.

    Args:
        archetype_name: Name of the archetype (e.g., "einstein", "curie")
        config: Experiment configuration
        agent_id: Unique ID for the agent
        agent_name: Display name (defaults to archetype name)
        generation: Generation number
        age: Agent age (defaults to outsider_injection_age)
        noise_sigma: Gaussian noise to add to trait values (0 = exact archetype)
        rng: Random number generator for noise

    Returns:
        Agent with traits from the archetype definition
    """
    archetype = get_archetype(archetype_name)
    ts = config.trait_system
    rng = rng or np.random.default_rng()

    # Build trait vector: default 0.5 for unspecified traits
    traits = np.full(ts.count, 0.5)
    for trait_name, value in archetype.trait_values.items():
        try:
            idx = ts.trait_index(trait_name)
            traits[idx] = value
        except KeyError:
            pass  # Trait not in current preset

    # Add noise if requested
    if noise_sigma > 0:
        noise = rng.normal(0, noise_sigma, size=ts.count)
        traits = np.clip(traits + noise, 0.0, 1.0)

    agent = Agent(
        id=agent_id,
        name=agent_name or archetype.name,
        age=age if age is not None else config.outsider_injection_age,
        generation=generation,
        birth_order=1,
        traits=traits,
        traits_at_birth=traits.copy(),
    )

    return agent
