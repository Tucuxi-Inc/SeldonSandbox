"""
Core agent dataclass for the Seldon Sandbox.

Agents have N-dimensional personality traits, are classified into RSH
processing regions, track full history for visualization, and carry
lore/memory and decision data for explainability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from seldon.core.processing import ProcessingRegion


@dataclass
class Agent:
    """A simulated individual with personality traits and history."""

    # === Identity ===
    id: str
    name: str
    age: int
    generation: int
    birth_order: int

    # === Personality (N-dimensional, where N = TraitSystem.count) ===
    traits: np.ndarray
    traits_at_birth: np.ndarray

    # === RSH Processing Region ===
    processing_region: ProcessingRegion = ProcessingRegion.OPTIMAL

    # === State ===
    suffering: float = 0.0
    burnout_level: float = 0.0
    is_alive: bool = True

    # === Relationships ===
    partner_id: str | None = None
    parent1_id: str | None = None
    parent2_id: str | None = None
    children_ids: list[str] = field(default_factory=list)
    relationship_status: str = "single"  # single, paired, widowed, dissolved

    # === Social ===
    social_bonds: dict[str, float] = field(default_factory=dict)  # agent_id -> strength

    # === Cognitive council ===
    dominant_voice: str | None = None  # Which sub-agent is strongest

    # === History (append each generation — drives all visualizations) ===
    trait_history: list[np.ndarray] = field(default_factory=list)
    region_history: list[ProcessingRegion] = field(default_factory=list)
    contribution_history: list[float] = field(default_factory=list)
    suffering_history: list[float] = field(default_factory=list)

    # === Lore / Memory ===
    personal_memories: list[dict[str, Any]] = field(default_factory=list)
    inherited_lore: list[dict[str, Any]] = field(default_factory=list)

    # === Decision history (for explainability) ===
    decision_history: list[dict[str, Any]] = field(default_factory=list)

    # === Social hierarchy (Phase 7) ===
    social_status: float = 0.0
    mentor_id: str | None = None
    mentee_ids: list[str] = field(default_factory=list)
    social_role: str | None = None
    influence_score: float = 0.0

    # === Genetics (Phase 8) ===
    genome: dict[str, tuple[str, str]] = field(default_factory=dict)
    epigenetic_state: dict[str, bool] = field(default_factory=dict)
    genetic_lineage: dict[str, Any] = field(default_factory=dict)

    # === Community (Phase 9) ===
    community_id: str | None = None

    # === Economics (Phase 10) ===
    wealth: float = 0.0
    occupation: str | None = None
    trade_history: list[dict[str, Any]] = field(default_factory=list)

    # === Extension hooks ===
    location_id: str | None = None
    resource_holdings: dict[str, float] = field(default_factory=dict)
    cultural_memes: list[str] = field(default_factory=list)
    skills: dict[str, float] = field(default_factory=dict)
    extension_data: dict[str, Any] = field(default_factory=dict)

    # === Outsider tracking ===
    is_outsider: bool = False
    outsider_origin: str | None = None
    injection_generation: int | None = None

    # === Demographics ===
    gender: str | None = None

    # === Fertility tracking ===
    last_birth_generation: int | None = None

    # === Tick-based time (Phase A) ===
    _age_ticks: int = 0                               # Internal age in ticks (months)
    life_phase: str | None = None                     # Current life phase value
    location: tuple[int, int] | None = None           # Hex coordinates (q, r)

    # === Needs (Phase A) ===
    needs: dict[str, float] = field(default_factory=lambda: {
        "hunger": 1.0, "thirst": 1.0, "shelter": 1.0,
        "safety": 1.0, "warmth": 1.0, "rest": 1.0,
    })
    health: float = 1.0                               # 0-1, damaged by chronic unmet needs

    # === Needs history (yearly snapshots) ===
    needs_history: list[dict[str, float]] = field(default_factory=list)
    health_history: list[float] = field(default_factory=list)

    def record_generation(self, contribution: float) -> None:
        """Append current state to all history lists."""
        self.trait_history.append(self.traits.copy())
        self.region_history.append(self.processing_region)
        self.contribution_history.append(contribution)
        self.suffering_history.append(self.suffering)
        self.needs_history.append(dict(self.needs))
        self.health_history.append(self.health)

    def is_eligible_for_pairing(self, min_age: int) -> bool:
        """Check if agent can seek a partner."""
        return (
            self.is_alive
            and self.age >= min_age
            and self.partner_id is None
            and self.relationship_status in ("single", "widowed", "dissolved")
        )

    @property
    def is_descendant_of_outsider(self) -> bool:
        """True if this agent is an outsider or has outsider ancestry."""
        return self.is_outsider or self.extension_data.get("outsider_ancestor", False)

    def __repr__(self) -> str:
        status = "alive" if self.is_alive else "dead"
        return (
            f"Agent(id={self.id!r}, name={self.name!r}, age={self.age}, "
            f"gen={self.generation}, region={self.processing_region.value}, {status})"
        )
