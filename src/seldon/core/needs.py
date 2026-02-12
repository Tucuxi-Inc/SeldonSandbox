"""
Needs system for the Seldon Sandbox.

Agents have six core survival needs (hunger, thirst, shelter, safety, warmth,
rest) that decay each tick (month). Decay rates are modified by terrain, season,
life phase, and burnout level. Chronically unmet needs cause suffering, health
damage, and eventually death.

Terrain/season keys are plain strings — no imports from hex_grid — so the
module stays decoupled from geography extensions.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from seldon.core.agent import Agent
    from seldon.core.traits import TraitSystem


# ---------------------------------------------------------------------------
# NeedType enum
# ---------------------------------------------------------------------------

class NeedType(str, Enum):
    """Core survival needs tracked per agent."""

    HUNGER = "hunger"
    THIRST = "thirst"
    SHELTER = "shelter"
    SAFETY = "safety"
    WARMTH = "warmth"
    REST = "rest"


# ---------------------------------------------------------------------------
# Default modifier dict (all 1.0)
# ---------------------------------------------------------------------------

DEFAULT_MODIFIERS: dict[str, float] = {need.value: 1.0 for need in NeedType}


# ---------------------------------------------------------------------------
# Terrain decay modifiers
# ---------------------------------------------------------------------------

TERRAIN_DECAY_MODIFIERS: dict[str, dict[str, float]] = {
    "ocean": {
        "hunger": 1.0, "thirst": 1.0, "shelter": 1.5,
        "safety": 1.5, "warmth": 1.3, "rest": 1.0,
    },
    "coast": {
        "hunger": 0.8, "thirst": 0.7, "shelter": 1.0,
        "safety": 0.9, "warmth": 0.9, "rest": 1.0,
    },
    "coastal_valley": {
        "hunger": 0.7, "thirst": 0.7, "shelter": 0.8,
        "safety": 0.8, "warmth": 0.9, "rest": 0.9,
    },
    "foothills": {
        "hunger": 0.9, "thirst": 0.9, "shelter": 1.0,
        "safety": 1.0, "warmth": 1.1, "rest": 1.0,
    },
    "mountains": {
        "hunger": 1.2, "thirst": 1.0, "shelter": 1.3,
        "safety": 1.2, "warmth": 1.4, "rest": 1.2,
    },
    "high_desert": {
        "hunger": 1.3, "thirst": 1.5, "shelter": 1.2,
        "safety": 1.0, "warmth": 1.0, "rest": 1.1,
    },
    "desert": {
        "hunger": 1.4, "thirst": 1.8, "shelter": 1.3,
        "safety": 1.0, "warmth": 0.8, "rest": 1.1,
    },
    "forest": {
        "hunger": 0.8, "thirst": 0.8, "shelter": 0.7,
        "safety": 1.1, "warmth": 0.8, "rest": 0.9,
    },
    "plains": {
        "hunger": 0.9, "thirst": 1.0, "shelter": 1.1,
        "safety": 0.9, "warmth": 1.0, "rest": 0.9,
    },
    "river_valley": {
        "hunger": 0.7, "thirst": 0.5, "shelter": 0.8,
        "safety": 0.9, "warmth": 0.9, "rest": 0.9,
    },
}


# ---------------------------------------------------------------------------
# Season decay modifiers
# ---------------------------------------------------------------------------

SEASON_DECAY_MODIFIERS: dict[str, dict[str, float]] = {
    "spring": {
        "hunger": 0.9, "thirst": 0.9, "shelter": 0.9,
        "safety": 1.0, "warmth": 0.9, "rest": 1.0,
    },
    "summer": {
        "hunger": 0.8, "thirst": 1.2, "shelter": 0.8,
        "safety": 1.0, "warmth": 0.7, "rest": 1.1,
    },
    "autumn": {
        "hunger": 0.85, "thirst": 0.9, "shelter": 1.0,
        "safety": 1.0, "warmth": 1.1, "rest": 1.0,
    },
    "winter": {
        "hunger": 1.3, "thirst": 0.9, "shelter": 1.2,
        "safety": 1.1, "warmth": 1.5, "rest": 1.1,
    },
}


# ---------------------------------------------------------------------------
# Life-phase decay modifiers
# ---------------------------------------------------------------------------

LIFE_PHASE_DECAY_MODIFIERS: dict[str, dict[str, float]] = {
    "infant": {
        "hunger": 1.3, "thirst": 1.3, "shelter": 1.4,
        "safety": 1.5, "warmth": 1.4, "rest": 0.8,
    },
    "child": {
        "hunger": 1.1, "thirst": 1.1, "shelter": 1.1,
        "safety": 1.2, "warmth": 1.1, "rest": 0.9,
    },
    "adolescent": {
        "hunger": 1.1, "thirst": 1.0, "shelter": 1.0,
        "safety": 1.0, "warmth": 1.0, "rest": 1.0,
    },
    "young_adult": {
        "hunger": 1.0, "thirst": 1.0, "shelter": 1.0,
        "safety": 1.0, "warmth": 1.0, "rest": 1.0,
    },
    "mature": {
        "hunger": 1.0, "thirst": 1.0, "shelter": 1.0,
        "safety": 1.0, "warmth": 1.0, "rest": 1.05,
    },
    "elder": {
        "hunger": 1.0, "thirst": 1.1, "shelter": 1.1,
        "safety": 1.1, "warmth": 1.2, "rest": 1.2,
    },
}


# ---------------------------------------------------------------------------
# NeedsSystem
# ---------------------------------------------------------------------------

class NeedsSystem:
    """
    Manages agent survival needs: decay, prioritisation, health impact,
    and mortality contribution.

    All thresholds and rates come from a config dict, never hardcoded.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config

        # Thresholds (need level below which penalties apply)
        self.warning_threshold: float = config.get("warning_threshold", 0.4)
        self.critical_threshold: float = config.get("critical_threshold", 0.2)
        self.lethal_threshold: float = config.get("lethal_threshold", 0.05)

        # Base decay rates per tick, keyed by need name
        self.base_decay_rates: dict[str, float] = config.get(
            "base_decay_rates",
            {need.value: 0.08 for need in NeedType},
        )

        # Suffering deltas per tick at each severity
        self.warning_suffering_per_tick: float = config.get(
            "warning_suffering_per_tick", 0.01,
        )
        self.critical_suffering_per_tick: float = config.get(
            "critical_suffering_per_tick", 0.03,
        )
        self.lethal_suffering_per_tick: float = config.get(
            "lethal_suffering_per_tick", 0.06,
        )

        # Health damage per tick at each severity
        self.critical_health_damage_per_tick: float = config.get(
            "critical_health_damage_per_tick", 0.03,
        )
        self.lethal_health_damage_per_tick: float = config.get(
            "lethal_health_damage_per_tick", 0.08,
        )

        # Recovery when all needs are met
        self.health_recovery_rate: float = config.get(
            "health_recovery_rate", 0.02,
        )

        # Mortality parameters
        self.health_mortality_threshold: float = config.get(
            "health_mortality_threshold", 0.5,
        )
        self.needs_mortality_multiplier: float = config.get(
            "needs_mortality_multiplier", 0.3,
        )

        # Burnout amplification
        self.burnout_decay_amplifier: float = config.get(
            "burnout_decay_amplifier", 0.2,
        )

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def initialize_needs(self, agent: Agent) -> None:
        """Set all needs to 1.0 and health to 1.0."""
        agent.needs = {need.value: 1.0 for need in NeedType}
        agent.health = 1.0

    # ------------------------------------------------------------------
    # Decay
    # ------------------------------------------------------------------

    def decay_needs(
        self,
        agent: Agent,
        terrain: str | None,
        season: str | None,
        life_phase: str | None,
    ) -> None:
        """
        Apply one tick of need decay to *agent*.

        Decay per need = base_rate * terrain_mod * season_mod * phase_mod
                         * (1 + burnout_level * burnout_amplifier)
        """
        terrain_mods = TERRAIN_DECAY_MODIFIERS.get(
            terrain, DEFAULT_MODIFIERS,
        ) if terrain is not None else DEFAULT_MODIFIERS

        season_mods = SEASON_DECAY_MODIFIERS.get(
            season, DEFAULT_MODIFIERS,
        ) if season is not None else DEFAULT_MODIFIERS

        phase_mods = LIFE_PHASE_DECAY_MODIFIERS.get(
            life_phase, DEFAULT_MODIFIERS,
        ) if life_phase is not None else DEFAULT_MODIFIERS

        burnout_factor = 1.0 + agent.burnout_level * self.burnout_decay_amplifier

        for need in NeedType:
            name = need.value
            base_rate = self.base_decay_rates.get(name, 0.08)
            decay = (
                base_rate
                * terrain_mods.get(name, 1.0)
                * season_mods.get(name, 1.0)
                * phase_mods.get(name, 1.0)
                * burnout_factor
            )
            agent.needs[name] = max(0.0, agent.needs[name] - decay)

    # ------------------------------------------------------------------
    # Prioritisation
    # ------------------------------------------------------------------

    def prioritize_needs(
        self,
        agent: Agent,
        trait_system: TraitSystem,
    ) -> list[str]:
        """
        Return need names sorted by urgency (most urgent first).

        Urgency = 1.0 - current_level.  Personality traits modulate:
        * High neuroticism amplifies perceived urgency.
        * High conscientiousness adds a pre-emptive bonus for needs
          approaching the warning zone (below 0.5).
        """
        urgencies: dict[str, float] = {}
        for need in NeedType:
            name = need.value
            level = agent.needs.get(name, 1.0)
            urgencies[name] = 1.0 - level

        # Neuroticism amplification
        try:
            neur_idx = trait_system.trait_index("neuroticism")
            neur_val = float(agent.traits[neur_idx])
            amplifier = 1.0 + 0.3 * neur_val
            urgencies = {k: v * amplifier for k, v in urgencies.items()}
        except (KeyError, IndexError):
            pass

        # Conscientiousness pre-prioritisation
        try:
            cons_idx = trait_system.trait_index("conscientiousness")
            cons_val = float(agent.traits[cons_idx])
            if cons_val > 0.5:
                for name in urgencies:
                    if agent.needs.get(name, 1.0) < 0.5:
                        urgencies[name] += 0.1
        except (KeyError, IndexError):
            pass

        # Sort descending by urgency
        sorted_needs = sorted(urgencies, key=lambda n: urgencies[n], reverse=True)
        return sorted_needs

    # ------------------------------------------------------------------
    # Health impact assessment
    # ------------------------------------------------------------------

    def assess_health_impact(self, agent: Agent) -> tuple[float, float]:
        """
        Evaluate the health and suffering consequences of the agent's
        current need levels.

        Returns
        -------
        (health_delta, suffering_delta)
            health_delta is negative when needs are critically low and
            positive when all needs are safely above the warning threshold.
        """
        health_delta = 0.0
        suffering_delta = 0.0

        for need in NeedType:
            level = agent.needs.get(need.value, 1.0)

            if level < self.lethal_threshold:
                suffering_delta += self.lethal_suffering_per_tick
                health_delta -= self.lethal_health_damage_per_tick
            elif level < self.critical_threshold:
                suffering_delta += self.critical_suffering_per_tick
                health_delta -= self.critical_health_damage_per_tick
            elif level < self.warning_threshold:
                suffering_delta += self.warning_suffering_per_tick

        # Recovery: all needs safely above warning
        if all(
            agent.needs.get(n.value, 1.0) >= self.warning_threshold
            for n in NeedType
        ):
            health_delta += self.health_recovery_rate

        return (health_delta, suffering_delta)

    # ------------------------------------------------------------------
    # Apply impact
    # ------------------------------------------------------------------

    def apply_health_impact(
        self,
        agent: Agent,
        health_delta: float,
        suffering_delta: float,
    ) -> None:
        """Apply computed health and suffering changes to *agent*."""
        agent.health = max(0.0, min(1.0, agent.health + health_delta))
        agent.suffering += suffering_delta  # suffering accumulates, no clip

    # ------------------------------------------------------------------
    # Mortality contribution
    # ------------------------------------------------------------------

    def compute_needs_mortality_factor(self, agent: Agent) -> float:
        """
        Additional mortality risk due to poor health.

        Returns 0.0 when health is above ``health_mortality_threshold``,
        otherwise returns ``needs_mortality_multiplier * (1 - health)``.
        """
        if agent.health <= self.health_mortality_threshold:
            return self.needs_mortality_multiplier * (1.0 - agent.health)
        return 0.0
