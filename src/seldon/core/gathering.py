"""
Gathering system for the Seldon Sandbox.

Agents satisfy their survival needs through gathering activities (foraging,
hunting, fishing, building shelter, etc.).  Each activity targets a specific
need and has a base yield modified by terrain suitability, seasonal
conditions, and the agent's personality traits.

Terrain/season keys are plain strings — no imports from hex_grid — so the
module stays decoupled from geography extensions.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from seldon.core.agent import Agent
    from seldon.core.traits import TraitSystem


# ---------------------------------------------------------------------------
# Activity definitions
# ---------------------------------------------------------------------------

GATHERING_ACTIVITIES: dict[str, dict[str, Any]] = {
    "forage": {
        "satisfies": "hunger",
        "base_yield": 0.15,
        "trait_boosts": {"conscientiousness": 0.1, "openness": 0.05},
    },
    "hunt": {
        "satisfies": "hunger",
        "base_yield": 0.25,
        "trait_boosts": {"resilience": 0.1, "ambition": 0.05},
    },
    "fish": {
        "satisfies": "hunger",
        "base_yield": 0.20,
        "trait_boosts": {"conscientiousness": 0.08, "resilience": 0.05},
    },
    "find_water": {
        "satisfies": "thirst",
        "base_yield": 0.25,
        "trait_boosts": {"conscientiousness": 0.1},
    },
    "build_shelter": {
        "satisfies": "shelter",
        "base_yield": 0.20,
        "trait_boosts": {"conscientiousness": 0.12, "resilience": 0.08},
    },
    "seek_warmth": {
        "satisfies": "warmth",
        "base_yield": 0.20,
        "trait_boosts": {"resilience": 0.1},
    },
    "rest": {
        "satisfies": "rest",
        "base_yield": 0.30,
        "trait_boosts": {},
    },
    "seek_safety": {
        "satisfies": "safety",
        "base_yield": 0.20,
        "trait_boosts": {"neuroticism": 0.05, "resilience": 0.08},
    },
}


# ---------------------------------------------------------------------------
# Terrain suitability per activity (multiplier on yield)
# ---------------------------------------------------------------------------

TERRAIN_ACTIVITY_SUITABILITY: dict[str, dict[str, float]] = {
    "ocean": {
        "forage": 0.0, "hunt": 0.0, "fish": 0.0, "find_water": 0.5,
        "build_shelter": 0.0, "seek_warmth": 0.0, "rest": 0.3, "seek_safety": 0.1,
    },
    "coast": {
        "forage": 0.5, "hunt": 0.3, "fish": 1.4, "find_water": 0.8,
        "build_shelter": 0.7, "seek_warmth": 0.7, "rest": 0.8, "seek_safety": 0.7,
    },
    "coastal_valley": {
        "forage": 1.2, "hunt": 0.8, "fish": 0.3, "find_water": 1.0,
        "build_shelter": 1.0, "seek_warmth": 0.9, "rest": 1.0, "seek_safety": 0.9,
    },
    "foothills": {
        "forage": 0.8, "hunt": 0.9, "fish": 0.2, "find_water": 0.6,
        "build_shelter": 0.8, "seek_warmth": 0.7, "rest": 0.8, "seek_safety": 0.7,
    },
    "mountains": {
        "forage": 0.3, "hunt": 0.6, "fish": 0.1, "find_water": 0.4,
        "build_shelter": 0.5, "seek_warmth": 0.4, "rest": 0.5, "seek_safety": 0.5,
    },
    "high_desert": {
        "forage": 0.2, "hunt": 0.4, "fish": 0.0, "find_water": 0.2,
        "build_shelter": 0.4, "seek_warmth": 0.5, "rest": 0.6, "seek_safety": 0.6,
    },
    "desert": {
        "forage": 0.1, "hunt": 0.2, "fish": 0.0, "find_water": 0.1,
        "build_shelter": 0.3, "seek_warmth": 0.3, "rest": 0.5, "seek_safety": 0.5,
    },
    "forest": {
        "forage": 1.3, "hunt": 1.2, "fish": 0.2, "find_water": 0.7,
        "build_shelter": 1.1, "seek_warmth": 1.0, "rest": 1.0, "seek_safety": 0.6,
    },
    "plains": {
        "forage": 0.9, "hunt": 1.0, "fish": 0.1, "find_water": 0.5,
        "build_shelter": 0.8, "seek_warmth": 0.6, "rest": 0.9, "seek_safety": 0.8,
    },
    "river_valley": {
        "forage": 1.1, "hunt": 0.7, "fish": 1.2, "find_water": 1.3,
        "build_shelter": 0.9, "seek_warmth": 0.8, "rest": 0.9, "seek_safety": 0.8,
    },
}


# ---------------------------------------------------------------------------
# Season suitability per activity
# ---------------------------------------------------------------------------

SEASON_ACTIVITY_SUITABILITY: dict[str, dict[str, float]] = {
    "spring": {
        "forage": 1.1, "hunt": 1.0, "fish": 1.0, "find_water": 1.1,
        "build_shelter": 1.1, "seek_warmth": 0.9, "rest": 1.0, "seek_safety": 1.0,
    },
    "summer": {
        "forage": 1.0, "hunt": 1.1, "fish": 1.1, "find_water": 0.8,
        "build_shelter": 1.2, "seek_warmth": 0.7, "rest": 0.9, "seek_safety": 1.0,
    },
    "autumn": {
        "forage": 1.3, "hunt": 1.1, "fish": 0.9, "find_water": 0.9,
        "build_shelter": 1.0, "seek_warmth": 1.0, "rest": 1.0, "seek_safety": 1.0,
    },
    "winter": {
        "forage": 0.5, "hunt": 0.7, "fish": 0.6, "find_water": 0.7,
        "build_shelter": 0.7, "seek_warmth": 1.3, "rest": 1.1, "seek_safety": 0.9,
    },
}


# ---------------------------------------------------------------------------
# GatheringSystem
# ---------------------------------------------------------------------------

class GatheringSystem:
    """
    Selects, executes, and applies gathering activities so agents can
    replenish their survival needs.
    """

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Activity selection
    # ------------------------------------------------------------------

    def choose_activity(
        self,
        agent: Agent,
        prioritized_needs: list[str],
        terrain: str | None,
        season: str | None,
        trait_system: TraitSystem,
    ) -> str:
        """
        Pick the best gathering activity for *agent* given prioritised needs.

        Iterates needs in priority order, scores candidate activities by
        terrain suitability, season suitability, and trait bonuses, then
        returns the highest-scoring activity for the most urgent need that
        has at least one viable candidate.
        """
        terrain_suits = TERRAIN_ACTIVITY_SUITABILITY.get(terrain, {}) if terrain else {}
        season_suits = SEASON_ACTIVITY_SUITABILITY.get(season, {}) if season else {}

        for need_name in prioritized_needs:
            # Find activities that satisfy this need
            candidates: list[tuple[str, float]] = []
            for act_name, act_def in GATHERING_ACTIVITIES.items():
                if act_def["satisfies"] != need_name:
                    continue

                t_suit = terrain_suits.get(act_name, 1.0)
                s_suit = season_suits.get(act_name, 1.0)
                trait_bonus = self._compute_trait_bonus(
                    agent, act_def["trait_boosts"], trait_system,
                )
                score = t_suit * s_suit + trait_bonus
                candidates.append((act_name, score))

            if candidates:
                # Pick highest-scoring candidate
                candidates.sort(key=lambda c: c[1], reverse=True)
                return candidates[0][0]

        # Fallback (should not happen since "rest" covers the rest need)
        return "rest"

    # ------------------------------------------------------------------
    # Gathering attempt
    # ------------------------------------------------------------------

    def attempt_gathering(
        self,
        agent: Agent,
        activity: str,
        terrain: str | None,
        season: str | None,
        trait_system: TraitSystem,
        rng: np.random.Generator,
    ) -> float:
        """
        Execute a gathering activity and return the yield amount.

        yield = base_yield * terrain_suit * season_suit + trait_bonus + noise
        Clamped to [0.0, 0.95].
        """
        act_def = GATHERING_ACTIVITIES.get(activity)
        if act_def is None:
            return 0.0

        base_yield = act_def["base_yield"]

        terrain_suit = (
            TERRAIN_ACTIVITY_SUITABILITY.get(terrain, {}).get(activity, 1.0)
            if terrain is not None
            else 1.0
        )
        season_suit = (
            SEASON_ACTIVITY_SUITABILITY.get(season, {}).get(activity, 1.0)
            if season is not None
            else 1.0
        )

        trait_bonus = self._compute_trait_bonus(
            agent, act_def["trait_boosts"], trait_system,
        )

        success = base_yield * terrain_suit * season_suit + trait_bonus
        success += rng.normal(0, 0.02)
        return float(min(0.95, max(0.0, success)))

    # ------------------------------------------------------------------
    # Need satisfaction
    # ------------------------------------------------------------------

    @staticmethod
    def satisfy_need(agent: Agent, need_name: str, amount: float) -> None:
        """Add *amount* to agent's need, clamped to [0, 1]."""
        current = agent.needs.get(need_name, 0.0)
        agent.needs[need_name] = min(1.0, current + amount)

    # ------------------------------------------------------------------
    # Caregiver sharing
    # ------------------------------------------------------------------

    @staticmethod
    def caregiver_share(
        caregiver: Agent,
        dependents: list[Agent],
        rng: np.random.Generator,
    ) -> None:
        """
        The caregiver shares a portion of resources with infant/child
        dependents whose needs are lower than the caregiver's own.

        The caregiver will not reduce their own needs below 0.3.
        """
        share_rate = 0.3
        for dep in dependents:
            if dep.life_phase not in ("infant", "child"):
                continue
            for need_name in dep.needs:
                dep_level = dep.needs[need_name]
                care_level = caregiver.needs.get(need_name, 0.0)
                if dep_level >= care_level:
                    continue
                deficit = care_level - dep_level
                transfer = min(
                    share_rate * max(0.0, care_level - 0.3),
                    deficit,
                )
                transfer = max(0.0, transfer)
                dep.needs[need_name] += transfer
                caregiver.needs[need_name] -= transfer

    # ------------------------------------------------------------------
    # Eligibility
    # ------------------------------------------------------------------

    @staticmethod
    def can_gather(agent: Agent) -> bool:
        """Return whether the agent is old enough to gather on their own."""
        if agent.life_phase is not None:
            return agent.life_phase not in ("infant", "child")
        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_trait_bonus(
        agent: Agent,
        trait_boosts: dict[str, float],
        trait_system: TraitSystem,
    ) -> float:
        """Sum trait-weighted bonuses, silently skipping unknown traits."""
        bonus = 0.0
        for trait_name, boost in trait_boosts.items():
            try:
                idx = trait_system.trait_index(trait_name)
                bonus += float(agent.traits[idx]) * boost
            except (KeyError, IndexError):
                continue
        return bonus
