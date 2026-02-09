"""
Social Dynamics Extension — wires hierarchy + mentorship into the engine.

Hooks into the simulation lifecycle to update social status, assign roles,
manage mentorships, and modify attraction/decision based on social structure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from seldon.extensions.base import SimulationExtension
from seldon.social.hierarchy import SocialHierarchyManager
from seldon.social.mentorship import MentorshipManager

if TYPE_CHECKING:
    from seldon.core.agent import Agent
    from seldon.core.config import ExperimentConfig


# Role compatibility pairs: these roles attract each other
_COMPATIBLE_ROLES = {
    ("leader", "mediator"),
    ("mediator", "leader"),
    ("innovator", "worker"),
    ("worker", "innovator"),
    ("leader", "innovator"),
    ("innovator", "leader"),
}


class SocialDynamicsExtension(SimulationExtension):
    """Social hierarchy, mentorship, and role-based dynamics."""

    def __init__(self) -> None:
        self._hierarchy: SocialHierarchyManager | None = None
        self._mentorship: MentorshipManager | None = None
        self._hierarchy_metrics: dict[str, Any] = {}
        self._mentorship_metrics: dict[str, Any] = {}
        self._rng: np.random.Generator | None = None

    @property
    def name(self) -> str:
        return "social_dynamics"

    @property
    def description(self) -> str:
        return "Social hierarchy, mentorship, and role-based dynamics"

    def get_default_config(self) -> dict[str, Any]:
        return {
            "role_attraction_bonus": 0.15,
            "leader_stay_bonus": 0.2,
            "bridge_migrate_bonus": 0.15,
        }

    def _get_config(self, config: ExperimentConfig) -> dict[str, Any]:
        defaults = self.get_default_config()
        overrides = config.extensions.get("social_dynamics", {})
        defaults.update(overrides)
        return defaults

    def on_simulation_start(
        self, population: list[Agent], config: ExperimentConfig,
    ) -> None:
        """Initialize managers and compute initial hierarchy."""
        self._hierarchy = SocialHierarchyManager(config)
        self._mentorship = MentorshipManager(config)
        self._rng = np.random.default_rng(config.random_seed)

        # Initial status computation
        self._hierarchy_metrics = self._hierarchy.update_all(
            population, self._rng,
        )

    def on_generation_start(
        self, generation: int, population: list[Agent],
        config: ExperimentConfig,
    ) -> None:
        """Update hierarchy at the start of each generation."""
        if self._hierarchy is None:
            self._hierarchy = SocialHierarchyManager(config)
            self._rng = np.random.default_rng(config.random_seed)

        self._hierarchy_metrics = self._hierarchy.update_all(
            population, self._rng,
        )

    def on_generation_end(
        self, generation: int, population: list[Agent],
        config: ExperimentConfig,
    ) -> None:
        """Manage mentorships at end of generation."""
        if self._mentorship is None:
            self._mentorship = MentorshipManager(config)
        if self._rng is None:
            self._rng = np.random.default_rng(config.random_seed)

        # Dissolve stale mentorships
        dissolved = self._mentorship.dissolve_mentorships(
            population, self._rng,
        )

        # Match new mentors
        new_matches = self._mentorship.match_mentors(population, self._rng)

        # Apply mentorship effects
        active_count = self._mentorship.apply_mentorship_effects(
            population, self._rng,
        )

        self._mentorship_metrics = {
            "active_mentorships": active_count,
            "new_matches": len(new_matches),
            "dissolved": len(dissolved),
        }

    def on_agent_created(
        self, agent: Agent, parents: tuple[Agent, Agent],
        config: ExperimentConfig,
    ) -> None:
        """Child inherits weak social bonds from parents' networks."""
        for parent in parents:
            for bonded_id, strength in parent.social_bonds.items():
                if bonded_id != agent.id and strength > 0.3:
                    # Inherit a weak version of parent's strong bonds
                    current = agent.social_bonds.get(bonded_id, 0.0)
                    agent.social_bonds[bonded_id] = max(current, strength * 0.1)

    def modify_attraction(
        self, agent1: Agent, agent2: Agent, base_score: float,
        config: ExperimentConfig,
    ) -> float:
        """Boost attraction between agents with compatible roles."""
        sd = self._get_config(config)
        role1 = agent1.social_role or "unassigned"
        role2 = agent2.social_role or "unassigned"

        if (role1, role2) in _COMPATIBLE_ROLES:
            return base_score + sd["role_attraction_bonus"]
        return base_score

    def modify_decision(
        self, agent: Agent, context: str, utilities: dict[str, float],
        config: ExperimentConfig,
    ) -> dict[str, float]:
        """
        Modify decision utilities based on social role.

        - Leaders get bonus for staying (settlement loyalty)
        - Outsider bridges get bonus for migration (connectivity)
        """
        sd = self._get_config(config)
        role = agent.social_role or "unassigned"

        if role == "leader" and "stay" in utilities:
            utilities["stay"] = utilities.get("stay", 0.0) + sd["leader_stay_bonus"]
        elif role == "outsider_bridge" and "migrate" in utilities:
            utilities["migrate"] = utilities.get("migrate", 0.0) + sd["bridge_migrate_bonus"]

        return utilities

    def get_metrics(self, population: list[Agent]) -> dict[str, Any]:
        return {
            **self._hierarchy_metrics,
            **self._mentorship_metrics,
        }
