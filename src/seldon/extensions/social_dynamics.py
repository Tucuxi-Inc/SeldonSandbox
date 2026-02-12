"""
Social Dynamics Extension — wires hierarchy + mentorship + marriage + clans
+ institutions into the engine.

Hooks into the simulation lifecycle to update social status, assign roles,
manage mentorships, formalize marriages, track clan membership, and run
institutional governance. Modifies attraction/decision based on social structure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from seldon.extensions.base import SimulationExtension
from seldon.social.clans import ClanManager
from seldon.social.hierarchy import SocialHierarchyManager
from seldon.social.institutions import InstitutionManager
from seldon.social.marriage import MarriageManager
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
    """Social hierarchy, mentorship, marriage, clans, and institutions."""

    def __init__(self) -> None:
        self._hierarchy: SocialHierarchyManager | None = None
        self._mentorship: MentorshipManager | None = None
        self._marriage: MarriageManager | None = None
        self._clans: ClanManager | None = None
        self._institutions: InstitutionManager | None = None
        self._hierarchy_metrics: dict[str, Any] = {}
        self._mentorship_metrics: dict[str, Any] = {}
        self._marriage_metrics: dict[str, Any] = {}
        self._clan_metrics: dict[str, Any] = {}
        self._institution_metrics: dict[str, Any] = {}
        self._rng: np.random.Generator | None = None

    @property
    def name(self) -> str:
        return "social_dynamics"

    @property
    def description(self) -> str:
        return "Social hierarchy, mentorship, marriage, clans, and institutions"

    def get_default_config(self) -> dict[str, Any]:
        return {
            "role_attraction_bonus": 0.15,
            "leader_stay_bonus": 0.2,
            "bridge_migrate_bonus": 0.15,
            "clan_alliance_attraction_bonus": 0.1,
            "same_institution_attraction_bonus": 0.05,
            "clan_stay_bonus": 0.1,
            "council_stay_bonus": 0.1,
        }

    def _get_config(self, config: ExperimentConfig) -> dict[str, Any]:
        defaults = self.get_default_config()
        overrides = config.extensions.get("social_dynamics", {})
        defaults.update(overrides)
        return defaults

    def on_simulation_start(
        self, population: list[Agent], config: ExperimentConfig,
    ) -> None:
        """Initialize all managers and compute initial hierarchy."""
        self._hierarchy = SocialHierarchyManager(config)
        self._mentorship = MentorshipManager(config)
        self._marriage = MarriageManager(config)
        self._clans = ClanManager(config)
        self._institutions = InstitutionManager(config)
        self._rng = np.random.default_rng(config.random_seed)

        # Initial status computation
        self._hierarchy_metrics = self._hierarchy.update_all(
            population, self._rng,
        )

    def on_generation_start(
        self, generation: int, population: list[Agent],
        config: ExperimentConfig,
    ) -> None:
        """Update clans, institutions, then hierarchy at the start of each generation."""
        if self._hierarchy is None:
            self._hierarchy = SocialHierarchyManager(config)
            self._rng = np.random.default_rng(config.random_seed)
        if self._clans is None:
            self._clans = ClanManager(config)
        if self._institutions is None:
            self._institutions = InstitutionManager(config)

        # Phase D: clans and institutions set extension_data bonuses BEFORE hierarchy
        self._clan_metrics = self._clans.update_clans(
            population, generation, self._rng, config,
        )
        self._institution_metrics = self._institutions.update_institutions(
            population, generation, self._rng, config,
        )

        # Hierarchy reads clan_honor_bonus + institution_prestige_bonus
        self._hierarchy_metrics = self._hierarchy.update_all(
            population, self._rng,
        )

    def on_generation_end(
        self, generation: int, population: list[Agent],
        config: ExperimentConfig,
    ) -> None:
        """Manage mentorships and marriages at end of generation."""
        if self._mentorship is None:
            self._mentorship = MentorshipManager(config)
        if self._marriage is None:
            self._marriage = MarriageManager(config)
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

        # Phase D: process marriages
        self._marriage_metrics = self._marriage.process_marriages(
            population, generation, self._rng, config,
        )

    def on_agent_created(
        self, agent: Agent, parents: tuple[Agent, Agent],
        config: ExperimentConfig,
    ) -> None:
        """Child inherits social bonds from parents' networks and joins clan."""
        # Existing: inherit weak social bonds
        for parent in parents:
            for bonded_id, strength in parent.social_bonds.items():
                if bonded_id != agent.id and strength > 0.3:
                    current = agent.social_bonds.get(bonded_id, 0.0)
                    agent.social_bonds[bonded_id] = max(current, strength * 0.1)

        # Phase D: assign to parent's clan
        if self._clans is not None:
            self._clans.assign_to_child(agent, parents, config)

    def modify_attraction(
        self, agent1: Agent, agent2: Agent, base_score: float,
        config: ExperimentConfig,
    ) -> float:
        """Boost attraction based on roles, clan alliances, and shared institutions."""
        sd = self._get_config(config)
        score = base_score

        # Existing: role compatibility
        role1 = agent1.social_role or "unassigned"
        role2 = agent2.social_role or "unassigned"
        if (role1, role2) in _COMPATIBLE_ROLES:
            score += sd["role_attraction_bonus"]

        # Phase D: clan alliance bonus (political marriage between their clans)
        clan1 = agent1.extension_data.get("clan_id")
        clan2 = agent2.extension_data.get("clan_id")
        if (clan1 is not None and clan2 is not None
                and clan1 != clan2
                and self._clans is not None):
            # Check if there's a political marriage alliance between these clans
            for clan in self._clans.clans.values():
                if clan.id in (clan1, clan2):
                    # Check if the other clan has a political marriage with this one
                    pass
            # Simpler: check if any agent has a political marriage bridging these clans
            if self._marriage is not None:
                for a in (agent1, agent2):
                    m = a.extension_data.get("marriage")
                    if (m is not None and m.get("is_political")
                            and m.get("alliance_clan_ids") is not None):
                        alliance = set(m["alliance_clan_ids"])
                        if clan1 in alliance and clan2 in alliance:
                            score += sd["clan_alliance_attraction_bonus"]
                            break

        # Phase D: same institution bonus
        inst1 = set(agent1.extension_data.get("institution_ids", []))
        inst2 = set(agent2.extension_data.get("institution_ids", []))
        if inst1 & inst2:
            score += sd["same_institution_attraction_bonus"]

        return score

    def modify_decision(
        self, agent: Agent, context: str, utilities: dict[str, float],
        config: ExperimentConfig,
    ) -> dict[str, float]:
        """
        Modify decision utilities based on social role, clan, and institutions.

        - Leaders get bonus for staying (settlement loyalty)
        - Outsider bridges get bonus for migration (connectivity)
        - Clan members with clanmates nearby get stay bonus
        - Council members get stay bonus in their community
        """
        sd = self._get_config(config)
        role = agent.social_role or "unassigned"

        # Existing: role-based modifiers
        if role == "leader" and "stay" in utilities:
            utilities["stay"] = utilities.get("stay", 0.0) + sd["leader_stay_bonus"]
        elif role == "outsider_bridge" and "migrate" in utilities:
            utilities["migrate"] = utilities.get("migrate", 0.0) + sd["bridge_migrate_bonus"]

        # Phase D: clan loyalty — stay bonus if agent is in a clan
        if "stay" in utilities and agent.extension_data.get("clan_id") is not None:
            utilities["stay"] = utilities.get("stay", 0.0) + sd["clan_stay_bonus"]

        # Phase D: council obligation — stay bonus for council members
        if "stay" in utilities:
            inst_ids = agent.extension_data.get("institution_ids", [])
            for iid in inst_ids:
                if iid.startswith("council_"):
                    utilities["stay"] = utilities.get("stay", 0.0) + sd["council_stay_bonus"]
                    break

        return utilities

    def get_metrics(self, population: list[Agent]) -> dict[str, Any]:
        return {
            **self._hierarchy_metrics,
            **self._mentorship_metrics,
            **self._marriage_metrics,
            **self._clan_metrics,
            **self._institution_metrics,
        }
