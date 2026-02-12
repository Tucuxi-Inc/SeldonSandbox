"""
Social Hierarchy Manager — status computation, influence propagation, role assignment.

Computes emergent social structure from agent traits, contribution history,
and social bonds. Status and roles are personality-driven, not arbitrary.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from seldon.core.processing import ProcessingRegion

if TYPE_CHECKING:
    from seldon.core.agent import Agent
    from seldon.core.config import ExperimentConfig


# Region prestige ordering (higher = more prestige)
_REGION_PRESTIGE = {
    ProcessingRegion.OPTIMAL: 1.0,
    ProcessingRegion.DEEP: 0.9,
    ProcessingRegion.UNDER_PROCESSING: 0.5,
    ProcessingRegion.SACRIFICIAL: 0.4,
    ProcessingRegion.PATHOLOGICAL: 0.1,
}


class SocialHierarchyManager:
    """Computes status, influence, and role assignments for agents."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.ts = config.trait_system
        self.hc = config.hierarchy_config

    def compute_status(
        self, agent: Agent, population: list[Agent],
    ) -> float:
        """
        Compute social status for an agent.

        Status = weighted sum of:
        - Contribution history mean
        - Age factor (experience)
        - Family size (children count)
        - Processing region prestige
        - Social bond count
        """
        hc = self.hc
        components: list[float] = []

        # 1. Contribution (mean of history, normalized to [0,1])
        if agent.contribution_history:
            mean_contrib = float(np.mean(agent.contribution_history))
            contrib_score = float(np.clip(mean_contrib / 2.0, 0.0, 1.0))
        else:
            contrib_score = 0.0
        components.append(contrib_score * hc["status_contribution_weight"])

        # 2. Age factor (peaks around 30-50, declines after)
        age = int(agent.age)
        if age < 16:
            age_score = age / 16.0 * 0.3
        elif age <= 50:
            age_score = 0.3 + 0.7 * min((age - 16) / 34.0, 1.0)
        else:
            age_score = max(0.3, 1.0 - (age - 50) * 0.02)
        components.append(age_score * hc["status_age_weight"])

        # 3. Family (children count, capped at 5 for max score)
        real_children = [c for c in agent.children_ids if c != "stillborn"]
        family_score = min(len(real_children) / 5.0, 1.0)
        # Bonus if partnered
        if agent.partner_id is not None:
            family_score = min(family_score + 0.1, 1.0)
        components.append(family_score * hc["status_family_weight"])

        # 4. Region prestige
        region_score = _REGION_PRESTIGE.get(agent.processing_region, 0.5)
        components.append(region_score * hc["status_region_weight"])

        # 5. Social bonds
        bond_count = len(agent.social_bonds)
        social_score = min(bond_count / 10.0, 1.0)
        components.append(social_score * hc["status_social_weight"])

        status = float(np.clip(sum(components), 0.0, 1.0))

        # Phase D bonuses (additive, from extension_data — 0.0 when Phase D not active)
        clan_bonus = agent.extension_data.get("clan_honor_bonus", 0.0)
        institution_bonus = agent.extension_data.get("institution_prestige_bonus", 0.0)
        status = min(status + clan_bonus + institution_bonus, 1.0)

        return status

    def compute_influence(
        self, agent: Agent, population: list[Agent],
    ) -> float:
        """
        Compute influence score: status * social reach * extraversion.
        """
        try:
            extra_idx = self.ts.trait_index("extraversion")
            extraversion = float(agent.traits[extra_idx])
        except KeyError:
            extraversion = 0.5

        bond_count = len(agent.social_bonds)
        bond_factor = min(bond_count / 10.0, 1.0)

        return float(np.clip(
            agent.social_status * (0.5 + 0.3 * bond_factor + 0.2 * extraversion),
            0.0, 1.0,
        ))

    def assign_roles(
        self, population: list[Agent],
    ) -> dict[str, str]:
        """
        Assign social roles based on traits and status.

        Returns mapping of agent_id -> role name.
        """
        if not population:
            return {}

        thresholds = self.hc["role_thresholds"]
        roles: dict[str, str] = {}

        # Compute status percentile threshold for leaders
        statuses = [a.social_status for a in population]
        if statuses:
            leader_cutoff = float(np.percentile(
                statuses, thresholds["leader_percentile"] * 100,
            ))
        else:
            leader_cutoff = 1.0

        for agent in population:
            role = self._classify_role(agent, leader_cutoff, thresholds)
            roles[agent.id] = role

        return roles

    def _classify_role(
        self, agent: Agent, leader_cutoff: float,
        thresholds: dict[str, Any],
    ) -> str:
        """Classify a single agent's role."""
        try:
            extra = float(agent.traits[self.ts.trait_index("extraversion")])
        except KeyError:
            extra = 0.5
        try:
            creativity = float(agent.traits[self.ts.trait_index("creativity")])
        except KeyError:
            creativity = 0.5
        try:
            agreeableness = float(agent.traits[self.ts.trait_index("agreeableness")])
        except KeyError:
            agreeableness = 0.5
        try:
            conscientiousness = float(agent.traits[self.ts.trait_index("conscientiousness")])
        except KeyError:
            conscientiousness = 0.5

        # Leader: top status + high extraversion
        if (agent.social_status >= leader_cutoff
                and extra >= thresholds["leader_extraversion_min"]):
            return "leader"

        # Innovator: R3/R4 + high creativity
        if (agent.processing_region in (
                ProcessingRegion.DEEP, ProcessingRegion.SACRIFICIAL)
                and creativity >= thresholds["innovator_creativity_min"]):
            return "innovator"

        # Mediator: high agreeableness
        if agreeableness >= thresholds["mediator_agreeableness_min"]:
            return "mediator"

        # Outsider bridge: outsider/descendant with social bonds
        if agent.is_descendant_of_outsider and len(agent.social_bonds) >= 3:
            return "outsider_bridge"

        # Worker: high conscientiousness + R2
        if (conscientiousness >= 0.6
                and agent.processing_region == ProcessingRegion.OPTIMAL):
            return "worker"

        return "unassigned"

    def update_social_bonds_from_hierarchy(
        self, population: list[Agent], rng: np.random.Generator,
    ) -> None:
        """
        Aspiration bonding: agents form weak bonds toward higher-status agents.

        Each generation, a small fraction of agents form new bonds toward
        higher-status agents they aren't already connected to.
        """
        if len(population) < 2:
            return

        # Only process a random subset to keep O(N)
        sample_size = min(len(population), max(5, len(population) // 5))
        indices = rng.choice(len(population), size=sample_size, replace=False)

        for idx in indices:
            agent = population[idx]
            # Find a random higher-status agent
            candidates = [
                a for a in population
                if a.id != agent.id
                and a.social_status > agent.social_status
                and a.id not in agent.social_bonds
            ]
            if not candidates:
                continue

            # Pick one weighted by status difference
            weights = np.array([c.social_status - agent.social_status for c in candidates])
            weights = weights / weights.sum()
            chosen = candidates[rng.choice(len(candidates), p=weights)]

            # Form weak bond
            agent.social_bonds[chosen.id] = 0.1
            chosen.social_bonds[agent.id] = max(
                chosen.social_bonds.get(agent.id, 0.0), 0.05,
            )

    def update_all(
        self, population: list[Agent], rng: np.random.Generator,
    ) -> dict[str, Any]:
        """
        Full hierarchy update: status → influence → roles → aspiration bonds.

        Returns metrics dict.
        """
        # Compute status for all agents
        for agent in population:
            agent.social_status = self.compute_status(agent, population)

        # Compute influence
        for agent in population:
            agent.influence_score = self.compute_influence(agent, population)

        # Assign roles
        role_map = self.assign_roles(population)
        for agent in population:
            agent.social_role = role_map.get(agent.id, "unassigned")

        # Aspiration bonding
        self.update_social_bonds_from_hierarchy(population, rng)

        # Compute metrics
        role_counts: dict[str, int] = {}
        status_values: list[float] = []
        for agent in population:
            role_counts[agent.social_role or "unassigned"] = (
                role_counts.get(agent.social_role or "unassigned", 0) + 1
            )
            status_values.append(agent.social_status)

        influence_gini = self._compute_gini(
            [a.influence_score for a in population]
        )

        return {
            "role_counts": role_counts,
            "mean_status": float(np.mean(status_values)) if status_values else 0.0,
            "status_std": float(np.std(status_values)) if status_values else 0.0,
            "influence_gini": influence_gini,
        }

    @staticmethod
    def _compute_gini(values: list[float]) -> float:
        """Compute Gini coefficient of a list of values."""
        if not values or max(values) == 0:
            return 0.0
        arr = np.array(sorted(values), dtype=float)
        n = len(arr)
        index = np.arange(1, n + 1)
        total = np.sum(arr)
        if total == 0:
            return 0.0
        return float(
            (2.0 * np.sum(index * arr) - (n + 1) * total) / (n * total)
        )
