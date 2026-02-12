"""
Institution Manager — formal councils (elder governance) and guilds (occupational).

Councils form in communities with enough elders. Guilds form when enough
agents share an occupation with sufficient skill. Leaders are elected by
influence score. Institutions grant prestige bonuses to members.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from seldon.core.agent import Agent
    from seldon.core.config import ExperimentConfig


@dataclass
class Institution:
    """A formal institution (council or guild)."""

    id: str
    type: str  # "council" or "guild"
    community_id: str | None = None
    occupation: str | None = None
    member_ids: set[str] = field(default_factory=set)
    leader_id: str | None = None
    prestige: float = 0.0
    founding_generation: int = 0
    last_election_generation: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "community_id": self.community_id,
            "occupation": self.occupation,
            "member_ids": list(self.member_ids),
            "leader_id": self.leader_id,
            "prestige": round(self.prestige, 4),
            "founding_generation": self.founding_generation,
            "member_count": len(self.member_ids),
        }


class InstitutionManager:
    """Manages formal institutions: elder councils and occupational guilds."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.ic = config.institutions_config
        self.institutions: dict[str, Institution] = {}

    def update_institutions(
        self,
        population: list[Agent],
        generation: int,
        rng: np.random.Generator,
        config: ExperimentConfig,
    ) -> dict[str, Any]:
        """
        Full institution lifecycle update.

        1. Detect/update councils from communities with elders
        2. Detect/update guilds from occupational groups
        3. Run elections when due
        4. Compute prestige
        5. Set agent bonuses
        6. Prune dissolved institutions

        Returns metrics dict.
        """
        if not self.ic.get("enabled", True):
            return {"institutions_enabled": False}

        council_min_elders = self.ic.get("council_min_elders", 3)
        council_elder_min_age = self.ic.get("council_elder_min_age", 40)
        council_min_community = self.ic.get("council_min_community_size", 10)
        guild_min_members = self.ic.get("guild_min_members", 5)
        guild_min_skill = self.ic.get("guild_min_skill", 0.3)
        prestige_weight = self.ic.get("institution_prestige_weight", 0.1)
        election_freq = self.ic.get("election_frequency_generations", 5)

        living = [a for a in population if a.is_alive]
        living_map = {a.id: a for a in living}

        # 1. Detect councils
        self._update_councils(
            living, living_map, generation,
            council_min_elders, council_elder_min_age, council_min_community,
        )

        # 2. Detect guilds
        self._update_guilds(
            living, living_map, generation,
            guild_min_members, guild_min_skill,
        )

        # 3. Elections
        for inst in self.institutions.values():
            if generation - inst.last_election_generation >= election_freq:
                self._run_election(inst, living_map, generation)

        # 4. Compute prestige
        for inst in self.institutions.values():
            influences = [
                living_map[mid].influence_score
                for mid in inst.member_ids
                if mid in living_map
            ]
            inst.prestige = float(np.mean(influences)) if influences else 0.0

        # 5. Set agent bonuses
        # First clear old institution data
        for agent in living:
            agent.extension_data.pop("institution_ids", None)
            agent.extension_data.pop("institution_prestige_bonus", None)

        agent_institutions: dict[str, list[str]] = {}
        agent_prestige: dict[str, float] = {}

        for inst in self.institutions.values():
            for mid in inst.member_ids:
                agent_institutions.setdefault(mid, []).append(inst.id)
                agent_prestige[mid] = (
                    agent_prestige.get(mid, 0.0) + inst.prestige * prestige_weight
                )

        for aid, inst_ids in agent_institutions.items():
            if aid in living_map:
                living_map[aid].extension_data["institution_ids"] = inst_ids
                living_map[aid].extension_data["institution_prestige_bonus"] = (
                    agent_prestige.get(aid, 0.0)
                )

        # 6. Prune institutions with no members
        to_remove = [
            iid for iid, inst in self.institutions.items()
            if not inst.member_ids
        ]
        for iid in to_remove:
            del self.institutions[iid]

        return {
            "council_count": sum(
                1 for i in self.institutions.values() if i.type == "council"
            ),
            "guild_count": sum(
                1 for i in self.institutions.values() if i.type == "guild"
            ),
            "total_institution_members": sum(
                len(i.member_ids) for i in self.institutions.values()
            ),
            "mean_prestige": (
                float(np.mean([i.prestige for i in self.institutions.values()]))
                if self.institutions else 0.0
            ),
        }

    def _update_councils(
        self,
        living: list[Agent],
        living_map: dict[str, Agent],
        generation: int,
        min_elders: int,
        elder_min_age: int,
        min_community_size: int,
    ) -> None:
        """Detect or update elder councils per community."""
        # Group by community
        communities: dict[str, list[Agent]] = {}
        for agent in living:
            if agent.community_id:
                communities.setdefault(agent.community_id, []).append(agent)

        active_council_ids: set[str] = set()

        for comm_id, members in communities.items():
            if len(members) < min_community_size:
                continue

            elders = [a for a in members if int(a.age) >= elder_min_age]
            if len(elders) < min_elders:
                continue

            council_id = f"council_{comm_id}"
            active_council_ids.add(council_id)

            if council_id not in self.institutions:
                self.institutions[council_id] = Institution(
                    id=council_id,
                    type="council",
                    community_id=comm_id,
                    founding_generation=generation,
                    last_election_generation=-1,  # Triggers election on first update
                )

            # Update membership to current elders
            self.institutions[council_id].member_ids = {a.id for a in elders}

        # Remove councils for communities that no longer qualify
        to_remove = [
            iid for iid, inst in self.institutions.items()
            if inst.type == "council" and iid not in active_council_ids
        ]
        for iid in to_remove:
            del self.institutions[iid]

    def _update_guilds(
        self,
        living: list[Agent],
        living_map: dict[str, Agent],
        generation: int,
        min_members: int,
        min_skill: float,
    ) -> None:
        """Detect or update occupational guilds."""
        # Group by occupation
        occupations: dict[str, list[Agent]] = {}
        for agent in living:
            if agent.occupation:
                occupations.setdefault(agent.occupation, []).append(agent)

        active_guild_ids: set[str] = set()

        for occ, members in occupations.items():
            # Filter to skilled members
            skilled = [
                a for a in members
                if a.skills.get(occ, 0.0) >= min_skill
            ]
            if len(skilled) < min_members:
                continue

            guild_id = f"guild_{occ}"
            active_guild_ids.add(guild_id)

            if guild_id not in self.institutions:
                self.institutions[guild_id] = Institution(
                    id=guild_id,
                    type="guild",
                    occupation=occ,
                    founding_generation=generation,
                    last_election_generation=-1,  # Triggers election on first update
                )

            # Update membership
            self.institutions[guild_id].member_ids = {a.id for a in skilled}

        # Remove guilds for occupations that no longer qualify
        to_remove = [
            iid for iid, inst in self.institutions.items()
            if inst.type == "guild" and iid not in active_guild_ids
        ]
        for iid in to_remove:
            del self.institutions[iid]

    def _run_election(
        self,
        institution: Institution,
        living_map: dict[str, Agent],
        generation: int,
    ) -> None:
        """Elect leader by highest influence score among members."""
        best_id: str | None = None
        best_influence: float = -1.0

        for mid in institution.member_ids:
            agent = living_map.get(mid)
            if agent is not None and agent.influence_score > best_influence:
                best_influence = agent.influence_score
                best_id = mid

        institution.leader_id = best_id
        institution.last_election_generation = generation

    def get_institution_data(self) -> list[dict[str, Any]]:
        """Return institution summaries for API consumption."""
        return [inst.to_dict() for inst in self.institutions.values()]
