"""
Clan Manager — multi-generational kinship groups with shared honor and rivalries.

Detects clan founders (high-status agents with living descendants), assigns
members by tracing ancestry, computes clan honor, and detects inter-clan
rivalries within shared settlements.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from seldon.core.agent import Agent
    from seldon.core.config import ExperimentConfig


@dataclass
class Clan:
    """A multi-generational kinship group."""

    id: str
    founder_id: str
    member_ids: set[str] = field(default_factory=set)
    honor: float = 0.0
    founding_generation: int = 0
    rival_clan_ids: set[str] = field(default_factory=set)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "founder_id": self.founder_id,
            "member_ids": list(self.member_ids),
            "honor": round(self.honor, 4),
            "founding_generation": self.founding_generation,
            "rival_clan_ids": list(self.rival_clan_ids),
        }


class ClanManager:
    """Manages family clan formation, membership, honor, and rivalries."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.cc = config.clan_config
        self.clans: dict[str, Clan] = {}

    def update_clans(
        self,
        population: list[Agent],
        generation: int,
        rng: np.random.Generator,
        config: ExperimentConfig,
    ) -> dict[str, Any]:
        """
        Full clan lifecycle update.

        1. Detect founders (high-status with descendants)
        2. Assign members by ancestry tracing
        3. Prune small clans
        4. Update honor
        5. Set agent extension_data bonuses
        6. Detect rivalries

        Returns metrics dict.
        """
        if not self.cc.get("enabled", True):
            return {"clans_enabled": False}

        min_members = self.cc.get("min_living_members", 3)
        founder_min_status = self.cc.get("founder_min_status", 0.6)
        honor_weight = self.cc.get("honor_status_weight", 0.1)
        rival_threshold = self.cc.get("rival_threshold", 0.3)
        max_clans = self.cc.get("max_clans", 10)
        max_depth = self.cc.get("max_depth", 3)

        living = [a for a in population if a.is_alive]
        living_map = {a.id: a for a in living}

        # Build parent→children map for living agents
        children_map: dict[str, list[str]] = {}
        for agent in living:
            for pid in (agent.parent1_id, agent.parent2_id):
                if pid is not None:
                    children_map.setdefault(pid, []).append(agent.id)

        # 1. Detect potential founders
        potential_founders = [
            a for a in living
            if (a.social_status >= founder_min_status
                and a.id not in {c.founder_id for c in self.clans.values()})
        ]

        # Check descendant count for new founders
        for founder in potential_founders:
            if len(self.clans) >= max_clans:
                break
            descendants = self._trace_living_descendants(
                founder.id, children_map, max_depth,
            )
            if len(descendants) >= min_members:
                clan_id = f"clan_{founder.id}"
                if clan_id not in self.clans:
                    self.clans[clan_id] = Clan(
                        id=clan_id,
                        founder_id=founder.id,
                        founding_generation=generation,
                    )

        # 2. Assign members: clear and rebuild
        for clan in self.clans.values():
            clan.member_ids.clear()

        # Build ancestor→clan map
        founder_to_clan: dict[str, str] = {
            c.founder_id: c.id for c in self.clans.values()
        }

        # For each living agent, trace ancestry up to max_depth
        for agent in living:
            clan_id = self._find_clan_for_agent(
                agent, living_map, founder_to_clan, max_depth,
            )
            if clan_id is not None:
                self.clans[clan_id].member_ids.add(agent.id)

        # 3. Prune clans below minimum
        to_remove = [
            cid for cid, clan in self.clans.items()
            if len(clan.member_ids) < min_members
        ]
        for cid in to_remove:
            # Clear extension_data for former members
            for mid in self.clans[cid].member_ids:
                if mid in living_map:
                    living_map[mid].extension_data.pop("clan_id", None)
                    living_map[mid].extension_data.pop("clan_honor_bonus", None)
            del self.clans[cid]

        # 4. Update honor
        for clan in self.clans.values():
            statuses = [
                living_map[mid].social_status
                for mid in clan.member_ids
                if mid in living_map
            ]
            clan.honor = float(np.mean(statuses)) if statuses else 0.0

        # 5. Set agent bonuses
        for clan in self.clans.values():
            for mid in clan.member_ids:
                if mid in living_map:
                    living_map[mid].extension_data["clan_id"] = clan.id
                    living_map[mid].extension_data["clan_honor_bonus"] = (
                        clan.honor * honor_weight
                    )

        # Clear clan data for agents not in any clan
        clan_members = set()
        for clan in self.clans.values():
            clan_members.update(clan.member_ids)
        for agent in living:
            if agent.id not in clan_members:
                agent.extension_data.pop("clan_id", None)
                agent.extension_data.pop("clan_honor_bonus", None)

        # 6. Detect rivalries
        self._detect_rivalries(living_map, rival_threshold)

        return {
            "clan_count": len(self.clans),
            "total_clan_members": sum(len(c.member_ids) for c in self.clans.values()),
            "mean_honor": (
                float(np.mean([c.honor for c in self.clans.values()]))
                if self.clans else 0.0
            ),
            "rivalry_count": sum(
                len(c.rival_clan_ids) for c in self.clans.values()
            ) // 2,
        }

    def assign_to_child(
        self,
        child: Agent,
        parents: tuple[Agent, Agent],
        config: ExperimentConfig,
    ) -> None:
        """Assign a newborn to the higher-honor parent's clan."""
        if not self.cc.get("enabled", True):
            return

        best_clan_id: str | None = None
        best_honor: float = -1.0

        for parent in parents:
            clan_id = parent.extension_data.get("clan_id")
            if clan_id is not None and clan_id in self.clans:
                clan = self.clans[clan_id]
                if clan.honor > best_honor:
                    best_honor = clan.honor
                    best_clan_id = clan_id

        if best_clan_id is not None:
            self.clans[best_clan_id].member_ids.add(child.id)
            child.extension_data["clan_id"] = best_clan_id
            honor_weight = self.cc.get("honor_status_weight", 0.1)
            child.extension_data["clan_honor_bonus"] = (
                self.clans[best_clan_id].honor * honor_weight
            )

    def _trace_living_descendants(
        self,
        agent_id: str,
        children_map: dict[str, list[str]],
        max_depth: int,
    ) -> set[str]:
        """Trace living descendants from an agent up to max_depth."""
        descendants: set[str] = set()
        frontier = {agent_id}

        for _ in range(max_depth):
            next_frontier: set[str] = set()
            for aid in frontier:
                for child_id in children_map.get(aid, []):
                    if child_id not in descendants:
                        descendants.add(child_id)
                        next_frontier.add(child_id)
            frontier = next_frontier
            if not frontier:
                break

        return descendants

    def _find_clan_for_agent(
        self,
        agent: Agent,
        living_map: dict[str, Agent],
        founder_to_clan: dict[str, str],
        max_depth: int,
    ) -> str | None:
        """Find which clan an agent belongs to by tracing ancestry."""
        # Check if agent itself is a founder
        if agent.id in founder_to_clan:
            return founder_to_clan[agent.id]

        # BFS upward through parents
        frontier = {agent.id}
        visited: set[str] = {agent.id}

        for _ in range(max_depth):
            next_frontier: set[str] = set()
            for aid in frontier:
                a = living_map.get(aid)
                if a is None:
                    continue
                for pid in (a.parent1_id, a.parent2_id):
                    if pid is not None and pid not in visited:
                        visited.add(pid)
                        if pid in founder_to_clan:
                            return founder_to_clan[pid]
                        next_frontier.add(pid)
            frontier = next_frontier
            if not frontier:
                break

        return None

    def _detect_rivalries(
        self,
        living_map: dict[str, Agent],
        rival_threshold: float,
    ) -> None:
        """Detect rivalries: clans sharing a community with large honor gap."""
        # Clear existing rivalries
        for clan in self.clans.values():
            clan.rival_clan_ids.clear()

        # Group clans by community
        community_clans: dict[str, list[str]] = {}
        for clan in self.clans.values():
            for mid in clan.member_ids:
                agent = living_map.get(mid)
                if agent and agent.community_id:
                    community_clans.setdefault(agent.community_id, [])
                    if clan.id not in community_clans[agent.community_id]:
                        community_clans[agent.community_id].append(clan.id)

        # Check honor differences within communities
        for clan_ids in community_clans.values():
            for i, cid1 in enumerate(clan_ids):
                for cid2 in clan_ids[i + 1:]:
                    c1 = self.clans.get(cid1)
                    c2 = self.clans.get(cid2)
                    if c1 is not None and c2 is not None:
                        if abs(c1.honor - c2.honor) > rival_threshold:
                            c1.rival_clan_ids.add(c2.id)
                            c2.rival_clan_ids.add(c1.id)

    def get_clan_data(self) -> list[dict[str, Any]]:
        """Return clan summaries for API consumption."""
        return [clan.to_dict() for clan in self.clans.values()]
