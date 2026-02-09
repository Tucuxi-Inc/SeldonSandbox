"""
Diplomacy Extension — inter-community relations, alliances, and rivalries.

Tracks diplomatic standing between communities, forming alliances when
standing is high and rivalries when low. Cultural exchange happens between
allied communities. Community leaders drive diplomatic decisions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from seldon.extensions.base import SimulationExtension
from seldon.social.community import CommunityManager

if TYPE_CHECKING:
    from seldon.core.agent import Agent
    from seldon.core.config import ExperimentConfig
    from seldon.extensions.geography import GeographyExtension


@dataclass
class DiplomaticRelation:
    """Tracks the diplomatic state between two communities."""
    community_a: str
    community_b: str
    standing: float = 0.0  # [-1, +1]
    alliance: bool = False
    rivalry: bool = False
    history: list[dict[str, Any]] = field(default_factory=list)


class DiplomacyExtension(SimulationExtension):
    """Inter-community diplomacy: alliances, rivalries, cultural exchange."""

    def __init__(self, geography: GeographyExtension) -> None:
        self.geography = geography
        self.relations: dict[tuple[str, str], DiplomaticRelation] = {}
        self.community_profiles: dict[str, dict[str, Any]] = {}
        self._community_manager: CommunityManager | None = None
        self._rng: np.random.Generator | None = None

    @property
    def name(self) -> str:
        return "diplomacy"

    @property
    def description(self) -> str:
        return "Inter-community diplomacy with alliances, rivalries, and cultural exchange"

    def get_default_config(self) -> dict[str, Any]:
        return {
            "requires": ["geography"],
            "standing_learning_rate": 0.05,
            "alliance_threshold": 0.7,
            "rivalry_threshold": -0.5,
            "cultural_exchange_rate": 0.1,
            "trait_compatibility_weight": 0.3,
            "leader_compatibility_weight": 0.2,
            "resource_competition_weight": 0.25,
            "cultural_similarity_weight": 0.25,
        }

    def _get_config(self, config: ExperimentConfig) -> dict[str, Any]:
        defaults = self.get_default_config()
        overrides = config.extensions.get("diplomacy", {})
        for k, v in overrides.items():
            if k != "requires":
                defaults[k] = v
        # Also merge diplomacy_config from ExperimentConfig
        dc = config.diplomacy_config
        for k, v in dc.items():
            if k != "enabled" and k != "requires":
                defaults[k] = v
        return defaults

    def _get_community_manager(self, config: ExperimentConfig) -> CommunityManager:
        if self._community_manager is None:
            self._community_manager = CommunityManager(config)
        return self._community_manager

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------
    def on_simulation_start(
        self, population: list[Agent], config: ExperimentConfig,
    ) -> None:
        self._rng = np.random.default_rng(config.random_seed)
        # Initialize community IDs from location IDs
        for agent in population:
            if agent.community_id is None and agent.location_id:
                agent.community_id = agent.location_id

    def on_agent_created(
        self, agent: Agent, parents: tuple[Agent, Agent],
        config: ExperimentConfig,
    ) -> None:
        """Children inherit community from first parent."""
        if parents and parents[0].community_id:
            agent.community_id = parents[0].community_id

    def on_generation_end(
        self, generation: int, population: list[Agent],
        config: ExperimentConfig,
    ) -> None:
        """Update diplomatic relations between all community pairs."""
        if self._rng is None:
            self._rng = np.random.default_rng(config.random_seed)

        cm = self._get_community_manager(config)
        dc = self._get_config(config)
        communities = cm.get_communities(population)

        # Update profiles
        self.community_profiles = {}
        for cid, agents in communities.items():
            self.community_profiles[cid] = cm.compute_personality_profile(agents)

        # Update relations for each pair
        community_ids = list(communities.keys())
        for i, cid_a in enumerate(community_ids):
            for cid_b in community_ids[i + 1:]:
                key = (cid_a, cid_b) if cid_a < cid_b else (cid_b, cid_a)
                if key not in self.relations:
                    self.relations[key] = DiplomaticRelation(
                        community_a=key[0], community_b=key[1],
                    )
                rel = self.relations[key]
                self._update_standing(
                    rel, communities[cid_a], communities[cid_b],
                    cm, dc, generation,
                )
                self._check_transitions(rel, dc, generation)

        # Cultural exchange between allied communities
        for key, rel in self.relations.items():
            if rel.alliance:
                self._cultural_exchange(
                    communities.get(key[0], []),
                    communities.get(key[1], []),
                    dc,
                )

    def _update_standing(
        self,
        rel: DiplomaticRelation,
        agents_a: list[Agent],
        agents_b: list[Agent],
        cm: CommunityManager,
        dc: dict[str, Any],
        generation: int,
    ) -> None:
        """Compute standing delta from multiple factors."""
        # Trait compatibility
        trait_compat = cm.trait_compatibility(agents_a, agents_b)

        # Leader compatibility: compare top-status agents
        leader_a = max(agents_a, key=lambda a: a.social_status) if agents_a else None
        leader_b = max(agents_b, key=lambda a: a.social_status) if agents_b else None
        if leader_a and leader_b:
            leader_compat = 1.0 - float(np.linalg.norm(
                leader_a.traits - leader_b.traits
            )) / np.sqrt(len(leader_a.traits))
            leader_compat = max(0.0, leader_compat)
        else:
            leader_compat = 0.5

        # Cultural similarity (Jaccard of memes)
        memes_a = set()
        memes_b = set()
        for a in agents_a:
            memes_a.update(a.cultural_memes)
        for a in agents_b:
            memes_b.update(a.cultural_memes)
        if memes_a or memes_b:
            cultural_sim = len(memes_a & memes_b) / max(len(memes_a | memes_b), 1)
        else:
            cultural_sim = 0.5

        # Resource competition (simple: population density comparison)
        resource_comp = 0.5  # Neutral baseline
        if self.geography.locations:
            loc_a_ids = {a.location_id for a in agents_a if a.location_id}
            loc_b_ids = {a.location_id for a in agents_b if a.location_id}
            overlap = loc_a_ids & loc_b_ids
            if overlap:
                resource_comp = max(0.0, 0.5 - 0.1 * len(overlap))

        # Weighted delta
        delta = (
            (trait_compat - 0.5) * dc.get("trait_compatibility_weight", 0.3)
            + (leader_compat - 0.5) * dc.get("leader_compatibility_weight", 0.2)
            + (resource_comp - 0.5) * dc.get("resource_competition_weight", 0.25)
            + (cultural_sim - 0.5) * dc.get("cultural_similarity_weight", 0.25)
        )

        lr = dc.get("standing_learning_rate", 0.05)
        rel.standing = float(np.clip(rel.standing + delta * lr, -1.0, 1.0))

    def _check_transitions(
        self, rel: DiplomaticRelation, dc: dict[str, Any], generation: int,
    ) -> None:
        """Check for alliance/rivalry transitions."""
        alliance_t = dc.get("alliance_threshold", 0.7)
        rivalry_t = dc.get("rivalry_threshold", -0.5)

        old_alliance = rel.alliance
        old_rivalry = rel.rivalry

        rel.alliance = rel.standing >= alliance_t
        rel.rivalry = rel.standing <= rivalry_t

        if rel.alliance != old_alliance:
            event = "alliance_formed" if rel.alliance else "alliance_dissolved"
            rel.history.append({"generation": generation, "event": event, "standing": round(rel.standing, 4)})
        if rel.rivalry != old_rivalry:
            event = "rivalry_declared" if rel.rivalry else "rivalry_ended"
            rel.history.append({"generation": generation, "event": event, "standing": round(rel.standing, 4)})

    def _cultural_exchange(
        self, agents_a: list[Agent], agents_b: list[Agent],
        dc: dict[str, Any],
    ) -> None:
        """Allied communities share cultural memes."""
        rate = dc.get("cultural_exchange_rate", 0.1)
        if not agents_a or not agents_b or self._rng is None:
            return

        memes_a = set()
        memes_b = set()
        for a in agents_a:
            memes_a.update(a.cultural_memes)
        for a in agents_b:
            memes_b.update(a.cultural_memes)

        # Share unique memes with probability
        unique_a = memes_a - memes_b
        unique_b = memes_b - memes_a

        for meme in unique_a:
            if self._rng.random() < rate:
                target = agents_b[self._rng.integers(len(agents_b))]
                if meme not in target.cultural_memes:
                    target.cultural_memes.append(meme)

        for meme in unique_b:
            if self._rng.random() < rate:
                target = agents_a[self._rng.integers(len(agents_a))]
                if meme not in target.cultural_memes:
                    target.cultural_memes.append(meme)

    # ------------------------------------------------------------------
    # Modifier hooks
    # ------------------------------------------------------------------
    def modify_attraction(
        self, agent1: Agent, agent2: Agent, base_score: float,
        config: ExperimentConfig,
    ) -> float:
        """Boost attraction for allies, penalize for rivals."""
        c1 = agent1.community_id or agent1.location_id
        c2 = agent2.community_id or agent2.location_id
        if not c1 or not c2 or c1 == c2:
            return base_score

        key = (c1, c2) if c1 < c2 else (c2, c1)
        rel = self.relations.get(key)
        if not rel:
            return base_score

        if rel.alliance:
            return base_score * 1.15  # 15% boost
        if rel.rivalry:
            return base_score * 0.7  # 30% penalty
        return base_score

    def modify_decision(
        self, agent: Agent, context: str, utilities: dict[str, float],
        config: ExperimentConfig,
    ) -> dict[str, float]:
        """Alliance members favor cooperation; rivals favor defense."""
        cid = agent.community_id or agent.location_id
        if not cid:
            return utilities

        # Check if this agent's community has any alliances or rivalries
        has_alliance = False
        has_rivalry = False
        for key, rel in self.relations.items():
            if cid in key:
                if rel.alliance:
                    has_alliance = True
                if rel.rivalry:
                    has_rivalry = True

        if has_alliance and "cooperate" in utilities:
            utilities["cooperate"] = utilities.get("cooperate", 0.0) + 0.1
        if has_rivalry and "defend" in utilities:
            utilities["defend"] = utilities.get("defend", 0.0) + 0.15
        if has_rivalry and "stay" in utilities:
            utilities["stay"] = utilities.get("stay", 0.0) + 0.05

        return utilities

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    def get_metrics(self, population: list[Agent]) -> dict[str, Any]:
        alliances = sum(1 for r in self.relations.values() if r.alliance)
        rivalries = sum(1 for r in self.relations.values() if r.rivalry)
        standings = [r.standing for r in self.relations.values()]
        mean_standing = float(np.mean(standings)) if standings else 0.0

        return {
            "total_relations": len(self.relations),
            "alliances": alliances,
            "rivalries": rivalries,
            "mean_standing": round(mean_standing, 4),
            "community_count": len(self.community_profiles),
        }

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------
    def get_all_relations(self) -> list[dict[str, Any]]:
        """Get all diplomatic relations as serializable dicts."""
        return [
            {
                "community_a": r.community_a,
                "community_b": r.community_b,
                "standing": round(r.standing, 4),
                "alliance": r.alliance,
                "rivalry": r.rivalry,
                "history": r.history,
            }
            for r in self.relations.values()
        ]
