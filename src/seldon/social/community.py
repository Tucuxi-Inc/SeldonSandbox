"""
Community Manager — personality profiles, cohesion, identity metrics.

Gives each settlement/community an emergent "personality" computed from
the aggregate traits of its members, weighted by influence. Detects
internal factions via trait clustering and computes distinctiveness
relative to other communities.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from seldon.core.agent import Agent
    from seldon.core.config import ExperimentConfig
    from seldon.core.traits import TraitSystem


# Character labels based on dominant trait profiles
_CHARACTER_LABELS = {
    "openness": "exploratory",
    "conscientiousness": "disciplined",
    "extraversion": "sociable",
    "agreeableness": "harmonious",
    "neuroticism": "anxious",
    "creativity": "innovative",
    "resilience": "resilient",
    "ambition": "ambitious",
    "empathy": "empathic",
    "depth_drive": "contemplative",
}


class CommunityManager:
    """Computes community-level personality profiles and cohesion metrics."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.ts = config.trait_system
        self.cc = config.community_config

    # ------------------------------------------------------------------
    # Community detection
    # ------------------------------------------------------------------
    def get_communities(
        self, population: list[Agent],
    ) -> dict[str, list[Agent]]:
        """
        Group agents by community_id (or location_id as fallback).

        Returns dict[community_id, list[Agent]].
        """
        communities: dict[str, list[Agent]] = {}
        for agent in population:
            if not agent.is_alive:
                continue
            cid = agent.community_id or agent.location_id or "default"
            if cid not in communities:
                communities[cid] = []
            communities[cid].append(agent)
        return communities

    # ------------------------------------------------------------------
    # Personality profile
    # ------------------------------------------------------------------
    def compute_personality_profile(
        self, community_agents: list[Agent],
    ) -> dict[str, Any]:
        """
        Compute aggregate personality profile for a community.

        Uses influence-weighted mean traits. Returns dict with
        trait means, variance, dominant region, and character label.
        """
        if not community_agents:
            return {"trait_means": {}, "trait_variance": 0.0, "dominant_region": "unknown", "character": "empty"}

        # Influence-weighted traits
        weights = np.array([max(a.influence_score, 0.1) for a in community_agents])
        weights = weights / weights.sum()

        trait_matrix = np.array([a.traits for a in community_agents])
        weighted_means = np.average(trait_matrix, axis=0, weights=weights)
        trait_variance = float(np.mean(np.var(trait_matrix, axis=0)))

        # Dominant processing region
        region_counts: dict[str, int] = {}
        for a in community_agents:
            r = a.processing_region.value
            region_counts[r] = region_counts.get(r, 0) + 1
        dominant_region = max(region_counts, key=region_counts.get) if region_counts else "unknown"

        # Character label from top trait
        trait_names = self.ts.names()
        named_means = {trait_names[i]: float(weighted_means[i]) for i in range(self.ts.count)}
        top_trait = max(named_means, key=named_means.get)
        character = _CHARACTER_LABELS.get(top_trait, "balanced")

        # Trait skewness (are traits concentrated or spread?)
        skewness = float(np.std(weighted_means))

        return {
            "trait_means": {k: round(v, 4) for k, v in named_means.items()},
            "trait_variance": round(trait_variance, 4),
            "trait_skewness": round(skewness, 4),
            "dominant_region": dominant_region,
            "region_distribution": region_counts,
            "character": character,
            "size": len(community_agents),
        }

    # ------------------------------------------------------------------
    # Cohesion
    # ------------------------------------------------------------------
    def compute_cohesion(
        self, community_agents: list[Agent],
    ) -> float:
        """
        Compute community cohesion [0, 1].

        Based on trait variance (low = high cohesion) and internal
        social bond density.
        """
        if len(community_agents) < 2:
            return 1.0

        # Trait cohesion: inverse of variance
        trait_matrix = np.array([a.traits for a in community_agents])
        trait_var = float(np.mean(np.var(trait_matrix, axis=0)))
        trait_cohesion = max(0.0, 1.0 - trait_var * 4)  # Scale so var=0.25→cohesion=0

        # Bond density: fraction of possible internal bonds that exist
        agent_ids = {a.id for a in community_agents}
        total_possible = len(community_agents) * (len(community_agents) - 1) / 2
        if total_possible == 0:
            bond_density = 1.0
        else:
            internal_bonds = 0
            for a in community_agents:
                for bonded_id in a.social_bonds:
                    if bonded_id in agent_ids:
                        internal_bonds += 1
            bond_density = min(1.0, internal_bonds / (2 * total_possible))

        # Weighted combination
        tw = self.cc.get("cohesion_trait_weight", 0.4)
        bw = self.cc.get("cohesion_bond_weight", 0.3)
        # Remaining weight distributed equally to culture and conflict (future)
        remaining = 1.0 - tw - bw

        cohesion = trait_cohesion * tw + bond_density * bw + 0.5 * remaining
        return float(np.clip(cohesion, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Identity / Distinctiveness
    # ------------------------------------------------------------------
    def compute_identity(
        self, community_agents: list[Agent],
        all_communities: dict[str, list[Agent]],
        community_id: str,
    ) -> dict[str, Any]:
        """
        Compute community identity and distinctiveness.

        Distinctiveness = L2 distance of this community's trait means
        from the mean of all other communities.
        """
        profile = self.compute_personality_profile(community_agents)
        if not profile["trait_means"]:
            return {"distinctiveness": 0.0, "character": "empty"}

        this_means = np.array(list(profile["trait_means"].values()))

        # Compute other communities' means
        other_means = []
        for cid, agents in all_communities.items():
            if cid == community_id or not agents:
                continue
            other_profile = self.compute_personality_profile(agents)
            if other_profile["trait_means"]:
                other_means.append(np.array(list(other_profile["trait_means"].values())))

        if not other_means:
            distinctiveness = 0.0
        else:
            avg_other = np.mean(other_means, axis=0)
            distinctiveness = float(np.linalg.norm(this_means - avg_other))

        return {
            "distinctiveness": round(distinctiveness, 4),
            "character": profile["character"],
            "dominant_region": profile["dominant_region"],
        }

    # ------------------------------------------------------------------
    # Faction detection
    # ------------------------------------------------------------------
    def detect_factions(
        self, community_agents: list[Agent],
    ) -> list[dict[str, Any]]:
        """
        Detect factions within a community via simple 2-means clustering.

        Returns a list of factions if the community is divisible,
        otherwise an empty list.
        """
        if len(community_agents) < self.cc.get("min_community_size", 5) * 2:
            return []

        trait_matrix = np.array([a.traits for a in community_agents])
        n = len(community_agents)

        # Simple 2-means clustering (avoid scipy dependency)
        rng = np.random.default_rng(42)
        idx1, idx2 = rng.choice(n, size=2, replace=False)
        c1 = trait_matrix[idx1].copy()
        c2 = trait_matrix[idx2].copy()

        for _ in range(20):
            # Assign
            d1 = np.linalg.norm(trait_matrix - c1, axis=1)
            d2 = np.linalg.norm(trait_matrix - c2, axis=1)
            labels = (d2 < d1).astype(int)  # 0=closer to c1, 1=closer to c2

            g0 = trait_matrix[labels == 0]
            g1 = trait_matrix[labels == 1]
            if len(g0) == 0 or len(g1) == 0:
                return []
            c1 = g0.mean(axis=0)
            c2 = g1.mean(axis=0)

        # Check if factions are sufficiently distinct
        inter_distance = float(np.linalg.norm(c1 - c2))
        threshold = self.cc.get("faction_detection_threshold", 0.3)
        if inter_distance < threshold:
            return []

        # Build faction profiles
        factions = []
        for label_val, centroid in [(0, c1), (1, c2)]:
            members = [community_agents[i] for i in range(n) if labels[i] == label_val]
            trait_names = self.ts.names()
            factions.append({
                "size": len(members),
                "trait_means": {
                    trait_names[j]: round(float(centroid[j]), 4)
                    for j in range(self.ts.count)
                },
                "member_ids": [a.id for a in members[:20]],  # Cap for serialization
            })

        return factions

    # ------------------------------------------------------------------
    # Trait compatibility between communities
    # ------------------------------------------------------------------
    def trait_compatibility(
        self, community_a: list[Agent], community_b: list[Agent],
    ) -> float:
        """
        Compute trait compatibility between two communities [0, 1].

        1 = identical profiles, 0 = maximally different.
        """
        if not community_a or not community_b:
            return 0.5

        means_a = np.mean([a.traits for a in community_a], axis=0)
        means_b = np.mean([a.traits for a in community_b], axis=0)

        # L2 distance, normalized
        max_distance = np.sqrt(self.ts.count)  # max possible L2 in [0,1]^N
        distance = float(np.linalg.norm(means_a - means_b))
        return float(np.clip(1.0 - distance / max_distance, 0.0, 1.0))
