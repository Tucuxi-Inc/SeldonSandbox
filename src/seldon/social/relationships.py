"""
Relationship Manager — pairing, dissolution, and infidelity dynamics.

Handles the full lifecycle of agent relationships:
- Dissolution: incompatible pairs may separate
- Infidelity: optional trait-driven infidelity model
- Pairing: attraction-weighted partner selection with cooldowns
- Single-by-choice: some agents prefer independence
- LGBTQ support: same-sex pairing with optional assisted reproduction
"""

from __future__ import annotations

from typing import Any

import numpy as np

from seldon.core.agent import Agent
from seldon.core.attraction import AttractionModel
from seldon.core.config import ExperimentConfig


class RelationshipManager:
    """
    Manages relationship formation, maintenance, and dissolution.

    All parameters come from config.relationship_config.
    """

    def __init__(self, config: ExperimentConfig, attraction: AttractionModel):
        self.config = config
        self.attraction = attraction
        self.rc = config.relationship_config

        # Cache config values
        self.pairing_min_age: int = self.rc["pairing_min_age"]
        self.dissolution_enabled: bool = self.rc["dissolution_enabled"]
        self.dissolution_threshold: float = self.rc["dissolution_compatibility_threshold"]
        self.dissolution_base_rate: float = self.rc["dissolution_base_rate"]
        self.infidelity_enabled: bool = self.rc["infidelity_enabled"]
        self.infidelity_base_rate: float = self.rc["infidelity_base_rate"]
        self.single_by_choice_rate: float = self.rc["single_by_choice_rate"]
        self.lgbtq_rate: float = self.rc["lgbtq_rate"]
        self.cooldown: int = self.rc["reparing_cooldown_generations"]

    def process_dissolutions(
        self,
        population: list[Agent],
        generation: int,
        rng: np.random.Generator,
    ) -> list[tuple[str, str]]:
        """
        Process pair dissolutions based on compatibility.

        Returns list of (agent1_id, agent2_id) tuples that dissolved.
        """
        if not self.dissolution_enabled:
            return []

        dissolved: list[tuple[str, str]] = []
        agent_map = {a.id: a for a in population}

        # Find all unique pairs
        seen_pairs: set[tuple[str, str]] = set()
        for agent in population:
            if agent.partner_id is None or not agent.is_alive:
                continue
            pair = tuple(sorted([agent.id, agent.partner_id]))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)

            partner = agent_map.get(agent.partner_id)
            if partner is None or not partner.is_alive:
                continue

            # Calculate compatibility (attraction score)
            compatibility = self.attraction.calculate(agent, partner, rng)

            # Low compatibility increases dissolution probability
            if compatibility < self.dissolution_threshold:
                dissolution_prob = self.dissolution_base_rate * (
                    1.0 + (self.dissolution_threshold - compatibility)
                )
                dissolution_prob = min(dissolution_prob, 0.5)

                if rng.random() < dissolution_prob:
                    # Dissolve the pair
                    agent.partner_id = None
                    agent.relationship_status = "dissolved"
                    partner.partner_id = None
                    partner.relationship_status = "dissolved"
                    # Record dissolution generation for cooldown
                    agent.extension_data["last_dissolution_gen"] = generation
                    partner.extension_data["last_dissolution_gen"] = generation
                    dissolved.append((agent.id, partner.id))

        return dissolved

    def form_pairs(
        self,
        population: list[Agent],
        generation: int,
        rng: np.random.Generator,
    ) -> list[tuple[Agent, Agent]]:
        """
        Form new pairs from eligible agents using attraction-weighted selection.

        Respects:
        - Minimum age requirement
        - Cooldown period after dissolution
        - Single-by-choice preference
        - LGBTQ pairing
        """
        eligible = [
            a for a in population
            if self._is_eligible(a, generation)
        ]

        # Remove single-by-choice agents
        filtered = []
        for agent in eligible:
            if rng.random() < self.single_by_choice_rate:
                continue
            filtered.append(agent)

        pairs: list[tuple[Agent, Agent]] = []
        unpaired = list(filtered)
        rng.shuffle(unpaired)

        while len(unpaired) >= 2:
            a1 = unpaired.pop(0)

            # Calculate attraction to all remaining
            scores = np.array([
                self.attraction.calculate(a1, a2, rng) for a2 in unpaired
            ])
            scores = np.maximum(scores, 0.0)

            if scores.sum() <= 0:
                continue

            # Weighted random selection
            probs = scores / scores.sum()
            idx = rng.choice(len(unpaired), p=probs)
            partner = unpaired.pop(idx)

            # Link them
            a1.partner_id = partner.id
            partner.partner_id = a1.id
            a1.relationship_status = "paired"
            partner.relationship_status = "paired"

            pairs.append((a1, partner))

        return pairs

    def check_infidelity(
        self,
        population: list[Agent],
        generation: int,
        rng: np.random.Generator,
    ) -> list[dict[str, Any]]:
        """
        Check for infidelity events among paired agents.

        Infidelity probability is modified by traits:
        - Higher with low self_control, low agreeableness
        - Lower with high conscientiousness, high trust

        Returns list of infidelity event dicts.
        """
        if not self.infidelity_enabled:
            return []

        events: list[dict[str, Any]] = []
        ts = self.config.trait_system

        for agent in population:
            if agent.partner_id is None or not agent.is_alive:
                continue

            # Trait-modified infidelity probability
            try:
                self_control = agent.traits[ts.trait_index("self_control")]
                agreeableness = agent.traits[ts.trait_index("agreeableness")]
                conscientiousness = agent.traits[ts.trait_index("conscientiousness")]
            except KeyError:
                self_control = 0.5
                agreeableness = 0.5
                conscientiousness = 0.5

            # Higher self-control/agreeableness/conscientiousness reduce infidelity
            modifier = (1.0 - self_control) * 0.3 + (1.0 - agreeableness) * 0.3 + (1.0 - conscientiousness) * 0.2
            prob = self.infidelity_base_rate * (0.5 + modifier)

            if rng.random() < prob:
                events.append({
                    "agent_id": agent.id,
                    "partner_id": agent.partner_id,
                    "generation": generation,
                    "probability": prob,
                })

        return events

    def _is_eligible(self, agent: Agent, generation: int) -> bool:
        """Check if an agent is eligible for pairing."""
        if not agent.is_eligible_for_pairing(self.pairing_min_age):
            return False

        # Check cooldown after dissolution
        last_dissolution = agent.extension_data.get("last_dissolution_gen")
        if last_dissolution is not None:
            if generation - last_dissolution < self.cooldown:
                return False

        return True
