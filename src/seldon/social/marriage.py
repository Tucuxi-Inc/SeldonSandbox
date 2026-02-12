"""
Marriage Manager — formalization of pair bonds with property sharing and divorce.

Detects newly paired agents, tracks courtship duration, formalizes marriages
after a configurable delay, handles divorce and widowing. Political marriages
between different clans create alliance bonds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from seldon.core.agent import Agent
    from seldon.core.config import ExperimentConfig


@dataclass
class MarriageContract:
    """Formal marriage contract between two agents."""

    partner1_id: str
    partner2_id: str
    generation_formed: int
    shared_wealth: float = 0.0
    is_political: bool = False
    alliance_clan_ids: tuple[str, str] | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "partner1_id": self.partner1_id,
            "partner2_id": self.partner2_id,
            "generation_formed": self.generation_formed,
            "shared_wealth": self.shared_wealth,
            "is_political": self.is_political,
        }
        if self.alliance_clan_ids is not None:
            d["alliance_clan_ids"] = list(self.alliance_clan_ids)
        return d


class MarriageManager:
    """Manages formal marriage contracts, courtship, divorce, and property."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.mc = config.marriage_config
        self.ts = config.trait_system
        self._divorce_count: int = 0

    def process_marriages(
        self,
        population: list[Agent],
        generation: int,
        rng: np.random.Generator,
        config: ExperimentConfig,
    ) -> dict[str, Any]:
        """
        Process marriage lifecycle for the population.

        1. Track courtship start for newly paired agents
        2. Formalize marriages after courtship delay
        3. Check for divorce
        4. Handle widowing

        Returns metrics dict.
        """
        if not self.mc.get("enabled", True):
            return {"marriages_enabled": False}

        delay = self.mc.get("formalization_delay_generations", 1)
        sharing_rate = self.mc.get("property_sharing_rate", 0.5)
        divorce_base = self.mc.get("divorce_base_rate", 0.02)

        agent_map = {a.id: a for a in population if a.is_alive}
        new_marriages = 0
        divorces = 0
        widowed_cleared = 0

        for agent in population:
            if not agent.is_alive:
                continue

            marriage = agent.extension_data.get("marriage")

            # --- Handle existing marriages ---
            if marriage is not None:
                partner = agent_map.get(marriage["partner2_id"]
                                        if marriage["partner1_id"] == agent.id
                                        else marriage["partner1_id"])

                # Widowing: partner dead
                if partner is None or not partner.is_alive:
                    agent.extension_data.pop("marriage", None)
                    agent.extension_data.pop("courtship_start_gen", None)
                    widowed_cleared += 1
                    continue

                # Only process divorce from one side (partner1)
                if marriage["partner1_id"] != agent.id:
                    continue

                # Divorce check
                divorce_prob = self._compute_divorce_probability(
                    agent, partner, divorce_base,
                )
                if rng.random() < divorce_prob:
                    self._execute_divorce(agent, partner, marriage)
                    divorces += 1
                continue

            # --- No marriage yet: check for new pairing ---
            if agent.partner_id is None:
                # Clear stale courtship if no longer paired
                agent.extension_data.pop("courtship_start_gen", None)
                continue

            # Agent is paired but not married — track courtship
            courtship_start = agent.extension_data.get("courtship_start_gen")
            if courtship_start is None:
                agent.extension_data["courtship_start_gen"] = generation
                courtship_start = generation

            # Check if courtship period is complete
            if generation - courtship_start >= delay:
                partner = agent_map.get(agent.partner_id)
                if partner is not None and partner.is_alive:
                    contract = self._create_contract(
                        agent, partner, generation, sharing_rate,
                    )
                    contract_dict = contract.to_dict()
                    agent.extension_data["marriage"] = contract_dict
                    partner.extension_data["marriage"] = contract_dict
                    agent.relationship_status = "married"
                    partner.relationship_status = "married"
                    # Clear courtship tracking
                    agent.extension_data.pop("courtship_start_gen", None)
                    partner.extension_data.pop("courtship_start_gen", None)
                    new_marriages += 1

        self._divorce_count += divorces

        return {
            "new_marriages": new_marriages,
            "divorces": divorces,
            "widowed_cleared": widowed_cleared,
            "total_divorces": self._divorce_count,
        }

    def _create_contract(
        self,
        agent1: Agent,
        agent2: Agent,
        generation: int,
        sharing_rate: float,
    ) -> MarriageContract:
        """Create a marriage contract and pool wealth."""
        shared = (agent1.wealth + agent2.wealth) * sharing_rate
        remainder_each = (agent1.wealth + agent2.wealth) * (1 - sharing_rate) / 2
        agent1.wealth = remainder_each
        agent2.wealth = remainder_each

        # Detect political marriage (cross-clan)
        clan1 = agent1.extension_data.get("clan_id")
        clan2 = agent2.extension_data.get("clan_id")
        is_political = (
            self.mc.get("political_marriage_enabled", True)
            and clan1 is not None
            and clan2 is not None
            and clan1 != clan2
        )

        return MarriageContract(
            partner1_id=agent1.id,
            partner2_id=agent2.id,
            generation_formed=generation,
            shared_wealth=shared,
            is_political=is_political,
            alliance_clan_ids=(clan1, clan2) if is_political else None,
        )

    def _compute_divorce_probability(
        self, agent1: Agent, agent2: Agent, base_rate: float,
    ) -> float:
        """Compute divorce probability from base rate and personality."""
        try:
            agree_idx = self.ts.trait_index("agreeableness")
            cons_idx = self.ts.trait_index("conscientiousness")
            agree_mean = (
                float(agent1.traits[agree_idx]) + float(agent2.traits[agree_idx])
            ) / 2
            cons_mean = (
                float(agent1.traits[cons_idx]) + float(agent2.traits[cons_idx])
            ) / 2
            stability = (agree_mean + cons_mean) / 2
        except KeyError:
            stability = 0.5

        return float(np.clip(base_rate * (1 - stability), 0.0, 1.0))

    def _execute_divorce(
        self, agent1: Agent, agent2: Agent, marriage: dict[str, Any],
    ) -> None:
        """Execute divorce: split wealth, clear contracts, update status."""
        shared = marriage.get("shared_wealth", 0.0)
        agent1.wealth += shared / 2
        agent2.wealth += shared / 2

        agent1.extension_data.pop("marriage", None)
        agent2.extension_data.pop("marriage", None)
        agent1.extension_data.pop("courtship_start_gen", None)
        agent2.extension_data.pop("courtship_start_gen", None)

        agent1.relationship_status = "dissolved"
        agent2.relationship_status = "dissolved"
        agent1.partner_id = None
        agent2.partner_id = None

    def get_political_marriages(self, population: list[Agent]) -> list[dict[str, Any]]:
        """Return all active political (cross-clan) marriages."""
        seen: set[str] = set()
        political: list[dict[str, Any]] = []

        for agent in population:
            if not agent.is_alive:
                continue
            marriage = agent.extension_data.get("marriage")
            if marriage is None or not marriage.get("is_political", False):
                continue
            key = tuple(sorted([marriage["partner1_id"], marriage["partner2_id"]]))
            if key in seen:
                continue
            seen.add(key)
            political.append(marriage)

        return political

    def get_marriage_stats(
        self, population: list[Agent], current_generation: int,
    ) -> dict[str, Any]:
        """Compute aggregate marriage statistics."""
        married_count = 0
        durations: list[int] = []
        seen: set[str] = set()

        for agent in population:
            if not agent.is_alive:
                continue
            marriage = agent.extension_data.get("marriage")
            if marriage is None:
                continue
            key = tuple(sorted([marriage["partner1_id"], marriage["partner2_id"]]))
            if key in seen:
                continue
            seen.add(key)
            married_count += 1
            durations.append(current_generation - marriage["generation_formed"])

        return {
            "married_count": married_count,
            "avg_duration": float(np.mean(durations)) if durations else 0.0,
            "total_divorces": self._divorce_count,
        }
