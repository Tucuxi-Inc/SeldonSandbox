"""
Resources Extension — scarcity dynamics.

Tracks food, shelter, and status resource pools. Scarcity increases
mortality and reduces fertility. Distribution can be equal or unequal.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from seldon.extensions.base import SimulationExtension

if TYPE_CHECKING:
    from seldon.core.agent import Agent
    from seldon.core.config import ExperimentConfig


class ResourcesExtension(SimulationExtension):
    """Resource pools with scarcity effects on mortality and fertility."""

    @property
    def name(self) -> str:
        return "resources"

    @property
    def description(self) -> str:
        return "Resource scarcity dynamics affecting mortality and fertility"

    def get_default_config(self) -> dict[str, Any]:
        return {
            "resource_types": ["food", "shelter", "status"],
            "base_regeneration_rate": 0.1,
            "consumption_per_agent": 0.05,
            "initial_pool_per_type": 10.0,
            "scarcity_threshold": 0.3,
            "scarcity_mortality_multiplier": 2.0,
            "scarcity_fertility_multiplier": 0.5,
            "scarcity_conflict_multiplier": 1.5,
            "distribution": "equal",
            "hoarding_enabled": False,
        }

    def __init__(self) -> None:
        self.resource_pools: dict[str, float] = {}
        self.scarcity_index: float = 0.0
        self._population_size: int = 0

    def _get_config(self, config: ExperimentConfig) -> dict[str, Any]:
        defaults = self.get_default_config()
        overrides = config.extensions.get("resources", {})
        defaults.update(overrides)
        return defaults

    def on_simulation_start(
        self, population: list[Agent], config: ExperimentConfig,
    ) -> None:
        """Initialize resource pools."""
        res = self._get_config(config)
        initial = res["initial_pool_per_type"]
        self.resource_pools = {
            rtype: initial for rtype in res["resource_types"]
        }
        self._population_size = len(population)
        self._update_scarcity()

    def on_generation_end(
        self, generation: int, population: list[Agent],
        config: ExperimentConfig,
    ) -> None:
        """Regenerate resources, consume per agent, recalculate scarcity.

        When economics settlements exist, aggregate their resource pools
        into the global pools to keep scarcity_index accurate.
        """
        res = self._get_config(config)
        self._population_size = len(population)
        regen_rate = res["base_regeneration_rate"]
        consumption = res["consumption_per_agent"]

        for rtype in self.resource_pools:
            # Regeneration
            self.resource_pools[rtype] += regen_rate * self.resource_pools[rtype]
            # Consumption
            total_consumed = consumption * len(population)
            self.resource_pools[rtype] = max(
                0.0, self.resource_pools[rtype] - total_consumed,
            )

        # Phase C: aggregate settlement resource pools into global pools
        self._aggregate_settlement_pools(config)

        self._update_scarcity()

    def _aggregate_settlement_pools(
        self, config: ExperimentConfig,
    ) -> None:
        """If economics extension has settlements, fold their pools in."""
        from seldon.extensions.economics import EconomicsExtension
        # Find economics extension via config — we check extensions dict
        # for the economics extension reference stored externally
        # Since we don't have direct registry access, check if any settlement
        # data was stored. We access it via the _economics_ref if set.
        econ_ref = getattr(self, '_economics_ref', None)
        if econ_ref is not None and hasattr(econ_ref, 'settlements'):
            for settlement in econ_ref.settlements.values():
                for rtype, amount in settlement.resource_pools.items():
                    if rtype in self.resource_pools:
                        self.resource_pools[rtype] += amount * 0.1

    def modify_mortality(
        self, agent: Agent, base_rate: float,
        config: ExperimentConfig,
    ) -> float:
        """Increase mortality during scarcity."""
        res = self._get_config(config)
        if self.scarcity_index > res["scarcity_threshold"]:
            scarcity_factor = self.scarcity_index * res["scarcity_mortality_multiplier"]
            return base_rate * (1.0 + scarcity_factor)
        return base_rate

    def get_metrics(self, population: list[Agent]) -> dict[str, Any]:
        return {
            "resource_levels": dict(self.resource_pools),
            "scarcity_index": self.scarcity_index,
            "distribution_gini": self._compute_gini(population),
        }

    # --- Helpers ---

    def _update_scarcity(self) -> None:
        """Compute scarcity index (0 = abundant, 1 = completely scarce)."""
        if not self.resource_pools:
            self.scarcity_index = 0.0
            return
        per_capita = sum(self.resource_pools.values()) / max(
            self._population_size, 1,
        )
        # Scarcity rises as per-capita resources drop below 1.0
        self.scarcity_index = float(np.clip(1.0 - per_capita, 0.0, 1.0))

    def _compute_gini(self, population: list[Agent]) -> float:
        """Compute Gini coefficient of resource holdings across agents."""
        if not population:
            return 0.0
        holdings = [
            sum(a.resource_holdings.values()) for a in population
        ]
        if not holdings or max(holdings) == 0:
            return 0.0
        holdings_arr = np.array(sorted(holdings), dtype=float)
        n = len(holdings_arr)
        index = np.arange(1, n + 1)
        return float(
            (2.0 * np.sum(index * holdings_arr) - (n + 1) * np.sum(holdings_arr))
            / (n * np.sum(holdings_arr))
        ) if np.sum(holdings_arr) > 0 else 0.0
