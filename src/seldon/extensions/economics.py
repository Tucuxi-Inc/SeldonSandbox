"""
Economics Extension — production, trade, markets, and wealth distribution.

Adds proper economics: personality-driven production, inter-settlement
trade routes, supply/demand markets, occupation assignment, and wealth
tracking with Gini coefficient computation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from seldon.extensions.base import SimulationExtension

if TYPE_CHECKING:
    from seldon.core.agent import Agent
    from seldon.core.config import ExperimentConfig


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TradeRoute:
    """Tracks trade between two communities/locations."""
    location_a: str
    location_b: str
    volume: float = 0.0
    goods_a_to_b: dict[str, float] = field(default_factory=dict)
    goods_b_to_a: dict[str, float] = field(default_factory=dict)


@dataclass
class MarketState:
    """Per-community market with supply, demand, and prices."""
    location_id: str
    prices: dict[str, float] = field(default_factory=dict)
    supply: dict[str, float] = field(default_factory=dict)
    demand: dict[str, float] = field(default_factory=dict)


# Occupations and their trait drivers
OCCUPATIONS = {
    "farmer": "conscientiousness",
    "artisan": "creativity",
    "trader": "extraversion",
    "scholar": "openness",
    "soldier": "dominance",
}

RESOURCE_TYPES = ["food", "goods", "knowledge"]


class EconomicsExtension(SimulationExtension):
    """Production, trade, markets, occupations, and wealth tracking."""

    def __init__(self) -> None:
        self.trade_routes: dict[tuple[str, str], TradeRoute] = {}
        self.markets: dict[str, MarketState] = {}
        self.gdp_by_location: dict[str, float] = {}
        self._rng: np.random.Generator | None = None

    @property
    def name(self) -> str:
        return "economics"

    @property
    def description(self) -> str:
        return "Production, trade, markets, and wealth distribution"

    def get_default_config(self) -> dict[str, Any]:
        return {
            "requires": ["geography", "resources"],
            "base_production_rate": 1.0,
            "trade_distance_cost": 0.1,
            "price_elasticity": 0.5,
            "poverty_threshold": 0.2,
            "poverty_mortality_multiplier": 1.5,
            "wealth_inheritance_rate": 0.7,
            "occupation_change_rate": 0.1,
            "specialization_bonus": 0.3,
        }

    def _get_config(self, config: ExperimentConfig) -> dict[str, Any]:
        defaults = self.get_default_config()
        overrides = config.extensions.get("economics", {})
        for k, v in overrides.items():
            if k != "requires":
                defaults[k] = v
        ec = config.economics_config
        for k, v in ec.items():
            if k != "enabled" and k != "requires":
                defaults[k] = v
        return defaults

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------
    def on_simulation_start(
        self, population: list[Agent], config: ExperimentConfig,
    ) -> None:
        self._rng = np.random.default_rng(config.random_seed)
        ec = self._get_config(config)

        # Initialize markets per unique location
        locations = set()
        for a in population:
            if a.location_id:
                locations.add(a.location_id)

        for loc_id in locations:
            self.markets[loc_id] = MarketState(
                location_id=loc_id,
                prices={r: 1.0 for r in RESOURCE_TYPES},
                supply={r: 5.0 for r in RESOURCE_TYPES},
                demand={r: 5.0 for r in RESOURCE_TYPES},
            )

        # Assign initial occupations
        self._assign_occupations(population, config)

    def on_agent_created(
        self, agent: Agent, parents: tuple[Agent, Agent],
        config: ExperimentConfig,
    ) -> None:
        """Children inherit some wealth from parents."""
        ec = self._get_config(config)
        rate = ec.get("wealth_inheritance_rate", 0.7)
        parent_wealth = (parents[0].wealth + parents[1].wealth) / 2
        agent.wealth = parent_wealth * rate * 0.1  # Small fraction at birth

    def on_generation_end(
        self, generation: int, population: list[Agent],
        config: ExperimentConfig,
    ) -> None:
        """Run the economic cycle: production → trade → market update."""
        if self._rng is None:
            self._rng = np.random.default_rng(config.random_seed)

        ec = self._get_config(config)

        # Group by location
        by_location: dict[str, list[Agent]] = {}
        for a in population:
            loc = a.location_id or "default"
            if loc not in by_location:
                by_location[loc] = []
            by_location[loc].append(a)

        # Phase 1: Production
        self.gdp_by_location = {}
        for loc_id, agents in by_location.items():
            gdp = self._run_production(agents, ec, config)
            self.gdp_by_location[loc_id] = gdp

        # Phase 2: Trade routes
        self._update_trade_routes(by_location, ec)

        # Phase 3: Market updates
        self._update_markets(by_location, ec)

        # Phase 4: Occupation reassignment (small fraction each gen)
        if self._rng.random() < ec.get("occupation_change_rate", 0.1):
            self._assign_occupations(population, config)

    # ------------------------------------------------------------------
    # Production
    # ------------------------------------------------------------------
    def _run_production(
        self, agents: list[Agent], ec: dict, config: ExperimentConfig,
    ) -> float:
        """Compute production output per agent based on occupation + traits."""
        ts = config.trait_system
        base_rate = ec.get("base_production_rate", 1.0)
        spec_bonus = ec.get("specialization_bonus", 0.3)
        total_gdp = 0.0

        for agent in agents:
            if not agent.is_alive:
                continue
            occupation = agent.occupation or "farmer"
            trait_name = OCCUPATIONS.get(occupation, "conscientiousness")
            try:
                idx = ts.trait_index(trait_name)
                trait_val = float(agent.traits[idx])
            except KeyError:
                trait_val = 0.5

            output = base_rate * (0.5 + trait_val * (1.0 + spec_bonus))
            agent.wealth += output * 0.1  # Income
            total_gdp += output

        return round(total_gdp, 4)

    # ------------------------------------------------------------------
    # Occupation assignment
    # ------------------------------------------------------------------
    def _assign_occupations(
        self, population: list[Agent], config: ExperimentConfig,
    ) -> None:
        """Assign occupations based on dominant traits."""
        ts = config.trait_system
        for agent in population:
            if not agent.is_alive:
                continue
            if agent.occupation and self._rng and self._rng.random() > 0.3:
                continue  # Most agents keep their occupation

            best_occ = "farmer"
            best_val = -1.0
            for occ, trait_name in OCCUPATIONS.items():
                try:
                    idx = ts.trait_index(trait_name)
                    val = float(agent.traits[idx])
                except KeyError:
                    val = 0.0
                if val > best_val:
                    best_val = val
                    best_occ = occ
            agent.occupation = best_occ

    # ------------------------------------------------------------------
    # Trade routes
    # ------------------------------------------------------------------
    def _update_trade_routes(
        self, by_location: dict[str, list[Agent]], ec: dict,
    ) -> None:
        """Evaluate and update trade between locations."""
        locations = list(by_location.keys())
        dist_cost = ec.get("trade_distance_cost", 0.1)

        for i, loc_a in enumerate(locations):
            for loc_b in locations[i + 1:]:
                key = (loc_a, loc_b) if loc_a < loc_b else (loc_b, loc_a)

                # Trade volume based on combined extraversion
                agents_a = by_location[loc_a]
                agents_b = by_location[loc_b]
                ext_a = np.mean([a.traits[2] if len(a.traits) > 2 else 0.5 for a in agents_a]) if agents_a else 0.5
                ext_b = np.mean([a.traits[2] if len(a.traits) > 2 else 0.5 for a in agents_b]) if agents_b else 0.5
                volume = float((ext_a + ext_b) / 2 * (1.0 - dist_cost))
                volume = max(0.0, volume)

                if key not in self.trade_routes:
                    self.trade_routes[key] = TradeRoute(
                        location_a=key[0], location_b=key[1],
                    )
                self.trade_routes[key].volume = round(volume, 4)

    # ------------------------------------------------------------------
    # Market updates
    # ------------------------------------------------------------------
    def _update_markets(
        self, by_location: dict[str, list[Agent]], ec: dict,
    ) -> None:
        """Update supply/demand/prices per location."""
        elasticity = ec.get("price_elasticity", 0.5)

        for loc_id, agents in by_location.items():
            if loc_id not in self.markets:
                self.markets[loc_id] = MarketState(
                    location_id=loc_id,
                    prices={r: 1.0 for r in RESOURCE_TYPES},
                    supply={r: 5.0 for r in RESOURCE_TYPES},
                    demand={r: 5.0 for r in RESOURCE_TYPES},
                )
            market = self.markets[loc_id]
            pop_size = max(len(agents), 1)

            for rtype in RESOURCE_TYPES:
                # Demand grows with population
                market.demand[rtype] = pop_size * 0.5
                # Supply grows with GDP
                gdp = self.gdp_by_location.get(loc_id, 0.0)
                market.supply[rtype] = max(0.1, gdp * 0.3)
                # Price = base * (demand / supply)^elasticity
                ratio = market.demand[rtype] / max(market.supply[rtype], 0.1)
                market.prices[rtype] = round(
                    float(np.clip(ratio ** elasticity, 0.1, 10.0)), 4,
                )

    # ------------------------------------------------------------------
    # Modifier hooks
    # ------------------------------------------------------------------
    def modify_mortality(
        self, agent: Agent, base_rate: float, config: ExperimentConfig,
    ) -> float:
        """Poverty increases mortality."""
        ec = self._get_config(config)
        poverty_threshold = ec.get("poverty_threshold", 0.2)
        if agent.wealth < poverty_threshold:
            multiplier = ec.get("poverty_mortality_multiplier", 1.5)
            return base_rate * multiplier
        return base_rate

    def modify_decision(
        self, agent: Agent, context: str, utilities: dict[str, float],
        config: ExperimentConfig,
    ) -> dict[str, float]:
        """Wealthy agents stay; poor agents migrate."""
        ec = self._get_config(config)
        poverty_threshold = ec.get("poverty_threshold", 0.2)
        if agent.wealth > 1.0 and "stay" in utilities:
            utilities["stay"] = utilities.get("stay", 0.0) + 0.1
        if agent.wealth < poverty_threshold and "migrate" in utilities:
            utilities["migrate"] = utilities.get("migrate", 0.0) + 0.15
        return utilities

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    def get_metrics(self, population: list[Agent]) -> dict[str, Any]:
        alive = [a for a in population if a.is_alive]
        wealth_vals = [a.wealth for a in alive]
        total_wealth = sum(wealth_vals)
        mean_wealth = total_wealth / max(len(alive), 1)

        # Gini coefficient
        gini = self._compute_gini(wealth_vals)

        # Occupation distribution
        occ_counts: dict[str, int] = {}
        for a in alive:
            occ = a.occupation or "unassigned"
            occ_counts[occ] = occ_counts.get(occ, 0) + 1

        # Trade volume
        total_trade = sum(r.volume for r in self.trade_routes.values())

        # GDP
        total_gdp = sum(self.gdp_by_location.values())

        return {
            "total_gdp": round(total_gdp, 4),
            "total_trade_volume": round(total_trade, 4),
            "mean_wealth": round(mean_wealth, 4),
            "gini_coefficient": round(gini, 4),
            "occupation_distribution": occ_counts,
            "trade_route_count": len(self.trade_routes),
            "market_count": len(self.markets),
        }

    @staticmethod
    def _compute_gini(values: list[float]) -> float:
        """Compute Gini coefficient for a list of values."""
        if not values or len(values) < 2:
            return 0.0
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        total = sum(sorted_vals)
        if total == 0:
            return 0.0
        cumulative = 0.0
        gini_sum = 0.0
        for i, v in enumerate(sorted_vals):
            cumulative += v
            gini_sum += (2 * (i + 1) - n - 1) * v
        return float(np.clip(gini_sum / (n * total), 0.0, 1.0))

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------
    def get_trade_routes_list(self) -> list[dict[str, Any]]:
        return [
            {
                "location_a": r.location_a,
                "location_b": r.location_b,
                "volume": r.volume,
            }
            for r in self.trade_routes.values()
        ]

    def get_market_data(self) -> list[dict[str, Any]]:
        return [
            {
                "location_id": m.location_id,
                "prices": dict(m.prices),
                "supply": {k: round(v, 4) for k, v in m.supply.items()},
                "demand": {k: round(v, 4) for k, v in m.demand.items()},
            }
            for m in self.markets.values()
        ]

    def get_wealth_distribution(self, population: list[Agent]) -> dict[str, Any]:
        """Compute wealth percentiles and Lorenz curve data."""
        alive = [a for a in population if a.is_alive]
        if not alive:
            return {"percentiles": {}, "lorenz_curve": []}

        wealth_vals = sorted([a.wealth for a in alive])
        n = len(wealth_vals)
        total = sum(wealth_vals)

        percentiles = {}
        for p in [10, 25, 50, 75, 90]:
            idx = min(int(n * p / 100), n - 1)
            percentiles[f"p{p}"] = round(wealth_vals[idx], 4)

        # Lorenz curve: (cumulative % of population, cumulative % of wealth)
        lorenz = []
        cumulative = 0.0
        for i, w in enumerate(wealth_vals):
            cumulative += w
            if (i + 1) % max(1, n // 20) == 0 or i == n - 1:
                lorenz.append({
                    "population_pct": round((i + 1) / n, 4),
                    "wealth_pct": round(cumulative / max(total, 1e-10), 4),
                })

        return {
            "percentiles": percentiles,
            "lorenz_curve": lorenz,
            "gini": round(self._compute_gini(wealth_vals), 4),
        }
