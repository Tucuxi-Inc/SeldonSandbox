"""
Economics Extension — production, trade, markets, and wealth distribution.

Adds proper economics: personality-driven production, inter-settlement
trade routes, supply/demand markets, occupation assignment, and wealth
tracking with Gini coefficient computation.

Phase C additions: Settlement dataclass, tick-level production/consumption,
skill accumulation, inter-settlement trade with goods flow, and governance
(taxation + poverty relief + infrastructure).
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
    distance: float = 0.0
    trader_count: int = 0


@dataclass
class MarketState:
    """Per-community market with supply, demand, and prices."""
    location_id: str
    prices: dict[str, float] = field(default_factory=dict)
    supply: dict[str, float] = field(default_factory=dict)
    demand: dict[str, float] = field(default_factory=dict)


@dataclass
class Settlement:
    """Hex-cluster settlement with resource economy (Phase C)."""
    id: str
    center: tuple[int, int]
    tiles: list[tuple[int, int]]
    agent_ids: list[str]
    terrain_types: set[str]
    resource_pools: dict[str, float] = field(default_factory=lambda: {
        "food": 0.0, "goods": 0.0, "knowledge": 0.0,
    })
    production_last_tick: dict[str, float] = field(default_factory=lambda: {
        "food": 0.0, "goods": 0.0, "knowledge": 0.0,
    })
    consumption_last_tick: dict[str, float] = field(default_factory=lambda: {
        "food": 0.0, "goods": 0.0, "knowledge": 0.0,
    })
    communal_pool: float = 0.0
    taxation_rate: float = 0.1
    leader_id: str | None = None
    infrastructure_level: float = 0.0


# Occupations and their trait drivers
OCCUPATIONS = {
    "farmer": "conscientiousness",
    "artisan": "creativity",
    "trader": "extraversion",
    "scholar": "openness",
    "soldier": "dominance",
}

# What resource type each occupation produces
OCCUPATION_OUTPUT: dict[str, str] = {
    "farmer": "food",
    "artisan": "goods",
    "trader": "goods",
    "scholar": "knowledge",
    "soldier": "food",
}

RESOURCE_TYPES = ["food", "goods", "knowledge"]


class EconomicsExtension(SimulationExtension):
    """Production, trade, markets, occupations, and wealth tracking."""

    def __init__(self) -> None:
        self.trade_routes: dict[tuple[str, str], TradeRoute] = {}
        self.markets: dict[str, MarketState] = {}
        self.gdp_by_location: dict[str, float] = {}
        self._rng: np.random.Generator | None = None
        # Phase C: settlement economy
        self.settlements: dict[str, Settlement] = {}
        self._tick_engine_ref: Any = None
        self._hex_enabled: bool = False

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

        # Detect if hex-enabled tick engine is driving us
        if self._tick_engine_ref is not None:
            hex_grid = getattr(self._tick_engine_ref, 'hex_grid', None)
            self._hex_enabled = hex_grid is not None

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

        # Phase C: initial settlement detection when hex-enabled
        if self._hex_enabled:
            self.detect_settlements(
                self._tick_engine_ref, population, config,
            )

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
        """Run the economic cycle: production -> trade -> market update.

        When hex-enabled (Phase C), tick-level production/consumption/trade
        already ran, so we only do occupation reassignment and market update.
        """
        if self._rng is None:
            self._rng = np.random.default_rng(config.random_seed)

        ec = self._get_config(config)

        if self._hex_enabled:
            # Tick-level economy already handled production/consumption/trade.
            # Only do occupation reassignment and market sync from settlements.
            if self._rng.random() < ec.get("occupation_change_rate", 0.1):
                self._assign_occupations(population, config)
            self._sync_markets_from_settlements(ec)
            return

        # --- Legacy generation-level economy ---
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
        self._update_trade_routes(by_location, ec, config)

        # Phase 3: Market updates
        self._update_markets(by_location, ec)

        # Phase 4: Occupation reassignment (small fraction each gen)
        if self._rng.random() < ec.get("occupation_change_rate", 0.1):
            self._assign_occupations(population, config)

    # ------------------------------------------------------------------
    # Phase C: Settlement detection
    # ------------------------------------------------------------------
    def detect_settlements(
        self, tick_engine: Any, population: list[Agent],
        config: ExperimentConfig,
    ) -> None:
        """Detect settlements from hex clusters and assign agents."""
        ec = self._get_config(config)
        min_size = ec.get("min_settlement_size", 3)
        clusters = tick_engine.get_agent_clusters(min_size)

        # Build agent lookup
        agent_map = {a.id: a for a in population if a.is_alive}

        old_ids = set(self.settlements.keys())
        new_ids: set[str] = set()

        for cluster in clusters:
            center = cluster["center"]
            sid = f"settlement_{center[0]}_{center[1]}"
            new_ids.add(sid)

            if sid in self.settlements:
                # Update existing settlement
                s = self.settlements[sid]
                s.tiles = cluster["tiles"]
                s.agent_ids = cluster["agent_ids"]
                s.terrain_types = cluster["terrain_types"]
                s.center = center
            else:
                # Create new settlement
                agent_count = cluster["agent_count"]
                s = Settlement(
                    id=sid,
                    center=center,
                    tiles=cluster["tiles"],
                    agent_ids=cluster["agent_ids"],
                    terrain_types=cluster["terrain_types"],
                    resource_pools={r: 2.0 * agent_count for r in RESOURCE_TYPES},
                )
                self.settlements[sid] = s

            # Set community_id on all agents in this settlement
            for aid in cluster["agent_ids"]:
                if aid in agent_map:
                    agent_map[aid].community_id = sid

        # Remove stale settlements
        for old_sid in old_ids - new_ids:
            del self.settlements[old_sid]

    # ------------------------------------------------------------------
    # Phase C: Tick-level production + consumption + skills
    # ------------------------------------------------------------------
    def process_economic_tick(
        self, population: list[Agent], config: ExperimentConfig,
        tick_in_year: int, season: str,
    ) -> None:
        """Run one tick of the settlement economy."""
        if not self.settlements:
            return

        ec = self._get_config(config)
        ts = config.trait_system
        tpy = config.tick_config.get("ticks_per_year", 12)

        agent_map = {a.id: a for a in population if a.is_alive}

        for settlement in self.settlements.values():
            # Reset per-tick accumulators
            settlement.production_last_tick = {r: 0.0 for r in RESOURCE_TYPES}
            settlement.consumption_last_tick = {r: 0.0 for r in RESOURCE_TYPES}

            for aid in settlement.agent_ids:
                agent = agent_map.get(aid)
                if agent is None:
                    continue

                # Production
                produced = self._produce_tick(
                    agent, settlement, ec, ts, tpy, season,
                )

                # Consumption
                self._consume_tick(agent, settlement, ec, tpy)

                # Skill accumulation
                self._accumulate_skill(agent, ec)

        # Inter-settlement trade (runs every trade_frequency_ticks ticks)
        freq = ec.get("trade_frequency_ticks", 3)
        if freq > 0 and tick_in_year % freq == 0:
            self._run_inter_settlement_trade(population, config)

    def _produce_tick(
        self, agent: Agent, settlement: Settlement,
        ec: dict, ts: Any, tpy: int, season: str,
    ) -> float:
        """Produce resources for one agent in one tick."""
        occupation = agent.occupation or "farmer"
        resource_type = OCCUPATION_OUTPUT.get(occupation, "food")
        trait_name = OCCUPATIONS.get(occupation, "conscientiousness")

        try:
            idx = ts.trait_index(trait_name)
            trait_val = float(agent.traits[idx])
        except KeyError:
            trait_val = 0.5

        base_rate = ec.get("base_production_rate", 1.0)
        spec_bonus = ec.get("specialization_bonus", 0.3)
        skill_val = agent.skills.get(occupation, 0.0)

        # Base production scaled to tick
        output = base_rate * (0.5 + trait_val * (1.0 + spec_bonus))
        output *= (1.0 + skill_val * spec_bonus)
        output /= tpy

        # Terrain modifier
        terrain = agent.extension_data.get("terrain_type", "")
        terrain_mods = ec.get("terrain_production_modifiers", {})
        occ_terrain = terrain_mods.get(occupation, {})
        terrain_mult = occ_terrain.get(terrain, 1.0)
        output *= terrain_mult

        # Season modifier
        season_mods = ec.get("season_production_modifiers", {})
        season_resource_mods = season_mods.get(season, {})
        season_mult = season_resource_mods.get(resource_type, 1.0)
        output *= season_mult

        output = max(0.0, output)

        # Tax
        tax_rate = settlement.taxation_rate
        tax_amount = output * tax_rate
        after_tax = output - tax_amount
        settlement.communal_pool += tax_amount

        # Add to settlement pool and agent wealth
        settlement.resource_pools[resource_type] = (
            settlement.resource_pools.get(resource_type, 0.0) + output
        )
        settlement.production_last_tick[resource_type] = (
            settlement.production_last_tick.get(resource_type, 0.0) + output
        )
        agent.wealth += after_tax * 0.1
        agent.resource_holdings[resource_type] = (
            agent.resource_holdings.get(resource_type, 0.0) + after_tax * 0.1
        )

        return output

    def _consume_tick(
        self, agent: Agent, settlement: Settlement,
        ec: dict, tpy: int,
    ) -> None:
        """Consume resources from settlement pool for one agent."""
        consumption_rates = ec.get("consumption_per_tick", {
            "food": 0.03, "goods": 0.01, "knowledge": 0.005,
        })

        for rtype, rate in consumption_rates.items():
            pool = settlement.resource_pools.get(rtype, 0.0)
            consumed = min(rate, pool)
            settlement.resource_pools[rtype] = pool - consumed
            settlement.consumption_last_tick[rtype] = (
                settlement.consumption_last_tick.get(rtype, 0.0) + consumed
            )

            # If pool was empty, agent loses wealth
            if pool < rate * 0.5:
                agent.wealth = max(0.0, agent.wealth - (rate - consumed) * 0.05)

    def _accumulate_skill(self, agent: Agent, ec: dict) -> None:
        """Grow skill in current occupation, decay others."""
        occupation = agent.occupation or "farmer"
        gain = ec.get("skill_gain_per_tick", 0.01)
        decay = ec.get("skill_decay_rate", 0.002)
        skill_max = ec.get("skill_max", 1.0)

        current = agent.skills.get(occupation, 0.0)
        agent.skills[occupation] = min(skill_max, current + gain)

        for occ in OCCUPATIONS:
            if occ != occupation and occ in agent.skills:
                agent.skills[occ] = max(0.0, agent.skills[occ] - decay)

    # ------------------------------------------------------------------
    # Phase C: Inter-settlement trade
    # ------------------------------------------------------------------
    def _run_inter_settlement_trade(
        self, population: list[Agent], config: ExperimentConfig,
    ) -> None:
        """Trade surplus resources between settlements."""
        from seldon.core.hex_grid import HexGrid

        ec = self._get_config(config)
        dist_cost = ec.get("trade_distance_cost", 0.1)
        trade_base = ec.get("trade_volume_base", 1.0)
        facilitation_income = ec.get("trader_facilitation_income", 0.05)
        agent_map = {a.id: a for a in population if a.is_alive}

        settlement_list = list(self.settlements.values())

        for i, sa in enumerate(settlement_list):
            for sb in settlement_list[i + 1:]:
                dist = float(HexGrid.hex_distance(sa.center, sb.center))
                if dist > 8:
                    continue

                key = (sa.id, sb.id) if sa.id < sb.id else (sb.id, sa.id)
                s_first = sa if sa.id == key[0] else sb
                s_second = sb if sb.id == key[1] else sa

                # Count traders in each settlement
                traders_a = sum(
                    1 for aid in sa.agent_ids
                    if aid in agent_map
                    and (agent_map[aid].occupation or "") == "trader"
                )
                traders_b = sum(
                    1 for aid in sb.agent_ids
                    if aid in agent_map
                    and (agent_map[aid].occupation or "") == "trader"
                )
                avg_traders = max((traders_a + traders_b) / 2, 0.5)

                if key not in self.trade_routes:
                    self.trade_routes[key] = TradeRoute(
                        location_a=key[0], location_b=key[1],
                    )
                route = self.trade_routes[key]
                route.distance = dist
                route.trader_count = traders_a + traders_b
                route.goods_a_to_b = {}
                route.goods_b_to_a = {}

                pop_a = max(len(sa.agent_ids), 1)
                pop_b = max(len(sb.agent_ids), 1)
                total_volume = 0.0

                for rtype in RESOURCE_TYPES:
                    pool_a = s_first.resource_pools.get(rtype, 0.0)
                    pool_b = s_second.resource_pools.get(rtype, 0.0)

                    surplus_a = pool_a - 2.0 * pop_a
                    surplus_b = pool_b - 2.0 * pop_b
                    deficit_a = 0.5 * pop_a - pool_a
                    deficit_b = 0.5 * pop_b - pool_b

                    # A has surplus, B has deficit
                    if surplus_a > 0 and deficit_b > 0:
                        vol = (
                            min(surplus_a, deficit_b)
                            * avg_traders
                            * trade_base
                            * max(0.0, 1.0 - dist * dist_cost)
                        )
                        vol = max(0.0, vol)
                        s_first.resource_pools[rtype] -= vol
                        s_second.resource_pools[rtype] += vol
                        route.goods_a_to_b[rtype] = (
                            route.goods_a_to_b.get(rtype, 0.0) + vol
                        )
                        total_volume += vol

                    # B has surplus, A has deficit
                    if surplus_b > 0 and deficit_a > 0:
                        vol = (
                            min(surplus_b, deficit_a)
                            * avg_traders
                            * trade_base
                            * max(0.0, 1.0 - dist * dist_cost)
                        )
                        vol = max(0.0, vol)
                        s_second.resource_pools[rtype] -= vol
                        s_first.resource_pools[rtype] += vol
                        route.goods_b_to_a[rtype] = (
                            route.goods_b_to_a.get(rtype, 0.0) + vol
                        )
                        total_volume += vol

                route.volume = round(total_volume, 4)

                # Traders gain wealth and trade_history entries
                for aid in sa.agent_ids + sb.agent_ids:
                    agent = agent_map.get(aid)
                    if agent is not None and agent.occupation == "trader":
                        income = facilitation_income * total_volume / max(
                            traders_a + traders_b, 1,
                        )
                        agent.wealth += income
                        agent.trade_history.append({
                            "route": f"{key[0]}->{key[1]}",
                            "volume": round(total_volume, 4),
                            "income": round(income, 4),
                        })

    # ------------------------------------------------------------------
    # Phase C: Governance
    # ------------------------------------------------------------------
    def _apply_governance_all(
        self, population: list[Agent], config: ExperimentConfig,
    ) -> None:
        """Apply governance to all settlements."""
        ec = self._get_config(config)
        agent_map = {a.id: a for a in population if a.is_alive}
        for settlement in self.settlements.values():
            self._apply_governance(settlement, agent_map, ec, config)

    def _apply_governance(
        self, settlement: Settlement, agent_map: dict[str, Any],
        ec: dict, config: ExperimentConfig,
    ) -> None:
        """Find leader, set taxation, distribute communal pool."""
        ts = config.trait_system

        # Find leader: social_role == "leader" with highest social_status
        leader = None
        best_status = -1.0
        for aid in settlement.agent_ids:
            agent = agent_map.get(aid)
            if agent is None:
                continue
            if getattr(agent, 'social_role', None) == "leader":
                status = getattr(agent, 'social_status', 0.0)
                if status > best_status:
                    best_status = status
                    leader = agent

        # Fallback: highest influence_score
        if leader is None:
            best_influence = -1.0
            for aid in settlement.agent_ids:
                agent = agent_map.get(aid)
                if agent is None:
                    continue
                influence = getattr(agent, 'influence_score', 0.0)
                if influence > best_influence:
                    best_influence = influence
                    leader = agent

        settlement.leader_id = leader.id if leader is not None else None

        # Set taxation rate based on leader's conscientiousness
        base_tax = ec.get("base_taxation_rate", 0.10)
        if leader is not None:
            try:
                cons_idx = ts.trait_index("conscientiousness")
                cons_val = float(leader.traits[cons_idx])
            except KeyError:
                cons_val = 0.5
            settlement.taxation_rate = min(0.25, base_tax + cons_val * 0.15)
        else:
            settlement.taxation_rate = base_tax

        # Allocate communal pool (yearly)
        if settlement.communal_pool <= 0:
            return

        poverty_threshold = ec.get("poverty_threshold", 0.2)
        relief_priority = ec.get("poverty_relief_priority", 0.5)

        if leader is not None:
            try:
                agree_idx = ts.trait_index("agreeableness")
                agree_val = float(leader.traits[agree_idx])
            except KeyError:
                agree_val = 0.5
            relief_priority = agree_val * relief_priority
        else:
            relief_priority = relief_priority * 0.5

        relief_budget = settlement.communal_pool * relief_priority
        infra_budget = settlement.communal_pool - relief_budget

        # Distribute relief to poor agents
        poor_agents = [
            agent_map[aid]
            for aid in settlement.agent_ids
            if aid in agent_map and agent_map[aid].wealth < poverty_threshold
        ]
        if poor_agents and relief_budget > 0:
            per_agent = relief_budget / len(poor_agents)
            for agent in poor_agents:
                agent.wealth += per_agent

        # Infrastructure growth
        growth_rate = ec.get("infrastructure_growth_rate", 0.01)
        settlement.infrastructure_level += growth_rate * (
            infra_budget / max(len(settlement.agent_ids), 1)
        )

        settlement.communal_pool = 0.0

    # ------------------------------------------------------------------
    # Legacy generation-level production
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
    # Legacy trade routes
    # ------------------------------------------------------------------
    def _update_trade_routes(
        self, by_location: dict[str, list[Agent]], ec: dict,
        config: ExperimentConfig,
    ) -> None:
        """Evaluate and update trade between locations."""
        locations = list(by_location.keys())
        dist_cost = ec.get("trade_distance_cost", 0.1)
        ts = config.trait_system
        try:
            ext_idx = ts.trait_index("extraversion")
        except KeyError:
            ext_idx = None

        for i, loc_a in enumerate(locations):
            for loc_b in locations[i + 1:]:
                key = (loc_a, loc_b) if loc_a < loc_b else (loc_b, loc_a)

                # Trade volume based on combined extraversion
                agents_a = by_location[loc_a]
                agents_b = by_location[loc_b]
                if ext_idx is not None:
                    ext_a = float(np.mean([float(a.traits[ext_idx]) for a in agents_a])) if agents_a else 0.5
                    ext_b = float(np.mean([float(a.traits[ext_idx]) for a in agents_b])) if agents_b else 0.5
                else:
                    ext_a = 0.5
                    ext_b = 0.5
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

    def _sync_markets_from_settlements(self, ec: dict) -> None:
        """Sync market data from settlement resource pools (hex mode)."""
        elasticity = ec.get("price_elasticity", 0.5)
        self.gdp_by_location = {}

        for settlement in self.settlements.values():
            sid = settlement.id
            pop_size = max(len(settlement.agent_ids), 1)
            total_production = sum(settlement.production_last_tick.values())
            self.gdp_by_location[sid] = round(total_production, 4)

            if sid not in self.markets:
                self.markets[sid] = MarketState(
                    location_id=sid,
                    prices={r: 1.0 for r in RESOURCE_TYPES},
                    supply={r: 0.0 for r in RESOURCE_TYPES},
                    demand={r: 0.0 for r in RESOURCE_TYPES},
                )
            market = self.markets[sid]
            for rtype in RESOURCE_TYPES:
                market.supply[rtype] = settlement.resource_pools.get(rtype, 0.0)
                market.demand[rtype] = pop_size * 0.5
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
        """Wealthy agents stay; poor agents migrate. Settlement-aware."""
        ec = self._get_config(config)
        poverty_threshold = ec.get("poverty_threshold", 0.2)
        if agent.wealth > 1.0 and "stay" in utilities:
            utilities["stay"] = utilities.get("stay", 0.0) + 0.1
        if agent.wealth < poverty_threshold and "migrate" in utilities:
            utilities["migrate"] = utilities.get("migrate", 0.0) + 0.15

        # Phase C: settlement food deficit/surplus awareness
        if self._hex_enabled and agent.community_id:
            settlement = self.settlements.get(agent.community_id)
            if settlement is not None:
                pop = max(len(settlement.agent_ids), 1)
                food_pool = settlement.resource_pools.get("food", 0.0)
                food_per_cap = food_pool / pop
                if food_per_cap < 0.5 and "migrate" in utilities:
                    utilities["migrate"] = utilities.get("migrate", 0.0) + 0.1
                elif food_per_cap > 3.0 and "stay" in utilities:
                    utilities["stay"] = utilities.get("stay", 0.0) + 0.05

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

        result = {
            "total_gdp": round(total_gdp, 4),
            "total_trade_volume": round(total_trade, 4),
            "mean_wealth": round(mean_wealth, 4),
            "gini_coefficient": round(gini, 4),
            "occupation_distribution": occ_counts,
            "trade_route_count": len(self.trade_routes),
            "market_count": len(self.markets),
            "settlement_count": len(self.settlements),
        }

        # Add total resource stocks from settlements
        if self.settlements:
            total_stocks: dict[str, float] = {r: 0.0 for r in RESOURCE_TYPES}
            for s in self.settlements.values():
                for r in RESOURCE_TYPES:
                    total_stocks[r] += s.resource_pools.get(r, 0.0)
            result["total_resource_stocks"] = {
                k: round(v, 4) for k, v in total_stocks.items()
            }

        return result

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
                "goods_a_to_b": dict(r.goods_a_to_b),
                "goods_b_to_a": dict(r.goods_b_to_a),
                "distance": r.distance,
                "trader_count": r.trader_count,
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

    def get_settlements_data(self) -> list[dict[str, Any]]:
        """Return settlement data for API consumption."""
        return [
            {
                "id": s.id,
                "center": list(s.center),
                "tile_count": len(s.tiles),
                "agent_count": len(s.agent_ids),
                "terrain_types": sorted(s.terrain_types),
                "resource_pools": {k: round(v, 4) for k, v in s.resource_pools.items()},
                "production_last_tick": {
                    k: round(v, 4) for k, v in s.production_last_tick.items()
                },
                "consumption_last_tick": {
                    k: round(v, 4) for k, v in s.consumption_last_tick.items()
                },
                "communal_pool": round(s.communal_pool, 4),
                "taxation_rate": round(s.taxation_rate, 4),
                "leader_id": s.leader_id,
                "infrastructure_level": round(s.infrastructure_level, 4),
            }
            for s in self.settlements.values()
        ]

    def get_settlement_detail(self, settlement_id: str) -> dict[str, Any] | None:
        """Return detailed data for a single settlement."""
        s = self.settlements.get(settlement_id)
        if s is None:
            return None
        return {
            "id": s.id,
            "center": list(s.center),
            "tiles": [list(t) for t in s.tiles],
            "agent_ids": list(s.agent_ids),
            "terrain_types": sorted(s.terrain_types),
            "resource_pools": {k: round(v, 4) for k, v in s.resource_pools.items()},
            "production_last_tick": {
                k: round(v, 4) for k, v in s.production_last_tick.items()
            },
            "consumption_last_tick": {
                k: round(v, 4) for k, v in s.consumption_last_tick.items()
            },
            "communal_pool": round(s.communal_pool, 4),
            "taxation_rate": round(s.taxation_rate, 4),
            "leader_id": s.leader_id,
            "infrastructure_level": round(s.infrastructure_level, 4),
        }

    def get_skill_distribution(self, population: list[Agent]) -> dict[str, Any]:
        """Return population skill distribution by occupation."""
        alive = [a for a in population if a.is_alive]
        result: dict[str, dict[str, float]] = {}
        counts: dict[str, int] = {}

        for agent in alive:
            for occ, skill_val in agent.skills.items():
                if occ not in result:
                    result[occ] = {"total": 0.0, "min": float('inf'), "max": 0.0}
                    counts[occ] = 0
                result[occ]["total"] += skill_val
                result[occ]["min"] = min(result[occ]["min"], skill_val)
                result[occ]["max"] = max(result[occ]["max"], skill_val)
                counts[occ] += 1

        summary: dict[str, Any] = {}
        for occ, data in result.items():
            n = counts[occ]
            summary[occ] = {
                "count": n,
                "mean": round(data["total"] / max(n, 1), 4),
                "min": round(data["min"], 4) if data["min"] != float('inf') else 0.0,
                "max": round(data["max"], 4),
            }

        return summary

    def get_production_summary(self) -> dict[str, Any]:
        """Return total production by resource type and settlement."""
        by_settlement: dict[str, dict[str, float]] = {}
        totals: dict[str, float] = {r: 0.0 for r in RESOURCE_TYPES}

        for s in self.settlements.values():
            by_settlement[s.id] = {
                k: round(v, 4) for k, v in s.production_last_tick.items()
            }
            for r in RESOURCE_TYPES:
                totals[r] += s.production_last_tick.get(r, 0.0)

        return {
            "totals": {k: round(v, 4) for k, v in totals.items()},
            "by_settlement": by_settlement,
        }
