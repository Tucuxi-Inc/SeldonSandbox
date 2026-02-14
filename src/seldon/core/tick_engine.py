"""
Tick-based simulation engine (Phase A).

Wraps the existing SimulationEngine to provide monthly tick resolution
while maintaining backward compatibility with generation-indexed data.
Internally processes 12 ticks per year; externally produces one
GenerationSnapshot per year — identical API for the dashboard and tests.

Key design:
- TickEngine delegates to SimulationEngine's subsystems via adapters
- Adapter classes scale annual rates to monthly granularity
- Year boundaries fire extension hooks and build snapshots
- Config toggle ``tick_config.enabled`` selects this engine
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from seldon.core.agent import Agent
from seldon.core.config import ExperimentConfig
from seldon.core.decision import DecisionContext
from seldon.core.engine import GenerationSnapshot, SimulationEngine, _generate_name
from seldon.core.gathering import GatheringSystem
from seldon.core.hex_grid import HexGrid, HexTile, TERRAIN_HABITABILITY
from seldon.core.map_generators import generate_map
from seldon.core.needs import NeedsSystem
from seldon.core.processing import ProcessingRegion


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class LifePhase(str, Enum):
    """Agent life phases based on age in years."""
    INFANT = "infant"           # 0-2 years
    CHILD = "child"             # 3-12 years
    ADOLESCENT = "adolescent"   # 13-17 years
    YOUNG_ADULT = "young_adult" # 18-30 years
    MATURE = "mature"           # 31-55 years
    ELDER = "elder"             # 56+ years


class Season(str, Enum):
    """Seasonal cycle: 3 ticks per season, 4 seasons per year."""
    SPRING = "spring"  # ticks 0,1,2 of year
    SUMMER = "summer"  # ticks 3,4,5
    AUTUMN = "autumn"  # ticks 6,7,8
    WINTER = "winter"  # ticks 9,10,11


# Phase capabilities — what actions agents can perform at each life phase
PHASE_CAPABILITIES: dict[str, frozenset[str]] = {
    "infant": frozenset(["exist"]),
    "child": frozenset(["exist", "learn", "play"]),
    "adolescent": frozenset(["exist", "learn", "gather", "pair_seek"]),
    "young_adult": frozenset(["exist", "learn", "gather", "pair_seek", "reproduce", "lead"]),
    "mature": frozenset(["exist", "learn", "gather", "pair_seek", "reproduce", "lead", "mentor"]),
    "elder": frozenset({"exist", "learn", "gather", "lead", "mentor"}),
}

# Season index for lookup
_SEASON_ORDER = [Season.SPRING, Season.SUMMER, Season.AUTUMN, Season.WINTER]


def _get_season(tick_in_year: int) -> Season:
    """Map a tick within a year (0-11) to its season."""
    return _SEASON_ORDER[tick_in_year // 3]


def classify_life_phase(age_years: int, boundaries: dict[str, list[int]]) -> LifePhase:
    """Determine life phase from age and configurable boundaries."""
    for phase in LifePhase:
        bounds = boundaries.get(phase.value)
        if bounds and bounds[0] <= age_years <= bounds[1]:
            return phase
    # Fallback for very old agents
    return LifePhase.ELDER


# ---------------------------------------------------------------------------
# Tick event accumulator
# ---------------------------------------------------------------------------

@dataclass
class TickEvents:
    """Accumulates events across ticks within a single year."""
    births: int = 0
    deaths: int = 0
    breakthroughs: int = 0
    pairs_formed: int = 0
    dissolutions: int = 0
    infidelity_events: int = 0
    outsiders_injected: int = 0
    memories_created: int = 0
    lore_transmitted: int = 0

    def to_events_dict(self) -> dict[str, Any]:
        return {
            "births": self.births,
            "deaths": self.deaths,
            "breakthroughs": self.breakthroughs,
            "pairs_formed": self.pairs_formed,
            "dissolutions": self.dissolutions,
            "infidelity_events": self.infidelity_events,
            "outsiders_injected": self.outsiders_injected,
            "memories_created": self.memories_created,
            "lore_transmitted": self.lore_transmitted,
        }


# ---------------------------------------------------------------------------
# World-view activity capture (per-agent-per-tick data)
# ---------------------------------------------------------------------------

@dataclass
class AgentTickActivity:
    """What one agent did during one tick — for world-view visualization."""
    agent_id: str
    location: tuple[int, int] | None = None
    previous_location: tuple[int, int] | None = None
    activity: str | None = None            # gathering activity name
    activity_need: str | None = None       # which need it satisfies
    life_phase: str = ""
    processing_region: str = ""
    needs_snapshot: dict[str, float] = field(default_factory=dict)
    health: float = 1.0
    suffering: float = 0.0
    is_pregnant: bool = False


@dataclass
class TickActivityLog:
    """Snapshot of one tick's activity for the entire population."""
    year: int
    tick_in_year: int
    global_tick: int
    season: str
    agent_activities: dict[str, AgentTickActivity] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    population_count: int = 0


# ---------------------------------------------------------------------------
# Adapter classes — scale annual rates to monthly
# ---------------------------------------------------------------------------

class TickDriftAdapter:
    """Scales trait drift operations to tick granularity."""

    def __init__(self, drift_engine, ticks_per_year: int = 12):
        self._drift = drift_engine
        self._tpy = ticks_per_year

    def apply_region_effects_tick(self, agent: Agent) -> np.ndarray:
        """Apply region effects scaled to one tick (1/tpy of annual effect)."""
        region_name = agent.processing_region.value
        effects = self._drift.config.region_effects.get(region_name, {})
        if not effects:
            return agent.traits

        modified = agent.traits.copy()
        ts = self._drift.ts
        for trait_name, modifier in effects.items():
            try:
                idx = ts.trait_index(trait_name)
                modified[idx] += modifier / self._tpy
            except KeyError:
                pass
        return np.clip(modified, 0.0, 1.0)

    def update_suffering_tick(self, agent: Agent) -> None:
        """Update suffering scaled to one tick."""
        suffering_rates = {
            ProcessingRegion.UNDER_PROCESSING: -0.05,
            ProcessingRegion.OPTIMAL: -0.03,
            ProcessingRegion.DEEP: 0.02,
            ProcessingRegion.SACRIFICIAL: 0.08,
            ProcessingRegion.PATHOLOGICAL: 0.12,
        }
        rate = suffering_rates.get(agent.processing_region, 0.0)
        agent.suffering = float(np.clip(
            agent.suffering + rate / self._tpy, 0.0, 1.0,
        ))

    def update_burnout_tick(self, agent: Agent) -> None:
        """Update burnout scaled to one tick."""
        burnout_rates = {
            ProcessingRegion.UNDER_PROCESSING: -0.02,
            ProcessingRegion.OPTIMAL: -0.01,
            ProcessingRegion.DEEP: 0.01,
            ProcessingRegion.SACRIFICIAL: 0.05,
            ProcessingRegion.PATHOLOGICAL: 0.08,
        }
        rate = burnout_rates.get(agent.processing_region, 0.0)
        agent.burnout_level = float(np.clip(
            agent.burnout_level + rate / self._tpy, 0.0, 1.0,
        ))

    def apply_random_drift_yearly(
        self, agent: Agent, rng: np.random.Generator,
    ) -> np.ndarray:
        """Apply random Gaussian trait drift once per year (unchanged)."""
        return self._drift.drift_traits(agent, rng)


class TickRelationshipAdapter:
    """Scales relationship operations to tick granularity."""

    def __init__(self, relationship_manager, ticks_per_year: int = 12):
        self._rm = relationship_manager
        self._tpy = ticks_per_year

    def tick_pairing_opportunity(
        self, population: list[Agent], tick: int,
        generation: int, rng: np.random.Generator,
    ) -> int:
        """Give each eligible agent a 1/tpy chance of entering the pairing pool."""
        # Only a fraction of the population seeks partners each tick
        eligible = [
            a for a in population
            if a.is_eligible_for_pairing(
                self._rm.config.relationship_config.get("pairing_min_age", 16)
            )
            and rng.random() < (1.0 / self._tpy)
        ]
        if len(eligible) < 2:
            return 0

        # Use the relationship manager's pairing logic on the subset
        pairs = self._rm.form_pairs(eligible, generation, rng)
        return len(pairs)


class TickReproductionAdapter:
    """Scales reproduction to tick granularity with pregnancy model."""

    def __init__(self, fertility_manager, ticks_per_year: int = 12):
        self._fm = fertility_manager
        self._tpy = ticks_per_year
        self._pregnancy_duration = 9  # ticks (months)

    def tick_conception_check(
        self, population: list[Agent], tick: int,
        year: int, rng: np.random.Generator,
    ) -> list[tuple[Agent, Agent]]:
        """Check for conceptions this tick. Returns newly pregnant pairs."""
        conceived: list[tuple[Agent, Agent]] = []
        paired = [
            a for a in population
            if a.partner_id is not None and a.is_alive
        ]

        # Build agent lookup for partner finding
        agent_map = {a.id: a for a in population}

        for agent in paired:
            partner = agent_map.get(agent.partner_id)
            if partner is None or not partner.is_alive:
                continue

            # Skip if either is already pregnant
            if agent.extension_data.get("pregnant") or partner.extension_data.get("pregnant"):
                continue

            # Only check from one direction (lower id initiates)
            if agent.id > partner.id:
                continue

            # Monthly conception probability from annual reproduction rate
            if self._fm.will_reproduce(agent, partner, year, rng):
                # Convert annual probability to monthly
                # P(month) ≈ 1 - (1-P(year))^(1/12)
                monthly_prob = 1.0 / self._tpy
                if rng.random() < monthly_prob:
                    # Mark pregnancy
                    agent.extension_data["pregnant"] = True
                    agent.extension_data["conception_tick"] = tick
                    agent.extension_data["partner_at_conception"] = partner.id
                    conceived.append((agent, partner))

        return conceived

    def tick_birth_check(
        self, population: list[Agent], tick: int,
    ) -> list[Agent]:
        """Find agents ready to give birth (9 ticks after conception)."""
        ready: list[Agent] = []
        for agent in population:
            if not agent.extension_data.get("pregnant"):
                continue
            conception_tick = agent.extension_data.get("conception_tick", 0)
            if tick - conception_tick >= self._pregnancy_duration:
                ready.append(agent)
        return ready

    def complete_birth(self, agent: Agent) -> None:
        """Clear pregnancy state after birth."""
        agent.extension_data.pop("pregnant", None)
        agent.extension_data.pop("conception_tick", None)
        agent.extension_data.pop("partner_at_conception", None)


# ---------------------------------------------------------------------------
# TickEngine
# ---------------------------------------------------------------------------

class TickEngine:
    """
    Tick-based simulation engine.

    Wraps SimulationEngine — reuses all its subsystems via adapter classes.
    Processes 12 ticks per year, produces one GenerationSnapshot per year.

    Exposes the same interface as SimulationEngine for SessionManager
    compatibility: population, history, _next_agent_id, rng, extensions,
    _run_generation(), _create_initial_population().
    """

    def __init__(
        self, config: ExperimentConfig,
        extensions=None,
    ):
        self.config = config
        self._tick_config = config.tick_config
        self._tpy = self._tick_config.get("ticks_per_year", 12)
        self._phase_boundaries = self._tick_config.get("life_phase_boundaries", {
            "infant": [0, 2], "child": [3, 12], "adolescent": [13, 17],
            "young_adult": [18, 30], "mature": [31, 55], "elder": [56, 999],
        })
        self._season_effects = self._tick_config.get("season_effects_enabled", True)

        # Wrapped engine — provides all subsystems
        self._engine = SimulationEngine(config, extensions=extensions)

        # Adapters
        self._drift_adapter = TickDriftAdapter(
            self._engine.drift_engine, self._tpy,
        )
        self._rel_adapter = TickRelationshipAdapter(
            self._engine.relationship_manager, self._tpy,
        )
        self._repro_adapter = TickReproductionAdapter(
            self._engine.fertility_manager, self._tpy,
        )

        # Hex grid (Phase B)
        hex_cfg = config.hex_grid_config
        self._hex_enabled = hex_cfg.get("enabled", False)
        self._hex_grid: HexGrid | None = None
        self._vision_range: int = hex_cfg.get("vision_range", 2)
        self._stay_bias: float = hex_cfg.get("stay_bias", 0.3)
        self._starting_hex: tuple[int, int] = tuple(hex_cfg.get("starting_hex", [5, 5]))

        if self._hex_enabled:
            gen_name = hex_cfg.get("map_generator", "california_slice")
            w = hex_cfg.get("width", 20)
            h = hex_cfg.get("height", 10)
            self._hex_grid = generate_map(gen_name, w, h, config.random_seed)

        # Needs + Gathering (Phase A3)
        needs_cfg = config.needs_config
        self._needs_enabled = needs_cfg.get("enabled", True)
        self._needs_system = NeedsSystem(needs_cfg) if self._needs_enabled else None
        self._gathering_system = GatheringSystem() if self._needs_enabled else None

        # Phase C: wire economics extension reference
        self._economics_ext = None
        resources_ext = None
        for ext in self.extensions.get_enabled():
            if ext.name == "economics":
                self._economics_ext = ext
                ext._tick_engine_ref = self
            elif ext.name == "resources":
                resources_ext = ext

        # Wire resources -> economics cross-reference for settlement pool aggregation
        if resources_ext is not None and self._economics_ext is not None:
            resources_ext._economics_ref = self._economics_ext

        # Global tick counter (monotonically increasing across all years)
        self._global_tick = 0

        # World-view activity capture (ring buffer of recent tick logs)
        self._current_tick_log: TickActivityLog | None = None
        self._tick_log_buffer: deque[TickActivityLog] = deque(maxlen=24)

        # State for single-tick stepping mode
        self._single_tick_year: int = 0
        self._single_tick_in_year: int = 0
        self._single_tick_events: TickEvents = TickEvents()
        self._single_tick_contributions: dict[str, float] = {}
        self._single_tick_initialized: bool = False

    # ------------------------------------------------------------------
    # Delegated properties (identical interface to SimulationEngine)
    # ------------------------------------------------------------------

    @property
    def population(self) -> list[Agent]:
        return self._engine.population

    @population.setter
    def population(self, value: list[Agent]) -> None:
        self._engine.population = value

    @property
    def history(self) -> list[GenerationSnapshot]:
        return self._engine.history

    @history.setter
    def history(self, value: list[GenerationSnapshot]) -> None:
        self._engine.history = value

    @property
    def rng(self) -> np.random.Generator:
        return self._engine.rng

    @rng.setter
    def rng(self, value: np.random.Generator) -> None:
        self._engine.rng = value

    @property
    def extensions(self):
        return self._engine.extensions

    @property
    def _next_agent_id(self) -> int:
        return self._engine._next_agent_id

    @_next_agent_id.setter
    def _next_agent_id(self, value: int) -> None:
        self._engine._next_agent_id = value

    @property
    def ts(self):
        return self._engine.ts

    @property
    def hex_grid(self) -> HexGrid | None:
        return self._hex_grid

    @property
    def outsider_interface(self):
        return self._engine.outsider_interface

    @property
    def ripple_tracker(self):
        return self._engine.ripple_tracker

    # ------------------------------------------------------------------
    # Population creation
    # ------------------------------------------------------------------

    def _create_initial_population(self) -> list[Agent]:
        """Create population with tick-specific field initialization."""
        pop = self._engine._create_initial_population()

        for agent in pop:
            # Initialize tick-based age: existing age in years * ticks/year
            agent._age_ticks = int(agent.age) * self._tpy
            agent.life_phase = classify_life_phase(
                int(agent.age), self._phase_boundaries,
            ).value

            # Initialize needs
            if self._needs_system is not None:
                self._needs_system.initialize_needs(agent)

        # Place agents on hex grid
        if self._hex_grid is not None:
            self._place_initial_agents(pop)

        return pop

    def _place_initial_agents(self, pop: list[Agent]) -> None:
        """Place all agents together as a village near starting_hex.

        Agents begin clustered on the starting hex and its immediate
        neighbors to simulate a founding village.  They will spread
        outward organically through tick-level movement decisions.
        """
        grid = self._hex_grid
        start = self._starting_hex

        # Get habitable tiles sorted by distance from starting hex
        habitable = sorted(
            grid.habitable_tiles(),
            key=lambda t: HexGrid.hex_distance(t.coords, start),
        )

        if not habitable:
            return

        # Use the starting hex (or closest habitable) as the village center.
        # Place everyone on the center tile and its immediate neighbors,
        # temporarily exceeding capacity to keep the village tight.
        village_tiles = [
            t for t in habitable
            if HexGrid.hex_distance(t.coords, habitable[0].coords) <= 1
        ]
        if not village_tiles:
            village_tiles = [habitable[0]]

        for i, agent in enumerate(pop):
            tile = village_tiles[i % len(village_tiles)]
            grid.place_agent(tile.q, tile.r, agent.id)
            agent.location = (tile.q, tile.r)
            agent.extension_data["terrain_type"] = tile.terrain_type.value

    # ------------------------------------------------------------------
    # Main entry points
    # ------------------------------------------------------------------

    def run(self, years: int | None = None) -> list[GenerationSnapshot]:
        """Run the simulation for the specified number of years."""
        years = years or self.config.generations_to_run
        self.population = self._create_initial_population()
        self.history = []

        # Extension hook: simulation start
        for ext in self.extensions.get_enabled():
            ext.on_simulation_start(self.population, self.config)

        for year in range(years):
            snapshot = self._run_generation(year)
            self.history.append(snapshot)

        return self.history

    def _run_generation(self, generation: int) -> GenerationSnapshot:
        """Backward-compatible entry point: one generation = one year."""
        return self._run_year(generation)

    # ------------------------------------------------------------------
    # Year processing
    # ------------------------------------------------------------------

    def _run_year(self, year: int) -> GenerationSnapshot:
        """Run 12 ticks for one year, then build a snapshot."""
        tick_events = TickEvents()

        # Extension hook: year start
        for ext in self.extensions.get_enabled():
            ext.on_generation_start(year, self.population, self.config)

        # Per-agent tick-level contribution accumulators
        tick_contributions: dict[str, float] = {}
        for agent in self.population:
            tick_contributions[agent.id] = 0.0

        for tick_in_year in range(self._tpy):
            season = _get_season(tick_in_year)
            self._run_tick(
                year, tick_in_year, season, tick_events, tick_contributions,
            )
            self._global_tick += 1

        # === Year-end processing ===

        # Random trait drift (once per year, unchanged from generation model)
        for agent in self.population:
            if agent.is_alive:
                agent.traits = self._drift_adapter.apply_random_drift_yearly(
                    agent, self.rng,
                )

        # Epigenetic updates (once per year)
        if self._engine.epigenetic_model.enabled:
            for agent in self.population:
                if agent.is_alive:
                    self._engine.epigenetic_model.update_epigenetic_state(
                        agent, self.ts,
                    )
                    self._engine.epigenetic_model.apply_epigenetic_modifiers(
                        agent, self.ts,
                    )

        # Record generation for all living agents
        generation_contributions: list[float] = []
        for agent in self.population:
            if agent.is_alive:
                contribution = tick_contributions.get(agent.id, 0.0)
                agent.record_generation(contribution)
                generation_contributions.append(contribution)

                # Breakthrough check on yearly total
                breakthrough = self._check_breakthrough(agent, contribution)
                if breakthrough:
                    tick_events.breakthroughs += 1
                    if self.config.lore_enabled:
                        mem = self._engine.lore_engine.create_breakthrough_memory(
                            agent.id, year, contribution,
                        )
                        agent.personal_memories.append(mem.to_dict())
                        tick_events.memories_created += 1

                # Suffering memory for R4/R5 agents
                if (self.config.lore_enabled
                        and agent.suffering > 0.6
                        and agent.processing_region in (
                            ProcessingRegion.SACRIFICIAL,
                            ProcessingRegion.PATHOLOGICAL,
                        )):
                    mem = self._engine.lore_engine.create_suffering_memory(
                        agent.id, year, agent.suffering,
                    )
                    agent.personal_memories.append(mem.to_dict())
                    tick_events.memories_created += 1

        # Dissolution checks (once per year)
        dissolved = self._engine.relationship_manager.process_dissolutions(
            self.population, year, self.rng,
        )
        tick_events.dissolutions += len(dissolved)

        # Infidelity checks (once per year)
        infidelity = self._engine.relationship_manager.check_infidelity(
            self.population, year, self.rng,
        )
        tick_events.infidelity_events += len(infidelity)

        # Lore evolution (once per year)
        if self.config.lore_enabled:
            pop_memories = [
                a.personal_memories + a.inherited_lore
                for a in self.population
            ]
            self._engine.lore_engine.evolve_societal_lore(
                pop_memories, year, self.rng,
            )

        # Outsider injection (once per year, generation = year)
        outsiders = self._engine.outsider_interface.process_scheduled_injections(
            year, self.rng,
        )
        for outsider in outsiders:
            # Initialize tick fields on outsiders
            outsider._age_ticks = int(outsider.age) * self._tpy
            outsider.life_phase = classify_life_phase(
                int(outsider.age), self._phase_boundaries,
            ).value
            if self._needs_system is not None:
                self._needs_system.initialize_needs(outsider)

            # Place outsider on hex grid at starting hex or nearest habitable
            if self._hex_grid is not None:
                tile = self._find_habitable_tile_near(self._starting_hex)
                if tile is not None:
                    self._hex_grid.place_agent(tile.q, tile.r, outsider.id)
                    outsider.location = (tile.q, tile.r)
                    outsider.extension_data["terrain_type"] = tile.terrain_type.value

            self.population.append(outsider)
            self._engine.ripple_tracker.track_injection(
                self._engine.outsider_interface.injections[-1]
            )
        tick_events.outsiders_injected += len(outsiders)

        self._engine.ripple_tracker.track_generation(self.population, year)

        # Annual mortality check
        for agent in self.population:
            if not agent.is_alive:
                continue
            # Compute mortality breakdown for cause-of-death tracking
            base = self.config.base_mortality_rate
            age_component = agent.age * self.config.age_mortality_factor
            burnout_component = agent.burnout_level * self.config.burnout_mortality_factor
            mortality_breakdown = {
                "base": round(float(base), 6),
                "age": round(float(age_component), 6),
                "burnout": round(float(burnout_component), 6),
            }
            death_rate = float(np.clip(base + age_component + burnout_component, 0.0, 1.0))

            # Add needs-based mortality
            if self._needs_system is not None:
                needs_factor = self._needs_system.compute_needs_mortality_factor(agent)
                if abs(needs_factor) > 1e-9:
                    mortality_breakdown["needs"] = round(float(needs_factor), 6)
                death_rate += needs_factor

            # Extension modifier hooks
            pre_ext_rate = death_rate
            for ext in self.extensions.get_enabled():
                death_rate = ext.modify_mortality(agent, death_rate, self.config)
            ext_modifier = death_rate - pre_ext_rate
            if abs(ext_modifier) > 1e-9:
                mortality_breakdown["extensions"] = round(float(ext_modifier), 6)

            death_rate = float(np.clip(death_rate, 0.0, 1.0))
            if self.rng.random() < death_rate:
                agent.is_alive = False
                tick_events.deaths += 1
                # Record cause of death
                primary_cause = max(mortality_breakdown, key=lambda k: mortality_breakdown[k])
                agent.extension_data["death_info"] = {
                    "generation": year,
                    "age_at_death": int(agent.age),
                    "mortality_breakdown": mortality_breakdown,
                    "primary_cause": primary_cause,
                    "total_mortality_rate": round(death_rate, 6),
                    "processing_region_at_death": agent.processing_region.value,
                    "suffering_at_death": round(float(agent.suffering), 4),
                    "burnout_at_death": round(float(agent.burnout_level), 4),
                }

                # Remove from hex grid
                if self._hex_grid is not None and agent.location is not None:
                    self._hex_grid.remove_agent(
                        agent.location[0], agent.location[1], agent.id,
                    )

                if agent.partner_id:
                    partner = self._engine._find_agent(agent.partner_id)
                    if partner:
                        partner.partner_id = None
                        partner.relationship_status = "widowed"

        # Remove dead agents
        self.population = [a for a in self.population if a.is_alive]

        # Phase C: re-detect settlements and apply governance (yearly)
        if self._economics_ext is not None:
            self._economics_ext.detect_settlements(
                self, self.population, self.config,
            )
            self._economics_ext._apply_governance_all(
                self.population, self.config,
            )

        # Extension hooks: year end + metrics
        for ext in self.extensions.get_enabled():
            ext.on_generation_end(year, self.population, self.config)

        events = tick_events.to_events_dict()
        for ext in self.extensions.get_enabled():
            ext_metrics = ext.get_metrics(self.population)
            if ext_metrics:
                events[f"ext_{ext.name}"] = ext_metrics

        # Build snapshot (reuse engine's builder)
        return self._engine._build_snapshot(year, events, generation_contributions)

    # ------------------------------------------------------------------
    # Year-end processing (extracted for reuse by _run_single_tick)
    # ------------------------------------------------------------------

    def _complete_year(
        self, year: int, tick_events: TickEvents,
        tick_contributions: dict[str, float],
    ) -> GenerationSnapshot:
        """Year-end processing: drift, epigenetics, mortality, snapshot.

        Extracted from ``_run_year`` so that ``_run_single_tick`` can call
        it at the 12th tick boundary without duplicating logic.
        """
        # Random trait drift (once per year)
        for agent in self.population:
            if agent.is_alive:
                agent.traits = self._drift_adapter.apply_random_drift_yearly(
                    agent, self.rng,
                )

        # Epigenetic updates (once per year)
        if self._engine.epigenetic_model.enabled:
            for agent in self.population:
                if agent.is_alive:
                    self._engine.epigenetic_model.update_epigenetic_state(
                        agent, self.ts,
                    )
                    self._engine.epigenetic_model.apply_epigenetic_modifiers(
                        agent, self.ts,
                    )

        # Record generation for all living agents
        generation_contributions: list[float] = []
        for agent in self.population:
            if agent.is_alive:
                contribution = tick_contributions.get(agent.id, 0.0)
                agent.record_generation(contribution)
                generation_contributions.append(contribution)

                # Breakthrough check on yearly total
                breakthrough = self._check_breakthrough(agent, contribution)
                if breakthrough:
                    tick_events.breakthroughs += 1
                    if self.config.lore_enabled:
                        mem = self._engine.lore_engine.create_breakthrough_memory(
                            agent.id, year, contribution,
                        )
                        agent.personal_memories.append(mem.to_dict())
                        tick_events.memories_created += 1

                # Suffering memory for R4/R5 agents
                if (self.config.lore_enabled
                        and agent.suffering > 0.6
                        and agent.processing_region in (
                            ProcessingRegion.SACRIFICIAL,
                            ProcessingRegion.PATHOLOGICAL,
                        )):
                    mem = self._engine.lore_engine.create_suffering_memory(
                        agent.id, year, agent.suffering,
                    )
                    agent.personal_memories.append(mem.to_dict())
                    tick_events.memories_created += 1

        # Dissolution checks (once per year)
        dissolved = self._engine.relationship_manager.process_dissolutions(
            self.population, year, self.rng,
        )
        tick_events.dissolutions += len(dissolved)

        # Infidelity checks (once per year)
        infidelity = self._engine.relationship_manager.check_infidelity(
            self.population, year, self.rng,
        )
        tick_events.infidelity_events += len(infidelity)

        # Lore evolution (once per year)
        if self.config.lore_enabled:
            pop_memories = [
                a.personal_memories + a.inherited_lore
                for a in self.population
            ]
            self._engine.lore_engine.evolve_societal_lore(
                pop_memories, year, self.rng,
            )

        # Outsider injection (once per year)
        outsiders = self._engine.outsider_interface.process_scheduled_injections(
            year, self.rng,
        )
        for outsider in outsiders:
            outsider._age_ticks = int(outsider.age) * self._tpy
            outsider.life_phase = classify_life_phase(
                int(outsider.age), self._phase_boundaries,
            ).value
            if self._needs_system is not None:
                self._needs_system.initialize_needs(outsider)

            if self._hex_grid is not None:
                tile = self._find_habitable_tile_near(self._starting_hex)
                if tile is not None:
                    self._hex_grid.place_agent(tile.q, tile.r, outsider.id)
                    outsider.location = (tile.q, tile.r)
                    outsider.extension_data["terrain_type"] = tile.terrain_type.value

            self.population.append(outsider)
            self._engine.ripple_tracker.track_injection(
                self._engine.outsider_interface.injections[-1]
            )
        tick_events.outsiders_injected += len(outsiders)

        self._engine.ripple_tracker.track_generation(self.population, year)

        # Annual mortality check
        for agent in self.population:
            if not agent.is_alive:
                continue
            base = self.config.base_mortality_rate
            age_component = agent.age * self.config.age_mortality_factor
            burnout_component = agent.burnout_level * self.config.burnout_mortality_factor
            mortality_breakdown = {
                "base": round(float(base), 6),
                "age": round(float(age_component), 6),
                "burnout": round(float(burnout_component), 6),
            }
            death_rate = float(np.clip(base + age_component + burnout_component, 0.0, 1.0))

            if self._needs_system is not None:
                needs_factor = self._needs_system.compute_needs_mortality_factor(agent)
                if abs(needs_factor) > 1e-9:
                    mortality_breakdown["needs"] = round(float(needs_factor), 6)
                death_rate += needs_factor

            pre_ext_rate = death_rate
            for ext in self.extensions.get_enabled():
                death_rate = ext.modify_mortality(agent, death_rate, self.config)
            ext_modifier = death_rate - pre_ext_rate
            if abs(ext_modifier) > 1e-9:
                mortality_breakdown["extensions"] = round(float(ext_modifier), 6)

            death_rate = float(np.clip(death_rate, 0.0, 1.0))
            if self.rng.random() < death_rate:
                agent.is_alive = False
                tick_events.deaths += 1
                primary_cause = max(mortality_breakdown, key=lambda k: mortality_breakdown[k])
                agent.extension_data["death_info"] = {
                    "generation": year,
                    "age_at_death": int(agent.age),
                    "mortality_breakdown": mortality_breakdown,
                    "primary_cause": primary_cause,
                    "total_mortality_rate": round(death_rate, 6),
                    "processing_region_at_death": agent.processing_region.value,
                    "suffering_at_death": round(float(agent.suffering), 4),
                    "burnout_at_death": round(float(agent.burnout_level), 4),
                }

                if self._hex_grid is not None and agent.location is not None:
                    self._hex_grid.remove_agent(
                        agent.location[0], agent.location[1], agent.id,
                    )

                if agent.partner_id:
                    partner = self._engine._find_agent(agent.partner_id)
                    if partner:
                        partner.partner_id = None
                        partner.relationship_status = "widowed"

                # Record death event in current tick log
                if self._current_tick_log is not None:
                    self._current_tick_log.events.append({
                        "type": "death",
                        "agent_id": agent.id,
                        "agent_name": agent.name,
                        "location": list(agent.location) if agent.location else None,
                        "age": int(agent.age),
                        "primary_cause": primary_cause,
                    })

        # Remove dead agents
        self.population = [a for a in self.population if a.is_alive]

        # Phase C: re-detect settlements and apply governance (yearly)
        if self._economics_ext is not None:
            self._economics_ext.detect_settlements(
                self, self.population, self.config,
            )
            self._economics_ext._apply_governance_all(
                self.population, self.config,
            )

        # Extension hooks: year end + metrics
        for ext in self.extensions.get_enabled():
            ext.on_generation_end(year, self.population, self.config)

        events = tick_events.to_events_dict()
        for ext in self.extensions.get_enabled():
            ext_metrics = ext.get_metrics(self.population)
            if ext_metrics:
                events[f"ext_{ext.name}"] = ext_metrics

        return self._engine._build_snapshot(year, events, generation_contributions)

    # ------------------------------------------------------------------
    # Single-tick stepping (for World View)
    # ------------------------------------------------------------------

    def _run_single_tick(self) -> TickActivityLog:
        """Run exactly one tick. Manages year boundaries.

        Used by the World View to advance the simulation one month at a time
        instead of a full year. Returns the activity log for visualization.
        """
        if not self._single_tick_initialized:
            self._single_tick_year = 0
            self._single_tick_in_year = 0
            self._single_tick_events = TickEvents()
            self._single_tick_contributions = {
                a.id: 0.0 for a in self.population
            }
            self._single_tick_initialized = True

        year = self._single_tick_year
        tick_in_year = self._single_tick_in_year
        season = _get_season(tick_in_year)

        # Extension hook: year start (only on first tick of year)
        if tick_in_year == 0:
            for ext in self.extensions.get_enabled():
                ext.on_generation_start(year, self.population, self.config)

        # Create activity log for this tick
        self._current_tick_log = TickActivityLog(
            year=year,
            tick_in_year=tick_in_year,
            global_tick=self._global_tick,
            season=season.value,
            population_count=sum(1 for a in self.population if a.is_alive),
        )

        # Run the tick
        self._run_tick(
            year, tick_in_year, season,
            self._single_tick_events, self._single_tick_contributions,
        )
        self._global_tick += 1

        # After tick: populate agent activities snapshot
        for agent in self.population:
            if not agent.is_alive:
                continue
            ata = self._current_tick_log.agent_activities.get(agent.id)
            if ata is None:
                ata = AgentTickActivity(agent_id=agent.id)
                self._current_tick_log.agent_activities[agent.id] = ata
            # Fill in final state
            ata.location = tuple(agent.location) if agent.location else None
            ata.life_phase = getattr(agent, "life_phase", "")
            ata.processing_region = agent.processing_region.value
            ata.needs_snapshot = dict(agent.needs) if agent.needs else {}
            ata.health = round(float(getattr(agent, "health", 1.0)), 4)
            ata.suffering = round(float(agent.suffering), 4)
            ata.is_pregnant = bool(agent.extension_data.get("pregnant", False))

        # Update population count after any births/deaths during tick
        self._current_tick_log.population_count = sum(
            1 for a in self.population if a.is_alive
        )

        # Store in ring buffer
        completed_log = self._current_tick_log
        self._tick_log_buffer.append(completed_log)
        self._current_tick_log = None

        # Advance tick counter
        self._single_tick_in_year += 1
        year_complete = False

        if self._single_tick_in_year >= self._tpy:
            # Year complete — run year-end processing
            snapshot = self._complete_year(
                year, self._single_tick_events, self._single_tick_contributions,
            )
            self.history.append(snapshot)

            # Reset for next year
            self._single_tick_in_year = 0
            self._single_tick_year += 1
            self._single_tick_events = TickEvents()
            self._single_tick_contributions = {
                a.id: 0.0 for a in self.population
            }
            year_complete = True

        # Annotate the log with year_complete flag
        completed_log.year_complete = year_complete  # type: ignore[attr-defined]

        return completed_log

    # ------------------------------------------------------------------
    # Per-tick processing
    # ------------------------------------------------------------------

    def _run_tick(
        self, year: int, tick_in_year: int, season: Season,
        events: TickEvents, tick_contributions: dict[str, float],
    ) -> None:
        """Process one monthly tick."""
        config = self.config

        for agent in self.population:
            if not agent.is_alive:
                continue

            # --- Age increment (1/tpy of a year) ---
            agent._age_ticks += 1
            agent.age = agent._age_ticks // self._tpy
            agent.life_phase = classify_life_phase(
                int(agent.age), self._phase_boundaries,
            ).value

            # --- Region effect micro-drift ---
            agent.traits = self._drift_adapter.apply_region_effects_tick(agent)

            # --- Suffering & burnout micro-update ---
            self._drift_adapter.update_suffering_tick(agent)
            self._drift_adapter.update_burnout_tick(agent)

            # --- Micro-contribution accrual ---
            contribution = self._calculate_tick_contribution(agent, season)
            tick_contributions[agent.id] = tick_contributions.get(
                agent.id, 0.0,
            ) + contribution

        # --- Quarterly processing (tick % 3 == 0) ---
        if tick_in_year % 3 == 0:
            for agent in self.population:
                if agent.is_alive:
                    agent.processing_region = self._engine.classifier.classify(agent)
                    agent.dominant_voice = self._engine.council.get_dominant_voice(agent)

        # --- Needs decay + gathering (every tick) ---
        if self._needs_system is not None and self._gathering_system is not None:
            self._process_needs_tick(season)

        # --- Economic tick (Phase C) ---
        if self._economics_ext is not None:
            self._economics_ext.process_economic_tick(
                self.population, self.config, tick_in_year, season.value,
            )

        # --- Movement (every tick when hex grid enabled) ---
        if self._hex_grid is not None:
            self._process_movement_tick(year, tick_in_year, season)

        # --- Pairing opportunity (per tick, probabilistic) ---
        if self._hex_grid is not None:
            # Vision-range limited pairing
            pairs = self._tick_pairing_by_range(tick_in_year, year)
        else:
            pairs = self._rel_adapter.tick_pairing_opportunity(
                self.population, tick_in_year, year, self.rng,
            )
        events.pairs_formed += pairs

        # Record pairing events (pairs is a count, not details — we can't get
        # names here without changing the adapter, so we just record the count)
        # Individual pair events are captured by the adapter if needed in future.

        # --- Conception checks ---
        conceived = self._repro_adapter.tick_conception_check(
            self.population, self._global_tick, year, self.rng,
        )

        # --- Birth checks ---
        ready_to_birth = self._repro_adapter.tick_birth_check(
            self.population, self._global_tick,
        )
        for mother in ready_to_birth:
            partner_id = mother.extension_data.get("partner_at_conception")
            partner = self._engine._find_agent(partner_id)
            if partner is None or not partner.is_alive:
                self._repro_adapter.complete_birth(mother)
                continue

            child = self._engine._create_child(mother, partner, year)
            if child is not None:
                # Initialize tick fields on newborn
                child._age_ticks = 0
                child.life_phase = LifePhase.INFANT.value
                if self._needs_system is not None:
                    self._needs_system.initialize_needs(child)

                # Place child at mother's hex location
                if self._hex_grid is not None and mother.location is not None:
                    mq, mr = mother.location
                    self._hex_grid.place_agent(mq, mr, child.id)
                    child.location = (mq, mr)
                    tile = self._hex_grid.get_tile(mq, mr)
                    if tile is not None:
                        child.extension_data["terrain_type"] = tile.terrain_type.value

                self.population.append(child)
                tick_contributions[child.id] = 0.0
                events.births += 1

                # Record birth event in activity log
                if self._current_tick_log is not None:
                    self._current_tick_log.events.append({
                        "type": "birth",
                        "child_id": child.id,
                        "child_name": child.name,
                        "mother_id": mother.id,
                        "mother_name": mother.name,
                        "father_id": partner.id,
                        "father_name": partner.name,
                        "location": list(child.location) if child.location else None,
                    })

                # Extension hook: agent created
                for ext in self.extensions.get_enabled():
                    ext.on_agent_created(child, (mother, partner), config)

                # Lore transmission
                if config.lore_enabled:
                    lore_count = self._engine._transmit_lore(
                        mother, partner, child,
                    )
                    events.lore_transmitted += lore_count

            self._repro_adapter.complete_birth(mother)

            # Maternal death: _create_child may set mother.is_alive=False
            # via SimulationEngine's maternal mortality check.  Remove
            # the dead mother from the hex grid so she doesn't linger.
            if not mother.is_alive and self._hex_grid is not None and mother.location is not None:
                self._hex_grid.remove_agent(
                    mother.location[0], mother.location[1], mother.id,
                )

    # ------------------------------------------------------------------
    # Needs processing
    # ------------------------------------------------------------------

    def _process_needs_tick(self, season: Season) -> None:
        """Decay needs, gather, share with dependents, assess health."""
        ns = self._needs_system
        gs = self._gathering_system
        ts = self.ts

        # Group by families for caregiver sharing
        agent_map = {a.id: a for a in self.population if a.is_alive}

        for agent in self.population:
            if not agent.is_alive:
                continue

            # Determine terrain (from location_id or default)
            terrain = agent.extension_data.get("terrain_type")

            # Decay needs
            ns.decay_needs(agent, terrain, season.value, agent.life_phase)

            # Gather resources (if capable) — two activities per tick
            if gs.can_gather(agent):
                prioritized = ns.prioritize_needs(agent, ts)
                first_activity = None
                first_need = None
                for _attempt in range(2):
                    activity = gs.choose_activity(
                        agent, prioritized, terrain, season.value, ts,
                    )
                    gathered = gs.attempt_gathering(
                        agent, activity, terrain, season.value, ts, self.rng,
                    )
                    need_name = _activity_satisfies(activity)
                    gs.satisfy_need(agent, need_name, gathered)
                    if first_activity is None:
                        first_activity = activity
                        first_need = need_name
                    # Re-prioritize after first gather
                    prioritized = ns.prioritize_needs(agent, ts)

                # Record primary gathering activity in tick log
                if self._current_tick_log is not None and first_activity:
                    ata = self._current_tick_log.agent_activities.get(agent.id)
                    if ata is None:
                        ata = AgentTickActivity(agent_id=agent.id)
                        self._current_tick_log.agent_activities[agent.id] = ata
                    ata.activity = first_activity
                    ata.activity_need = first_need

        # Caregiver sharing pass
        for agent in self.population:
            if not agent.is_alive:
                continue
            if agent.life_phase in ("infant", "child"):
                continue
            # Find dependents (children of this agent)
            dependents = [
                agent_map[cid] for cid in agent.children_ids
                if cid in agent_map
                and agent_map[cid].is_alive
                and agent_map[cid].life_phase in ("infant", "child")
            ]
            if dependents:
                gs.caregiver_share(agent, dependents, self.rng)

        # Assess health impact
        for agent in self.population:
            if not agent.is_alive:
                continue
            health_delta, suffering_delta = ns.assess_health_impact(agent)
            ns.apply_health_impact(agent, health_delta, suffering_delta)

    # ------------------------------------------------------------------
    # Contribution
    # ------------------------------------------------------------------

    def _calculate_tick_contribution(
        self, agent: Agent, season: Season,
    ) -> float:
        """Calculate micro-contribution for one tick."""
        cc = self.config.contribution_config
        region_mult = cc["region_multipliers"].get(
            agent.processing_region.value, 0.5,
        )

        ts = self.ts
        creativity = agent.traits[ts.trait_index("creativity")]
        resilience = agent.traits[ts.trait_index("resilience")]
        conscientiousness = agent.traits[ts.trait_index("conscientiousness")]

        trait_contribution = (
            creativity * cc["creativity_weight"]
            + resilience * cc["resilience_weight"]
            + conscientiousness * cc["conscientiousness_weight"]
        )

        contribution = cc["base_contribution"] * region_mult * (0.5 + trait_contribution)

        # Scale to monthly
        contribution /= self._tpy

        # Season modifier
        if self._season_effects:
            modifiers = self._tick_config.get("season_modifiers", {})
            season_mod = modifiers.get(season.value, {})
            contribution *= season_mod.get("contribution_mult", 1.0)

        return float(np.clip(contribution, 0.0, 2.0))

    def _check_breakthrough(self, agent: Agent, yearly_contribution: float) -> bool:
        """Check for breakthrough on yearly contribution total."""
        cc = self.config.contribution_config
        if agent.processing_region not in (
            ProcessingRegion.DEEP, ProcessingRegion.SACRIFICIAL,
        ):
            return False
        if yearly_contribution > cc["breakthrough_threshold"]:
            if self.rng.random() < cc["breakthrough_base_probability"]:
                return True
        return False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_agent(self, agent_id: str | None) -> Agent | None:
        return self._engine._find_agent(agent_id)

    def _new_id(self) -> str:
        return self._engine._new_id()

    def _find_habitable_tile_near(
        self, center: tuple[int, int],
    ) -> HexTile | None:
        """Find the nearest habitable tile to center, preferring less crowded tiles."""
        grid = self._hex_grid
        if grid is None:
            return None

        # Try the center tile first
        center_tile = grid.get_tile(*center)
        if center_tile is not None and center_tile.is_habitable:
            if center_tile.capacity <= 0 or len(center_tile.current_agents) < center_tile.capacity:
                return center_tile

        # Expand outward
        habitable = sorted(
            grid.habitable_tiles(),
            key=lambda t: (HexGrid.hex_distance(t.coords, center), len(t.current_agents)),
        )
        for tile in habitable:
            if tile.capacity <= 0 or len(tile.current_agents) < tile.capacity:
                return tile

        # All at capacity — return nearest
        return habitable[0] if habitable else None

    # ------------------------------------------------------------------
    # Movement system (Phase B)
    # ------------------------------------------------------------------

    def _process_movement_tick(
        self, year: int, tick_in_year: int, season: Season,
    ) -> None:
        """Process agent movement decisions for one tick."""
        grid = self._hex_grid
        if grid is None:
            return

        ts = self.ts
        agent_map = {a.id: a for a in self.population if a.is_alive}
        decision_model = self._engine.decision_model

        # Identify dependents (infants/children who move with caregivers)
        dependent_ids: set[str] = set()
        for agent in self.population:
            if agent.is_alive and agent.life_phase in ("infant", "child"):
                dependent_ids.add(agent.id)

        for agent in self.population:
            if not agent.is_alive:
                continue
            if agent.id in dependent_ids:
                continue  # Dependents move with caregivers
            if agent.location is None:
                continue

            q, r = agent.location
            current_tile = grid.get_tile(q, r)
            if current_tile is None:
                continue

            # Get passable neighbors
            neighbors = grid.passable_neighbors(q, r)
            if not neighbors:
                continue

            # Build actions: stay + move to each neighbor
            actions = ["stay"]
            action_tiles: dict[str, HexTile] = {"stay": current_tile}
            for nb in neighbors:
                action_key = f"move_{nb.q}_{nb.r}"
                actions.append(action_key)
                action_tiles[action_key] = nb

            # Build situation vector
            situation = self._build_movement_situation(
                agent, current_tile, neighbors, agent_map, ts,
            )

            # Build per-action weight vectors and biases
            action_weights, action_biases = self._build_movement_weights(
                agent, current_tile, action_tiles, ts,
            )

            # Decide via DecisionModel
            result = decision_model.decide(
                agent, DecisionContext.MIGRATION, situation,
                actions, action_weights, action_biases, self.rng,
            )

            chosen = result.chosen_action

            # Record movement in activity log
            if self._current_tick_log is not None:
                prev_loc = tuple(agent.location) if agent.location else None
                ata = self._current_tick_log.agent_activities.get(agent.id)
                if ata is None:
                    ata = AgentTickActivity(agent_id=agent.id)
                    self._current_tick_log.agent_activities[agent.id] = ata
                if chosen != "stay":
                    ata.previous_location = prev_loc

            if chosen != "stay":
                dest_tile = action_tiles[chosen]
                self._move_agent_to(agent, dest_tile, agent_map, dependent_ids)

    def _build_movement_situation(
        self, agent: Agent, current_tile: HexTile,
        neighbors: list[HexTile], agent_map: dict[str, Agent],
        ts: Any,
    ) -> dict[str, float]:
        """Build the situation vector for a movement decision."""
        # Need urgency — average of lowest needs
        needs = agent.needs
        need_vals = list(needs.values()) if needs else [1.0]
        avg_need = float(np.mean(need_vals))
        min_need = float(min(need_vals)) if need_vals else 1.0

        # Overcrowding at current location
        overcrowding = 0.0
        if current_tile.capacity > 0:
            overcrowding = max(0.0, len(current_tile.current_agents) / current_tile.capacity - 0.8)

        # Social pull — is partner on a different tile?
        social_pull = 0.0
        if agent.partner_id and agent.partner_id in agent_map:
            partner = agent_map[agent.partner_id]
            if partner.location is not None and partner.location != agent.location:
                social_pull = 0.5

        # Best neighbor habitability
        best_neighbor_hab = max(
            (nb.habitability for nb in neighbors), default=0.0,
        )

        return {
            "need_urgency": 1.0 - avg_need,
            "min_need": min_need,
            "overcrowding": overcrowding,
            "social_pull": social_pull,
            "current_habitability": current_tile.habitability,
            "best_neighbor_habitability": best_neighbor_hab,
        }

    def _build_movement_weights(
        self, agent: Agent, current_tile: HexTile,
        action_tiles: dict[str, HexTile], ts: Any,
    ) -> tuple[dict[str, np.ndarray], dict[str, float]]:
        """Build per-action trait weight vectors and biases for movement."""
        n_traits = ts.count
        weights: dict[str, np.ndarray] = {}
        biases: dict[str, float] = {}

        # Trait indices (with fallbacks)
        def _idx(name: str) -> int | None:
            try:
                return ts.trait_index(name)
            except KeyError:
                return None

        cons_idx = _idx("conscientiousness")
        risk_idx = _idx("risk_taking")
        adapt_idx = _idx("adaptability")
        ambi_idx = _idx("ambition")
        open_idx = _idx("openness")

        for action, tile in action_tiles.items():
            w = np.zeros(n_traits)
            if action == "stay":
                # Stay biased by conscientiousness (routine), penalized by risk_taking
                if cons_idx is not None:
                    w[cons_idx] = 0.3
                if risk_idx is not None:
                    w[risk_idx] = -0.1
                biases[action] = self._stay_bias
            else:
                # Move biased by risk_taking, adaptability, ambition, openness
                if risk_idx is not None:
                    w[risk_idx] = 0.2
                if adapt_idx is not None:
                    w[adapt_idx] = 0.15
                if ambi_idx is not None:
                    w[ambi_idx] = 0.1
                if open_idx is not None:
                    w[open_idx] = 0.1

                # Bias from destination habitability minus overcrowding penalty
                hab_bonus = tile.habitability * 0.3
                crowd_penalty = 0.0
                if tile.capacity > 0:
                    crowd_penalty = max(0.0, len(tile.current_agents) / tile.capacity - 0.8) * 0.2
                biases[action] = hab_bonus - crowd_penalty

            weights[action] = w

        return weights, biases

    def _move_agent_to(
        self, agent: Agent, dest_tile: HexTile,
        agent_map: dict[str, Agent], dependent_ids: set[str],
    ) -> None:
        """Execute agent move + dependent follow."""
        grid = self._hex_grid
        if grid is None or agent.location is None:
            return

        old_q, old_r = agent.location
        grid.move_agent(old_q, old_r, dest_tile.q, dest_tile.r, agent.id)
        agent.location = (dest_tile.q, dest_tile.r)
        agent.extension_data["terrain_type"] = dest_tile.terrain_type.value

        # Move dependents (infant/child children)
        for cid in agent.children_ids:
            if cid not in dependent_ids or cid not in agent_map:
                continue
            child = agent_map[cid]
            if not child.is_alive or child.location is None:
                continue
            cq, cr = child.location
            grid.move_agent(cq, cr, dest_tile.q, dest_tile.r, child.id)
            child.location = (dest_tile.q, dest_tile.r)
            child.extension_data["terrain_type"] = dest_tile.terrain_type.value

    # ------------------------------------------------------------------
    # Vision-range interaction filtering (Phase B)
    # ------------------------------------------------------------------

    def _group_by_interaction_range(self) -> list[list[Agent]]:
        """Group agents into clusters where members share hexes within vision_range."""
        grid = self._hex_grid
        if grid is None:
            return [list(self.population)]

        # Build location → agents mapping
        loc_agents: dict[tuple[int, int], list[Agent]] = {}
        for agent in self.population:
            if not agent.is_alive or agent.location is None:
                continue
            loc = agent.location
            if loc not in loc_agents:
                loc_agents[loc] = []
            loc_agents[loc].append(agent)

        if not loc_agents:
            return []

        # Union-find on locations
        locations = list(loc_agents.keys())
        parent: dict[tuple[int, int], tuple[int, int]] = {loc: loc for loc in locations}

        def find(x: tuple[int, int]) -> tuple[int, int]:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: tuple[int, int], b: tuple[int, int]) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for i, loc_a in enumerate(locations):
            for loc_b in locations[i + 1:]:
                if HexGrid.hex_distance(loc_a, loc_b) <= self._vision_range:
                    union(loc_a, loc_b)

        # Group by root
        groups: dict[tuple[int, int], list[Agent]] = {}
        for loc in locations:
            root = find(loc)
            if root not in groups:
                groups[root] = []
            groups[root].extend(loc_agents[loc])

        return list(groups.values())

    def _tick_pairing_by_range(
        self, tick_in_year: int, year: int,
    ) -> int:
        """Run pairing per interaction cluster instead of whole population."""
        clusters = self._group_by_interaction_range()
        total_pairs = 0
        for cluster in clusters:
            pairs = self._rel_adapter.tick_pairing_opportunity(
                cluster, tick_in_year, year, self.rng,
            )
            total_pairs += pairs
        return total_pairs

    # ------------------------------------------------------------------
    # Cluster detection (Phase B)
    # ------------------------------------------------------------------

    def get_agent_clusters(self, min_size: int = 3) -> list[dict[str, Any]]:
        """Find clusters of occupied adjacent tiles for settlement emergence.

        Returns list of cluster dicts with keys:
            tiles: list of (q, r) coordinates
            agent_count: total agents in cluster
            agent_ids: list of agent IDs
            center: (q, r) approximate center
            terrain_types: set of terrain type strings
        """
        grid = self._hex_grid
        if grid is None:
            return []

        # Find occupied tiles
        occupied = {
            coord: tile for coord, tile in grid.tiles.items()
            if tile.current_agents
        }
        if not occupied:
            return []

        # Union-find on adjacent occupied tiles
        parent: dict[tuple[int, int], tuple[int, int]] = {c: c for c in occupied}

        def find(x: tuple[int, int]) -> tuple[int, int]:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: tuple[int, int], b: tuple[int, int]) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for coord in occupied:
            for nb_coord in grid.neighbors(*coord):
                if nb_coord in occupied:
                    union(coord, nb_coord)

        # Group by root
        groups: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for coord in occupied:
            root = find(coord)
            if root not in groups:
                groups[root] = []
            groups[root].append(coord)

        # Build result, filtering by min_size
        result: list[dict[str, Any]] = []
        for tiles_coords in groups.values():
            agent_ids: list[str] = []
            terrain_types: set[str] = set()
            for coord in tiles_coords:
                tile = occupied[coord]
                agent_ids.extend(tile.current_agents)
                terrain_types.add(tile.terrain_type.value)

            if len(agent_ids) < min_size:
                continue

            # Approximate center
            avg_q = sum(c[0] for c in tiles_coords) / len(tiles_coords)
            avg_r = sum(c[1] for c in tiles_coords) / len(tiles_coords)

            result.append({
                "tiles": tiles_coords,
                "agent_count": len(agent_ids),
                "agent_ids": agent_ids,
                "center": (round(avg_q), round(avg_r)),
                "terrain_types": terrain_types,
            })

        return sorted(result, key=lambda c: c["agent_count"], reverse=True)


def _activity_satisfies(activity: str) -> str:
    """Get which need an activity satisfies."""
    from seldon.core.gathering import GATHERING_ACTIVITIES
    return GATHERING_ACTIVITIES.get(activity, {}).get("satisfies", "hunger")
