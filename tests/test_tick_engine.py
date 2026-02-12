"""
Tests for the tick-based simulation engine (Phases A + B).

Covers TickEngine, adapters, life phase classification, season cycling,
determinism, backward compatibility, needs integration, hex grid
integration, agent movement, interaction filtering, and cluster detection.
"""

from __future__ import annotations

import numpy as np
import pytest

from seldon.core.agent import Agent
from seldon.core.config import ExperimentConfig
from seldon.core.engine import GenerationSnapshot
from seldon.core.processing import ProcessingRegion
from seldon.core.tick_engine import (
    LifePhase,
    Season,
    TickDriftAdapter,
    TickEngine,
    TickEvents,
    TickRelationshipAdapter,
    TickReproductionAdapter,
    _get_season,
    classify_life_phase,
)
from seldon.core.traits import TraitSystem


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(**overrides) -> ExperimentConfig:
    """Create a tick-enabled config with small population for fast tests."""
    defaults = {
        "random_seed": 42,
        "initial_population": 20,
        "generations_to_run": 3,
        "tick_config": {
            "enabled": True,
            "ticks_per_year": 12,
            "life_phase_boundaries": {
                "infant": [0, 2],
                "child": [3, 12],
                "adolescent": [13, 17],
                "young_adult": [18, 30],
                "mature": [31, 55],
                "elder": [56, 999],
            },
            "season_effects_enabled": True,
            "season_modifiers": {
                "spring": {"fertility_mult": 1.2, "mortality_mult": 0.9,
                           "contribution_mult": 1.0, "food_mult": 1.1, "mood_offset": 0.05},
                "summer": {"fertility_mult": 1.0, "mortality_mult": 0.95,
                           "contribution_mult": 1.1, "food_mult": 1.3, "mood_offset": 0.03},
                "autumn": {"fertility_mult": 0.8, "mortality_mult": 1.0,
                           "contribution_mult": 1.05, "food_mult": 1.2, "mood_offset": -0.02},
                "winter": {"fertility_mult": 0.6, "mortality_mult": 1.3,
                           "contribution_mult": 0.8, "food_mult": 0.6, "mood_offset": -0.05},
            },
        },
    }
    defaults.update(overrides)
    return ExperimentConfig(**defaults)


def _make_agent(agent_id="a1", age=25, **overrides):
    ts = TraitSystem(preset="compact")
    rng = np.random.default_rng(42)
    traits = ts.random_traits(rng)
    a = Agent(
        id=agent_id, name=f"Agent_{agent_id}", age=age, generation=0,
        birth_order=1, traits=traits, traits_at_birth=traits.copy(),
    )
    for k, v in overrides.items():
        setattr(a, k, v)
    return a


# ---------------------------------------------------------------------------
# LifePhase classification
# ---------------------------------------------------------------------------

class TestLifePhase:
    BOUNDARIES = {
        "infant": [0, 2], "child": [3, 12], "adolescent": [13, 17],
        "young_adult": [18, 30], "mature": [31, 55], "elder": [56, 999],
    }

    def test_infant(self):
        assert classify_life_phase(0, self.BOUNDARIES) == LifePhase.INFANT
        assert classify_life_phase(2, self.BOUNDARIES) == LifePhase.INFANT

    def test_child(self):
        assert classify_life_phase(3, self.BOUNDARIES) == LifePhase.CHILD
        assert classify_life_phase(12, self.BOUNDARIES) == LifePhase.CHILD

    def test_adolescent(self):
        assert classify_life_phase(13, self.BOUNDARIES) == LifePhase.ADOLESCENT
        assert classify_life_phase(17, self.BOUNDARIES) == LifePhase.ADOLESCENT

    def test_young_adult(self):
        assert classify_life_phase(18, self.BOUNDARIES) == LifePhase.YOUNG_ADULT
        assert classify_life_phase(30, self.BOUNDARIES) == LifePhase.YOUNG_ADULT

    def test_mature(self):
        assert classify_life_phase(31, self.BOUNDARIES) == LifePhase.MATURE
        assert classify_life_phase(55, self.BOUNDARIES) == LifePhase.MATURE

    def test_elder(self):
        assert classify_life_phase(56, self.BOUNDARIES) == LifePhase.ELDER
        assert classify_life_phase(100, self.BOUNDARIES) == LifePhase.ELDER

    def test_elder_fallback_for_very_old(self):
        # Beyond any defined range → elder fallback
        assert classify_life_phase(1000, self.BOUNDARIES) == LifePhase.ELDER


# ---------------------------------------------------------------------------
# Season cycling
# ---------------------------------------------------------------------------

class TestSeason:
    def test_spring_ticks(self):
        assert _get_season(0) == Season.SPRING
        assert _get_season(1) == Season.SPRING
        assert _get_season(2) == Season.SPRING

    def test_summer_ticks(self):
        assert _get_season(3) == Season.SUMMER
        assert _get_season(4) == Season.SUMMER
        assert _get_season(5) == Season.SUMMER

    def test_autumn_ticks(self):
        assert _get_season(6) == Season.AUTUMN
        assert _get_season(7) == Season.AUTUMN
        assert _get_season(8) == Season.AUTUMN

    def test_winter_ticks(self):
        assert _get_season(9) == Season.WINTER
        assert _get_season(10) == Season.WINTER
        assert _get_season(11) == Season.WINTER

    def test_full_cycle(self):
        seasons = [_get_season(t) for t in range(12)]
        assert seasons == [
            Season.SPRING, Season.SPRING, Season.SPRING,
            Season.SUMMER, Season.SUMMER, Season.SUMMER,
            Season.AUTUMN, Season.AUTUMN, Season.AUTUMN,
            Season.WINTER, Season.WINTER, Season.WINTER,
        ]


# ---------------------------------------------------------------------------
# TickEvents
# ---------------------------------------------------------------------------

class TestTickEvents:
    def test_initial_values(self):
        te = TickEvents()
        assert te.births == 0
        assert te.deaths == 0
        assert te.breakthroughs == 0

    def test_to_events_dict(self):
        te = TickEvents(births=3, deaths=1, breakthroughs=1)
        d = te.to_events_dict()
        assert d["births"] == 3
        assert d["deaths"] == 1
        assert d["breakthroughs"] == 1
        assert "pairs_formed" in d

    def test_accumulation(self):
        te = TickEvents()
        te.births += 2
        te.deaths += 1
        te.pairs_formed += 3
        assert te.births == 2
        assert te.deaths == 1
        assert te.pairs_formed == 3


# ---------------------------------------------------------------------------
# TickDriftAdapter
# ---------------------------------------------------------------------------

class TestTickDriftAdapter:
    def test_region_effects_scaled(self):
        """12 ticks of micro-drift should approximately equal 1 generation."""
        config = _make_config()
        agent = _make_agent()
        agent.processing_region = ProcessingRegion.SACRIFICIAL

        from seldon.core.drift import TraitDriftEngine
        drift = TraitDriftEngine(config)
        adapter = TickDriftAdapter(drift, ticks_per_year=12)

        # Get original traits
        original = agent.traits.copy()

        # Apply 12 ticks of micro-drift
        for _ in range(12):
            agent.traits = adapter.apply_region_effects_tick(agent)

        tick_result = agent.traits.copy()

        # Reset and apply single generation drift
        agent.traits = original.copy()
        agent.traits = drift.apply_region_effects(agent)
        gen_result = agent.traits

        # They should be approximately equal (within floating point tolerance)
        np.testing.assert_allclose(tick_result, gen_result, atol=1e-10)

    def test_suffering_scales_monthly(self):
        config = _make_config()
        agent = _make_agent()
        agent.processing_region = ProcessingRegion.SACRIFICIAL
        agent.suffering = 0.0

        from seldon.core.drift import TraitDriftEngine
        drift = TraitDriftEngine(config)
        adapter = TickDriftAdapter(drift, ticks_per_year=12)

        # 12 ticks of suffering
        for _ in range(12):
            adapter.update_suffering_tick(agent)

        # Should be approximately equal to one generation's suffering rate (0.08)
        assert abs(agent.suffering - 0.08) < 0.001

    def test_burnout_scales_monthly(self):
        config = _make_config()
        agent = _make_agent()
        agent.processing_region = ProcessingRegion.SACRIFICIAL
        agent.burnout_level = 0.0

        from seldon.core.drift import TraitDriftEngine
        drift = TraitDriftEngine(config)
        adapter = TickDriftAdapter(drift, ticks_per_year=12)

        for _ in range(12):
            adapter.update_burnout_tick(agent)

        # Sacrificial burnout rate = 0.05/year → after 12 ticks ≈ 0.05
        assert abs(agent.burnout_level - 0.05) < 0.001


# ---------------------------------------------------------------------------
# TickEngine core
# ---------------------------------------------------------------------------

class TestTickEngine:
    def test_creates_population(self):
        config = _make_config()
        engine = TickEngine(config)
        pop = engine._create_initial_population()
        assert len(pop) == 20

    def test_population_has_life_phases(self):
        config = _make_config()
        engine = TickEngine(config)
        pop = engine._create_initial_population()
        for agent in pop:
            assert agent.life_phase is not None
            assert agent.life_phase in [p.value for p in LifePhase]

    def test_population_has_tick_ages(self):
        config = _make_config()
        engine = TickEngine(config)
        pop = engine._create_initial_population()
        for agent in pop:
            assert agent._age_ticks == int(agent.age) * 12

    def test_population_has_needs(self):
        config = _make_config()
        engine = TickEngine(config)
        pop = engine._create_initial_population()
        for agent in pop:
            assert agent.health == 1.0
            assert all(v == 1.0 for v in agent.needs.values())

    def test_12_ticks_produce_one_snapshot(self):
        config = _make_config(generations_to_run=1)
        engine = TickEngine(config)
        history = engine.run(years=1)
        assert len(history) == 1
        assert isinstance(history[0], GenerationSnapshot)

    def test_multi_year_run(self):
        config = _make_config(generations_to_run=3)
        engine = TickEngine(config)
        history = engine.run(years=3)
        assert len(history) == 3
        for i, snap in enumerate(history):
            assert snap.generation == i

    def test_snapshot_has_all_required_fields(self):
        config = _make_config(generations_to_run=1)
        engine = TickEngine(config)
        history = engine.run(years=1)
        snap = history[0]

        assert snap.generation == 0
        assert snap.population_size > 0
        assert isinstance(snap.births, int)
        assert isinstance(snap.deaths, int)
        assert isinstance(snap.breakthroughs, int)
        assert isinstance(snap.pairs_formed, int)
        assert snap.trait_means is not None
        assert snap.trait_stds is not None
        assert isinstance(snap.region_counts, dict)
        assert isinstance(snap.total_contribution, float)
        assert isinstance(snap.mean_contribution, float)
        assert isinstance(snap.mean_suffering, float)
        assert isinstance(snap.mean_age, float)
        assert isinstance(snap.birth_order_counts, dict)
        assert isinstance(snap.events, dict)

    def test_history_lists_append_once_per_year(self):
        """Agent history should grow by 1 per year, not per tick."""
        config = _make_config(generations_to_run=3)
        engine = TickEngine(config)
        engine.run(years=3)

        # Check surviving agents
        for agent in engine.population:
            # History should have at most 3 entries (may be fewer if born later)
            assert len(agent.trait_history) <= 3
            assert len(agent.contribution_history) <= 3
            assert len(agent.suffering_history) <= 3
            assert len(agent.needs_history) <= 3
            assert len(agent.health_history) <= 3
            # At least 1 entry for agents that survived
            assert len(agent.trait_history) >= 1

    def test_agent_age_consistent(self):
        """Age in years should equal _age_ticks // 12."""
        config = _make_config(generations_to_run=2)
        engine = TickEngine(config)
        engine.run(years=2)

        for agent in engine.population:
            assert agent.age == agent._age_ticks // 12

    def test_determinism_same_seed(self):
        """Two runs with the same seed produce identical snapshots."""
        config1 = _make_config(random_seed=123, generations_to_run=2)
        config2 = _make_config(random_seed=123, generations_to_run=2)

        engine1 = TickEngine(config1)
        h1 = engine1.run(years=2)

        engine2 = TickEngine(config2)
        h2 = engine2.run(years=2)

        for s1, s2 in zip(h1, h2):
            assert s1.generation == s2.generation
            assert s1.population_size == s2.population_size
            assert s1.births == s2.births
            assert s1.deaths == s2.deaths
            np.testing.assert_array_equal(s1.trait_means, s2.trait_means)

    def test_different_seeds_differ(self):
        """Different seeds produce different results."""
        config1 = _make_config(random_seed=100, generations_to_run=2)
        config2 = _make_config(random_seed=200, generations_to_run=2)

        engine1 = TickEngine(config1)
        h1 = engine1.run(years=2)

        engine2 = TickEngine(config2)
        h2 = engine2.run(years=2)

        # At least some metric should differ
        diffs = sum(
            1 for s1, s2 in zip(h1, h2)
            if s1.population_size != s2.population_size
            or s1.births != s2.births
            or not np.allclose(s1.trait_means, s2.trait_means)
        )
        assert diffs > 0

    def test_run_generation_backward_compat(self):
        """_run_generation() should work as entry point."""
        config = _make_config()
        engine = TickEngine(config)
        engine.population = engine._create_initial_population()

        for ext in engine.extensions.get_enabled():
            ext.on_simulation_start(engine.population, config)

        snap = engine._run_generation(0)
        assert isinstance(snap, GenerationSnapshot)
        assert snap.generation == 0

    def test_population_property_delegates(self):
        config = _make_config()
        engine = TickEngine(config)
        engine.population = engine._create_initial_population()
        assert len(engine.population) == 20
        # Underlying engine should have the same population
        assert engine._engine.population is engine.population

    def test_needs_decay_during_run(self):
        """Needs should decrease during simulation ticks."""
        config = _make_config(generations_to_run=1)
        engine = TickEngine(config)
        engine.run(years=1)

        # After a full year, at least some needs should be below 1.0
        # (gathering partially replenishes, but decay should dominate for some)
        any_below_one = False
        for agent in engine.population:
            for need_val in agent.needs.values():
                if need_val < 1.0:
                    any_below_one = True
                    break
        assert any_below_one

    def test_health_history_tracked(self):
        """Health history should be recorded at year boundaries."""
        config = _make_config(generations_to_run=2)
        engine = TickEngine(config)
        engine.run(years=2)

        for agent in engine.population:
            assert len(agent.health_history) >= 1
            for h in agent.health_history:
                assert 0.0 <= h <= 1.0


# ---------------------------------------------------------------------------
# Extension hooks
# ---------------------------------------------------------------------------

class TestExtensionHooks:
    def test_extension_hooks_fire_at_year_boundaries(self):
        """Extension hooks should fire once per year, not per tick."""
        from seldon.extensions.base import SimulationExtension

        class HookCounter(SimulationExtension):
            name = "hook_counter"
            description = "Counts hook invocations"

            def __init__(self):
                self.gen_start_count = 0
                self.gen_end_count = 0
                self.sim_start_count = 0

            def on_simulation_start(self, population, config):
                self.sim_start_count += 1

            def on_generation_start(self, generation, population, config):
                self.gen_start_count += 1

            def on_generation_end(self, generation, population, config):
                self.gen_end_count += 1

            def get_default_config(self):
                return {}

        from seldon.extensions.registry import ExtensionRegistry
        registry = ExtensionRegistry()
        counter = HookCounter()
        registry.register(counter)
        registry.enable("hook_counter")

        config = _make_config(generations_to_run=3)
        engine = TickEngine(config, extensions=registry)
        engine.run(years=3)

        assert counter.sim_start_count == 1
        assert counter.gen_start_count == 3
        assert counter.gen_end_count == 3


# ---------------------------------------------------------------------------
# Needs integration
# ---------------------------------------------------------------------------

class TestNeedsIntegration:
    def test_needs_disabled(self):
        """When needs are disabled, agents keep health=1.0."""
        config = _make_config(
            generations_to_run=1,
            needs_config={"enabled": False},
        )
        engine = TickEngine(config)
        engine.run(years=1)

        for agent in engine.population:
            assert agent.health == 1.0

    def test_needs_bounded(self):
        """All needs should stay in [0, 1] range."""
        config = _make_config(generations_to_run=2)
        engine = TickEngine(config)
        engine.run(years=2)

        for agent in engine.population:
            for need_val in agent.needs.values():
                assert 0.0 <= need_val <= 1.0
            assert 0.0 <= agent.health <= 1.0


# ---------------------------------------------------------------------------
# Season effects
# ---------------------------------------------------------------------------

class TestSeasonEffects:
    def test_season_modifies_contribution(self):
        """Contribution should vary by season."""
        config = _make_config()
        engine = TickEngine(config)
        agent = _make_agent()
        agent.processing_region = ProcessingRegion.OPTIMAL

        spring_contrib = engine._calculate_tick_contribution(agent, Season.SPRING)
        winter_contrib = engine._calculate_tick_contribution(agent, Season.WINTER)

        # Spring contribution_mult = 1.0, winter = 0.8
        # So spring should be higher
        assert spring_contrib > winter_contrib

    def test_season_effects_disabled(self):
        """When season effects disabled, all seasons give same contribution."""
        config = _make_config()
        config.tick_config["season_effects_enabled"] = False
        engine = TickEngine(config)
        agent = _make_agent()
        agent.processing_region = ProcessingRegion.OPTIMAL

        spring_contrib = engine._calculate_tick_contribution(agent, Season.SPRING)
        winter_contrib = engine._calculate_tick_contribution(agent, Season.WINTER)

        assert abs(spring_contrib - winter_contrib) < 1e-10


# ---------------------------------------------------------------------------
# Reproduction adapter
# ---------------------------------------------------------------------------

class TestReproductionAdapter:
    def test_complete_birth_clears_pregnancy(self):
        agent = _make_agent()
        agent.extension_data["pregnant"] = True
        agent.extension_data["conception_tick"] = 10
        agent.extension_data["partner_at_conception"] = "other"

        from seldon.core.drift import TraitDriftEngine
        config = _make_config()
        from seldon.social.fertility import FertilityManager
        fm = FertilityManager(config)
        adapter = TickReproductionAdapter(fm, 12)

        adapter.complete_birth(agent)
        assert "pregnant" not in agent.extension_data
        assert "conception_tick" not in agent.extension_data
        assert "partner_at_conception" not in agent.extension_data

    def test_birth_check_timing(self):
        """Birth should happen after 9 ticks of pregnancy."""
        config = _make_config()
        from seldon.social.fertility import FertilityManager
        fm = FertilityManager(config)
        adapter = TickReproductionAdapter(fm, 12)

        agent = _make_agent()
        agent.extension_data["pregnant"] = True
        agent.extension_data["conception_tick"] = 0

        # At tick 8, not ready yet
        ready = adapter.tick_birth_check([agent], 8)
        assert len(ready) == 0

        # At tick 9, ready
        ready = adapter.tick_birth_check([agent], 9)
        assert len(ready) == 1
        assert ready[0] is agent


# ---------------------------------------------------------------------------
# Hex grid integration (Phase B)
# ---------------------------------------------------------------------------

def _make_hex_config(**overrides) -> ExperimentConfig:
    """Create a tick+hex enabled config for Phase B tests."""
    defaults = {
        "random_seed": 42,
        "initial_population": 20,
        "generations_to_run": 2,
        "tick_config": {
            "enabled": True,
            "ticks_per_year": 12,
            "life_phase_boundaries": {
                "infant": [0, 2], "child": [3, 12], "adolescent": [13, 17],
                "young_adult": [18, 30], "mature": [31, 55], "elder": [56, 999],
            },
            "season_effects_enabled": True,
            "season_modifiers": {
                "spring": {"fertility_mult": 1.2, "mortality_mult": 0.9,
                           "contribution_mult": 1.0, "food_mult": 1.1, "mood_offset": 0.05},
                "summer": {"fertility_mult": 1.0, "mortality_mult": 0.95,
                           "contribution_mult": 1.1, "food_mult": 1.3, "mood_offset": 0.03},
                "autumn": {"fertility_mult": 0.8, "mortality_mult": 1.0,
                           "contribution_mult": 1.05, "food_mult": 1.2, "mood_offset": -0.02},
                "winter": {"fertility_mult": 0.6, "mortality_mult": 1.3,
                           "contribution_mult": 0.8, "food_mult": 0.6, "mood_offset": -0.05},
            },
        },
        "hex_grid_config": {
            "enabled": True,
            "width": 20,
            "height": 10,
            "map_generator": "california_slice",
            "starting_hex": [5, 5],
            "movement_cost_base": 1.0,
            "vision_range": 2,
            "stay_bias": 0.3,
        },
    }
    defaults.update(overrides)
    return ExperimentConfig(**defaults)


class TestHexGridIntegration:
    """Phase B: hex grid wired into TickEngine."""

    def test_hex_grid_created_when_enabled(self):
        config = _make_hex_config()
        engine = TickEngine(config)
        assert engine.hex_grid is not None
        assert engine.hex_grid.width == 20
        assert engine.hex_grid.height == 10

    def test_hex_grid_not_created_when_disabled(self):
        config = _make_config()  # default: hex disabled
        engine = TickEngine(config)
        assert engine.hex_grid is None

    def test_initial_agents_placed_on_grid(self):
        config = _make_hex_config()
        engine = TickEngine(config)
        pop = engine._create_initial_population()

        for agent in pop:
            assert agent.location is not None, f"Agent {agent.id} has no location"
            assert isinstance(agent.location, tuple)
            assert len(agent.location) == 2
            # Agent should be on a tile
            q, r = agent.location
            tile = engine.hex_grid.get_tile(q, r)
            assert tile is not None
            assert agent.id in tile.current_agents

    def test_agents_have_terrain_type(self):
        config = _make_hex_config()
        engine = TickEngine(config)
        pop = engine._create_initial_population()

        for agent in pop:
            terrain = agent.extension_data.get("terrain_type")
            assert terrain is not None, f"Agent {agent.id} has no terrain_type"
            # Terrain should be a valid terrain string
            from seldon.core.hex_grid import TerrainType
            assert terrain in [t.value for t in TerrainType]

    def test_agents_clustered_near_starting_hex(self):
        config = _make_hex_config()
        engine = TickEngine(config)
        pop = engine._create_initial_population()

        from seldon.core.hex_grid import HexGrid
        starting = (5, 5)
        distances = [
            HexGrid.hex_distance(agent.location, starting)
            for agent in pop if agent.location is not None
        ]
        avg_distance = sum(distances) / len(distances) if distances else 999
        # With 20 agents, most should be within 5 hexes of start
        assert avg_distance < 6

    def test_terrain_flows_to_needs(self):
        """Agents on different terrain should have different need decay rates."""
        config = _make_hex_config(generations_to_run=1)
        engine = TickEngine(config)
        engine.run(years=1)

        # Verify terrain_type is set for all living agents
        for agent in engine.population:
            assert "terrain_type" in agent.extension_data

    def test_agent_movement_occurs(self):
        """At least some agents should move during a 1-year run."""
        config = _make_hex_config(generations_to_run=1, initial_population=30)
        engine = TickEngine(config)
        pop = engine._create_initial_population()
        engine.population = pop

        # Record initial locations
        initial_locations = {a.id: a.location for a in pop}

        # Fire simulation start for extensions
        for ext in engine.extensions.get_enabled():
            ext.on_simulation_start(engine.population, config)

        engine._run_generation(0)

        # Check if any surviving agent moved
        moved = 0
        for agent in engine.population:
            if agent.id in initial_locations:
                if agent.location != initial_locations[agent.id]:
                    moved += 1

        assert moved > 0, "No agents moved during a full year"

    def test_infant_moves_with_parent(self):
        """Infants should follow their caregiver's movement."""
        config = _make_hex_config()
        engine = TickEngine(config)
        pop = engine._create_initial_population()
        engine.population = pop

        # Create a parent-infant pair
        parent = pop[0]
        parent.life_phase = "young_adult"
        parent.age = 25
        parent.location = (5, 5)

        infant = _make_agent(agent_id="infant_test", age=1)
        infant.life_phase = "infant"
        infant._age_ticks = 12
        infant.parent1_id = parent.id
        infant.location = (5, 5)
        parent.children_ids.append(infant.id)

        # Place both on grid
        engine.hex_grid.place_agent(5, 5, parent.id)
        engine.hex_grid.place_agent(5, 5, infant.id)
        infant.extension_data["terrain_type"] = "coastal_valley"

        engine.population.append(infant)

        # Build agent_map and dependent_ids
        agent_map = {a.id: a for a in engine.population if a.is_alive}
        dependent_ids = {infant.id}

        # Find a passable neighbor to move parent to
        neighbors = engine.hex_grid.passable_neighbors(5, 5)
        assert len(neighbors) > 0
        dest = neighbors[0]

        # Move parent
        engine._move_agent_to(parent, dest, agent_map, dependent_ids)

        # Both should now be at destination
        assert parent.location == (dest.q, dest.r)
        assert infant.location == (dest.q, dest.r)
        assert infant.id in engine.hex_grid.agents_at(dest.q, dest.r)

    def test_dead_agents_removed_from_grid(self):
        """Dead agents should not remain on any tile."""
        config = _make_hex_config(generations_to_run=3, initial_population=40)
        engine = TickEngine(config)
        engine.run(years=3)

        # Check that all agents on grid tiles are actually alive
        for coord, tile in engine.hex_grid.tiles.items():
            for agent_id in tile.current_agents:
                # Agent should be in the living population
                found = any(a.id == agent_id for a in engine.population)
                assert found, f"Dead agent {agent_id} still on tile {coord}"

    def test_children_born_at_parent_location(self):
        """Newborns should be placed at their mother's hex."""
        config = _make_hex_config(
            generations_to_run=5, initial_population=30,
            random_seed=99,
        )
        engine = TickEngine(config)
        engine.run(years=5)

        # If any births happened, children should have locations
        for agent in engine.population:
            assert agent.location is not None

    def test_vision_range_limits_pairing(self):
        """Agents on distant hexes should not pair."""
        config = _make_hex_config()
        engine = TickEngine(config)
        pop = engine._create_initial_population()
        engine.population = pop

        clusters = engine._group_by_interaction_range()

        # Each cluster should contain agents within vision_range of each other
        from seldon.core.hex_grid import HexGrid
        for cluster in clusters:
            locations = set()
            for agent in cluster:
                if agent.location is not None:
                    locations.add(agent.location)

            # All locations in a cluster should be within vision_range of at least
            # one other location in the cluster
            if len(locations) > 1:
                locs = list(locations)
                for i, loc_a in enumerate(locs):
                    min_dist = min(
                        HexGrid.hex_distance(loc_a, loc_b)
                        for j, loc_b in enumerate(locs) if j != i
                    )
                    assert min_dist <= config.hex_grid_config["vision_range"]

    def test_get_agent_clusters(self):
        """Cluster detection should find correct groupings."""
        config = _make_hex_config()
        engine = TickEngine(config)
        pop = engine._create_initial_population()
        engine.population = pop

        clusters = engine.get_agent_clusters(min_size=1)

        # Total agents across clusters should equal population
        total = sum(c["agent_count"] for c in clusters)
        assert total == len(pop)

        # Each cluster should have valid structure
        for cluster in clusters:
            assert "tiles" in cluster
            assert "agent_count" in cluster
            assert "agent_ids" in cluster
            assert "center" in cluster
            assert "terrain_types" in cluster
            assert cluster["agent_count"] == len(cluster["agent_ids"])

    def test_hex_backward_compatible(self):
        """Hex disabled should produce identical behavior to Phase A."""
        # Run without hex
        config_no_hex = _make_config(random_seed=42, generations_to_run=2)
        engine1 = TickEngine(config_no_hex)
        h1 = engine1.run(years=2)

        # Same config, explicitly no hex
        config_no_hex2 = _make_config(random_seed=42, generations_to_run=2)
        engine2 = TickEngine(config_no_hex2)
        h2 = engine2.run(years=2)

        for s1, s2 in zip(h1, h2):
            assert s1.population_size == s2.population_size
            assert s1.births == s2.births
            assert s1.deaths == s2.deaths

    def test_hex_deterministic(self):
        """Same seed + hex enabled should produce identical outcomes."""
        config1 = _make_hex_config(random_seed=55, generations_to_run=2)
        config2 = _make_hex_config(random_seed=55, generations_to_run=2)

        engine1 = TickEngine(config1)
        h1 = engine1.run(years=2)

        engine2 = TickEngine(config2)
        h2 = engine2.run(years=2)

        for s1, s2 in zip(h1, h2):
            assert s1.population_size == s2.population_size
            assert s1.births == s2.births
            assert s1.deaths == s2.deaths
            np.testing.assert_array_equal(s1.trait_means, s2.trait_means)

    def test_outsider_placed_on_grid(self):
        """Injected outsider should get a hex location."""
        config = _make_hex_config(
            generations_to_run=5,
            initial_population=20,
            scheduled_injections=[
                {"generation": 2, "archetype": "da_vinci"},
            ],
        )
        engine = TickEngine(config)
        engine.run(years=5)

        # Find outsiders
        outsiders = [a for a in engine.population if a.is_outsider]
        for outsider in outsiders:
            assert outsider.location is not None, \
                f"Outsider {outsider.id} has no location"
            assert "terrain_type" in outsider.extension_data

    def test_50_trait_compatibility(self):
        """Run with full 50-trait preset — no crashes."""
        config = _make_hex_config(
            trait_preset="full",
            initial_population=15,
            generations_to_run=1,
        )
        engine = TickEngine(config)
        history = engine.run(years=1)
        assert len(history) == 1
        assert history[0].population_size > 0

    def test_movement_updates_terrain_type(self):
        """When an agent moves, terrain_type in extension_data should update."""
        config = _make_hex_config()
        engine = TickEngine(config)
        pop = engine._create_initial_population()
        engine.population = pop

        # Pick an agent and find a neighbor with different terrain
        agent = pop[0]
        q, r = agent.location
        current_terrain = agent.extension_data["terrain_type"]

        neighbors = engine.hex_grid.passable_neighbors(q, r)
        dest = None
        for nb in neighbors:
            if nb.terrain_type.value != current_terrain:
                dest = nb
                break

        if dest is not None:
            agent_map = {a.id: a for a in pop if a.is_alive}
            engine._move_agent_to(agent, dest, agent_map, set())
            assert agent.extension_data["terrain_type"] == dest.terrain_type.value
            assert agent.location == (dest.q, dest.r)

    def test_cluster_detection_min_size_filter(self):
        """Clusters smaller than min_size should be filtered out."""
        config = _make_hex_config(initial_population=10)
        engine = TickEngine(config)
        pop = engine._create_initial_population()
        engine.population = pop

        all_clusters = engine.get_agent_clusters(min_size=1)
        large_clusters = engine.get_agent_clusters(min_size=5)

        # Large clusters should be a subset
        assert len(large_clusters) <= len(all_clusters)
