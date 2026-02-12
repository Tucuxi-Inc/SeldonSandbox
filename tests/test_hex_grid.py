"""
Tests for the hex grid geography system.

Covers HexTile properties and serialization, HexGrid operations
(neighbors, distance, pathfinding, area queries, agent management),
and map generators (California slice, factory function).
"""

import math

import pytest
import numpy as np

from seldon.core.hex_grid import (
    TerrainType,
    HexTile,
    HexGrid,
    TERRAIN_MOVEMENT_COST,
    TERRAIN_HABITABILITY,
)
from seldon.core.map_generators import (
    generate_california_slice,
    generate_map,
    MAP_GENERATORS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tile(q: int = 0, r: int = 0, terrain: TerrainType = TerrainType.PLAINS) -> HexTile:
    """Create a simple tile for testing."""
    return HexTile(q=q, r=r, terrain_type=terrain)


def _make_grid() -> HexGrid:
    """Create a small 5x5 grid with mixed terrain for testing.

    Layout (terrain by column):
        Column 0: OCEAN
        Column 1: COAST
        Column 2: PLAINS
        Column 3: FOREST
        Column 4: MOUNTAINS
    """
    grid = HexGrid(5, 5)
    terrain_by_col = {
        0: TerrainType.OCEAN,
        1: TerrainType.COAST,
        2: TerrainType.PLAINS,
        3: TerrainType.FOREST,
        4: TerrainType.MOUNTAINS,
    }
    for r in range(5):
        for q in range(5):
            tile = HexTile(
                q=q,
                r=r,
                terrain_type=terrain_by_col[q],
                elevation=q * 100.0,
                water_access=max(0.0, 1.0 - q * 0.2),
                soil_quality=0.5,
                capacity=10 if terrain_by_col[q] != TerrainType.OCEAN else 0,
            )
            grid.add_tile(tile)
    return grid


# ===========================================================================
# HexTile Tests
# ===========================================================================


class TestHexTile:
    """Tests for HexTile dataclass."""

    def test_tile_coords(self):
        """Verify coords property returns (q, r) tuple."""
        tile = _make_tile(3, 7)
        assert tile.coords == (3, 7)

    def test_tile_cube_coords(self):
        """Verify cube coordinate conversion satisfies x + y + z = 0."""
        tile = _make_tile(3, 2)
        x, y, z = tile.cube_coords
        assert x == 3
        assert y == -3 - 2  # -q - r = -5
        assert z == 2
        assert x + y + z == 0

    def test_tile_cube_coords_origin(self):
        """Cube coords at origin should be (0, 0, 0)."""
        tile = _make_tile(0, 0)
        assert tile.cube_coords == (0, 0, 0)

    def test_tile_is_habitable_ocean(self):
        """Ocean tiles are not habitable."""
        tile = _make_tile(terrain=TerrainType.OCEAN)
        assert not tile.is_habitable

    def test_tile_is_habitable_river_valley(self):
        """River valley tiles are habitable."""
        tile = _make_tile(terrain=TerrainType.RIVER_VALLEY)
        assert tile.is_habitable

    def test_tile_movement_cost_ocean(self):
        """Ocean has infinite movement cost."""
        tile = _make_tile(terrain=TerrainType.OCEAN)
        assert tile.movement_cost == float("inf")

    def test_tile_movement_cost_plains(self):
        """Plains have movement cost of 1.0."""
        tile = _make_tile(terrain=TerrainType.PLAINS)
        assert tile.movement_cost == 1.0

    def test_tile_habitability_values(self):
        """Verify habitability constants cover all terrain types."""
        for terrain in TerrainType:
            assert terrain in TERRAIN_HABITABILITY
            assert terrain in TERRAIN_MOVEMENT_COST
            tile = _make_tile(terrain=terrain)
            assert tile.habitability == TERRAIN_HABITABILITY[terrain]

    def test_tile_temperature_for_season(self):
        """Season 0 (summer) should be warmer than season 2 (winter)."""
        tile = HexTile(
            q=0, r=0, terrain_type=TerrainType.PLAINS,
            base_temperature=20.0, temperature_variance=10.0,
        )
        summer = tile.temperature_for_season(0)  # cos(0) = 1
        winter = tile.temperature_for_season(2)  # cos(pi) = -1
        assert summer == pytest.approx(30.0)
        assert winter == pytest.approx(10.0)
        assert summer > winter

    def test_tile_temperature_equinox(self):
        """Equinox seasons (1, 3) should equal base temperature."""
        tile = HexTile(
            q=0, r=0, terrain_type=TerrainType.PLAINS,
            base_temperature=20.0, temperature_variance=10.0,
        )
        # cos(pi/2) = 0, cos(3*pi/2) = 0
        spring = tile.temperature_for_season(1)
        fall = tile.temperature_for_season(3)
        assert spring == pytest.approx(20.0, abs=1e-10)
        assert fall == pytest.approx(20.0, abs=1e-10)

    def test_tile_serialization_roundtrip(self):
        """to_dict() then from_dict() should produce an equivalent tile."""
        original = HexTile(
            q=5, r=3, terrain_type=TerrainType.FOREST,
            elevation=450.0, water_access=0.6, soil_quality=0.5,
            natural_resources=0.7, vegetation=0.8, wildlife=0.6,
            base_temperature=14.0, temperature_variance=11.0,
            precipitation=0.65, capacity=6,
            current_agents=["agent_1", "agent_2"],
        )
        d = original.to_dict()
        restored = HexTile.from_dict(d)

        assert restored.q == original.q
        assert restored.r == original.r
        assert restored.terrain_type == original.terrain_type
        assert restored.elevation == original.elevation
        assert restored.water_access == original.water_access
        assert restored.soil_quality == original.soil_quality
        assert restored.natural_resources == original.natural_resources
        assert restored.vegetation == original.vegetation
        assert restored.wildlife == original.wildlife
        assert restored.base_temperature == original.base_temperature
        assert restored.temperature_variance == original.temperature_variance
        assert restored.precipitation == original.precipitation
        assert restored.capacity == original.capacity
        assert restored.current_agents == original.current_agents


# ===========================================================================
# HexGrid Tests
# ===========================================================================


class TestHexGrid:
    """Tests for HexGrid class."""

    def test_grid_add_and_get(self):
        """Adding a tile and retrieving it by coords."""
        grid = HexGrid(10, 10)
        tile = _make_tile(3, 4)
        grid.add_tile(tile)
        retrieved = grid.get_tile(3, 4)
        assert retrieved is tile

    def test_grid_get_nonexistent(self):
        """Getting a tile that doesn't exist returns None."""
        grid = HexGrid(5, 5)
        assert grid.get_tile(99, 99) is None

    def test_grid_has_tile(self):
        """has_tile correctly reports presence/absence."""
        grid = HexGrid(5, 5)
        tile = _make_tile(1, 2)
        grid.add_tile(tile)
        assert grid.has_tile(1, 2)
        assert not grid.has_tile(9, 9)

    def test_grid_len(self):
        """__len__ returns the number of tiles."""
        grid = _make_grid()
        assert len(grid) == 25  # 5x5

    def test_grid_neighbors_interior(self):
        """Interior tile should have 6 neighbors."""
        grid = _make_grid()
        nbrs = grid.neighbors(2, 2)
        assert len(nbrs) == 6
        # Verify all 6 directions
        expected = {(3, 2), (3, 1), (2, 1), (1, 2), (1, 3), (2, 3)}
        assert set(nbrs) == expected

    def test_grid_neighbors_edge(self):
        """Corner tile should have fewer than 6 neighbors."""
        grid = _make_grid()
        nbrs = grid.neighbors(0, 0)
        # (0,0) neighbors: (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)
        # Only (1,0) and (0,1) exist in a 5x5 grid (q,r in 0..4)
        # (1,-1) -> r=-1: not in grid
        # (0,-1) -> r=-1: not in grid
        # (-1,0) -> q=-1: not in grid
        # (-1,1) -> q=-1: not in grid
        assert len(nbrs) < 6
        assert (1, 0) in nbrs
        assert (0, 1) in nbrs

    def test_grid_valid_neighbors(self):
        """valid_neighbors returns HexTile instances for existing neighbors."""
        grid = _make_grid()
        tiles = grid.valid_neighbors(2, 2)
        assert len(tiles) == 6
        assert all(isinstance(t, HexTile) for t in tiles)

    def test_grid_passable_neighbors(self):
        """passable_neighbors excludes ocean tiles."""
        grid = _make_grid()
        # Tile at (1, 2) is COAST. Its neighbor at (0, 2) is OCEAN.
        passable = grid.passable_neighbors(1, 2)
        coords = [t.coords for t in passable]
        # (0, 2) is ocean -> should be excluded
        assert (0, 2) not in coords
        # (2, 2) is plains -> should be included
        assert (2, 2) in coords

    def test_grid_hex_distance_same(self):
        """Distance from a tile to itself is 0."""
        assert HexGrid.hex_distance((0, 0), (0, 0)) == 0

    def test_grid_hex_distance_adjacent(self):
        """Distance to adjacent tile is 1."""
        assert HexGrid.hex_distance((0, 0), (1, 0)) == 1
        assert HexGrid.hex_distance((0, 0), (0, 1)) == 1

    def test_grid_hex_distance_known(self):
        """Verify known distances."""
        # (0,0) to (3,0): straight line along q axis = 3
        assert HexGrid.hex_distance((0, 0), (3, 0)) == 3
        # (0,0) to (1,1): cube coords (0,0,0) to (1,-2,1) -> max(1,2,1) = 2
        assert HexGrid.hex_distance((0, 0), (1, 1)) == 2
        # (0,0) to (2,-1): cube (0,0,0) to (2,-1,-1) -> max(2,1,1) = 2
        assert HexGrid.hex_distance((0, 0), (2, -1)) == 2

    def test_grid_find_path_simple(self):
        """Find a straight-line path on passable terrain."""
        grid = _make_grid()
        # Path from (2,0) plains to (2,4) plains: straight along column 2
        path = grid.find_path((2, 0), (2, 4))
        assert path is not None
        assert path[0] == (2, 0)
        assert path[-1] == (2, 4)
        assert len(path) >= 5  # At least 5 steps (inclusive)

    def test_grid_find_path_around_obstacle(self):
        """Path should route around impassable ocean tiles."""
        grid = _make_grid()
        # Path from (1, 0) COAST to (1, 4) COAST
        # Direct path goes through column 1 (all coast) so it should work
        path = grid.find_path((1, 0), (1, 4))
        assert path is not None
        assert path[0] == (1, 0)
        assert path[-1] == (1, 4)
        # Verify no ocean tiles in path
        for q, r in path:
            tile = grid.get_tile(q, r)
            assert tile is not None
            assert tile.terrain_type != TerrainType.OCEAN

    def test_grid_find_path_no_route(self):
        """Return None when completely blocked."""
        # Create a grid where start and goal are separated by ocean
        grid = HexGrid(5, 1)
        grid.add_tile(HexTile(q=0, r=0, terrain_type=TerrainType.PLAINS))
        grid.add_tile(HexTile(q=1, r=0, terrain_type=TerrainType.OCEAN))
        grid.add_tile(HexTile(q=2, r=0, terrain_type=TerrainType.OCEAN))
        grid.add_tile(HexTile(q=3, r=0, terrain_type=TerrainType.OCEAN))
        grid.add_tile(HexTile(q=4, r=0, terrain_type=TerrainType.PLAINS))
        path = grid.find_path((0, 0), (4, 0))
        assert path is None

    def test_grid_find_path_start_is_goal(self):
        """Path from a tile to itself should return single-element list."""
        grid = _make_grid()
        path = grid.find_path((2, 2), (2, 2))
        assert path == [(2, 2)]

    def test_grid_find_path_nonexistent_tile(self):
        """Path involving nonexistent tile returns None."""
        grid = _make_grid()
        assert grid.find_path((99, 99), (2, 2)) is None
        assert grid.find_path((2, 2), (99, 99)) is None

    def test_grid_find_path_impassable_goal(self):
        """Path to impassable tile returns None."""
        grid = _make_grid()
        # (0, 0) is ocean
        path = grid.find_path((1, 0), (0, 0))
        assert path is None

    def test_grid_tiles_in_range(self):
        """tiles_in_range returns tiles within hex distance."""
        grid = _make_grid()
        tiles = grid.tiles_in_range((2, 2), 1)
        # Should include center + up to 6 neighbors = 7
        assert len(tiles) == 7
        coords = {t.coords for t in tiles}
        assert (2, 2) in coords

    def test_grid_tiles_in_range_zero(self):
        """Range 0 returns only the center tile."""
        grid = _make_grid()
        tiles = grid.tiles_in_range((2, 2), 0)
        assert len(tiles) == 1
        assert tiles[0].coords == (2, 2)

    def test_grid_habitable_tiles(self):
        """habitable_tiles filters out ocean."""
        grid = _make_grid()
        habitable = grid.habitable_tiles()
        # Column 0 = ocean (5 tiles), rest habitable (20 tiles)
        assert len(habitable) == 20
        for tile in habitable:
            assert tile.terrain_type != TerrainType.OCEAN

    def test_grid_tiles_by_terrain(self):
        """tiles_by_terrain returns only matching terrain."""
        grid = _make_grid()
        oceans = grid.tiles_by_terrain(TerrainType.OCEAN)
        assert len(oceans) == 5  # column 0, 5 rows
        plains = grid.tiles_by_terrain(TerrainType.PLAINS)
        assert len(plains) == 5  # column 2, 5 rows

    def test_grid_place_agent(self):
        """Placing an agent on a tile."""
        grid = _make_grid()
        assert grid.place_agent(2, 2, "agent_1")
        assert "agent_1" in grid.agents_at(2, 2)
        assert grid.agent_count(2, 2) == 1

    def test_grid_place_agent_nonexistent_tile(self):
        """Placing an agent on a nonexistent tile returns False."""
        grid = _make_grid()
        assert not grid.place_agent(99, 99, "agent_1")

    def test_grid_remove_agent(self):
        """Removing an agent from a tile."""
        grid = _make_grid()
        grid.place_agent(2, 2, "agent_1")
        assert grid.remove_agent(2, 2, "agent_1")
        assert grid.agent_count(2, 2) == 0

    def test_grid_remove_agent_not_present(self):
        """Removing a non-present agent returns False."""
        grid = _make_grid()
        assert not grid.remove_agent(2, 2, "ghost")

    def test_grid_move_agent(self):
        """Moving an agent between tiles."""
        grid = _make_grid()
        grid.place_agent(2, 2, "agent_1")
        assert grid.move_agent(2, 2, 3, 2, "agent_1")
        assert grid.agent_count(2, 2) == 0
        assert grid.agent_count(3, 2) == 1
        assert "agent_1" in grid.agents_at(3, 2)

    def test_grid_move_agent_to_nonexistent(self):
        """Moving to nonexistent tile fails and restores agent."""
        grid = _make_grid()
        grid.place_agent(2, 2, "agent_1")
        assert not grid.move_agent(2, 2, 99, 99, "agent_1")
        # Agent should still be at original location
        assert grid.agent_count(2, 2) == 1

    def test_grid_agents_at_empty(self):
        """agents_at returns empty list for tile with no agents."""
        grid = _make_grid()
        assert grid.agents_at(2, 2) == []

    def test_grid_agents_at_nonexistent(self):
        """agents_at returns empty list for nonexistent tile."""
        grid = _make_grid()
        assert grid.agents_at(99, 99) == []

    def test_grid_serialization_roundtrip(self):
        """Full grid serialization and deserialization."""
        grid = _make_grid()
        grid.place_agent(2, 2, "agent_1")
        grid.place_agent(3, 1, "agent_2")

        d = grid.to_dict()
        restored = HexGrid.from_dict(d)

        assert restored.width == grid.width
        assert restored.height == grid.height
        assert len(restored) == len(grid)
        assert restored.agent_count(2, 2) == 1
        assert "agent_1" in restored.agents_at(2, 2)
        assert "agent_2" in restored.agents_at(3, 1)

        # Verify tile properties survived
        orig_tile = grid.get_tile(2, 2)
        rest_tile = restored.get_tile(2, 2)
        assert rest_tile is not None
        assert orig_tile is not None
        assert rest_tile.terrain_type == orig_tile.terrain_type
        assert rest_tile.elevation == orig_tile.elevation


# ===========================================================================
# Map Generator Tests
# ===========================================================================


class TestMapGenerators:
    """Tests for procedural map generation."""

    def test_california_slice_dimensions(self):
        """Default California slice should have 200 tiles (20x10)."""
        grid = generate_california_slice(seed=42)
        assert grid.width == 20
        assert grid.height == 10
        assert len(grid) == 200

    def test_california_slice_custom_dimensions(self):
        """Non-default dimensions produce the right tile count."""
        grid = generate_california_slice(width=10, height=5, seed=42)
        assert grid.width == 10
        assert grid.height == 5
        assert len(grid) == 50

    def test_california_slice_terrain_distribution(self):
        """Column ranges map to correct terrain types."""
        grid = generate_california_slice(seed=42)

        # Check terrain for row 0 across all columns
        terrain_by_col = {}
        for q in range(20):
            tile = grid.get_tile(q, 0)
            assert tile is not None
            terrain_by_col[q] = tile.terrain_type

        # Ocean: columns 0-1
        assert terrain_by_col[0] == TerrainType.OCEAN
        assert terrain_by_col[1] == TerrainType.OCEAN
        # Coast: column 2
        assert terrain_by_col[2] == TerrainType.COAST
        # Coastal valley: columns 3-4
        assert terrain_by_col[3] == TerrainType.COASTAL_VALLEY
        assert terrain_by_col[4] == TerrainType.COASTAL_VALLEY
        # Foothills: columns 5-7
        assert terrain_by_col[5] == TerrainType.FOOTHILLS
        assert terrain_by_col[6] == TerrainType.FOOTHILLS
        assert terrain_by_col[7] == TerrainType.FOOTHILLS
        # River valley: columns 8-10
        assert terrain_by_col[8] == TerrainType.RIVER_VALLEY
        assert terrain_by_col[9] == TerrainType.RIVER_VALLEY
        assert terrain_by_col[10] == TerrainType.RIVER_VALLEY
        # Forest: columns 11-13
        assert terrain_by_col[11] == TerrainType.FOREST
        assert terrain_by_col[12] == TerrainType.FOREST
        assert terrain_by_col[13] == TerrainType.FOREST
        # Mountains: columns 14-16
        assert terrain_by_col[14] == TerrainType.MOUNTAINS
        assert terrain_by_col[15] == TerrainType.MOUNTAINS
        assert terrain_by_col[16] == TerrainType.MOUNTAINS
        # High desert: columns 17-18
        assert terrain_by_col[17] == TerrainType.HIGH_DESERT
        assert terrain_by_col[18] == TerrainType.HIGH_DESERT
        # Desert: column 19
        assert terrain_by_col[19] == TerrainType.DESERT

    def test_california_slice_deterministic(self):
        """Same seed produces identical maps."""
        grid1 = generate_california_slice(seed=123)
        grid2 = generate_california_slice(seed=123)

        assert len(grid1) == len(grid2)
        for (q, r), tile1 in grid1.tiles.items():
            tile2 = grid2.get_tile(q, r)
            assert tile2 is not None
            assert tile1.terrain_type == tile2.terrain_type
            assert tile1.elevation == tile2.elevation
            assert tile1.water_access == tile2.water_access
            assert tile1.soil_quality == tile2.soil_quality
            assert tile1.capacity == tile2.capacity

    def test_california_slice_different_seeds(self):
        """Different seeds produce different property values."""
        grid1 = generate_california_slice(seed=1)
        grid2 = generate_california_slice(seed=2)

        # Terrain types should be the same (column-based), but
        # continuous properties should differ due to noise
        differences = 0
        for (q, r), tile1 in grid1.tiles.items():
            tile2 = grid2.get_tile(q, r)
            assert tile2 is not None
            assert tile1.terrain_type == tile2.terrain_type
            if tile1.elevation != tile2.elevation:
                differences += 1
        assert differences > 0

    def test_california_slice_ocean_not_habitable(self):
        """All ocean tiles should be uninhabitable with 0 capacity."""
        grid = generate_california_slice(seed=42)
        ocean_tiles = grid.tiles_by_terrain(TerrainType.OCEAN)
        assert len(ocean_tiles) > 0
        for tile in ocean_tiles:
            assert not tile.is_habitable
            assert tile.capacity == 0

    def test_california_slice_river_valley_fertile(self):
        """River valley tiles should have high soil quality."""
        grid = generate_california_slice(seed=42)
        rv_tiles = grid.tiles_by_terrain(TerrainType.RIVER_VALLEY)
        assert len(rv_tiles) > 0
        avg_soil = sum(t.soil_quality for t in rv_tiles) / len(rv_tiles)
        # Base soil quality is 0.9 +/- 15% noise, so average should be ~0.9
        assert avg_soil > 0.7

    def test_california_slice_elevation_profile(self):
        """Mountains should be higher than coast."""
        grid = generate_california_slice(seed=42)
        coast_tiles = grid.tiles_by_terrain(TerrainType.COAST)
        mountain_tiles = grid.tiles_by_terrain(TerrainType.MOUNTAINS)

        avg_coast_elev = sum(t.elevation for t in coast_tiles) / len(coast_tiles)
        avg_mtn_elev = sum(t.elevation for t in mountain_tiles) / len(mountain_tiles)

        assert avg_mtn_elev > avg_coast_elev
        assert avg_mtn_elev > 2000  # Mountains should be high
        assert avg_coast_elev < 100  # Coast should be near sea level

    def test_california_slice_temperature_gradient(self):
        """Desert should be warmer than mountains."""
        grid = generate_california_slice(seed=42)
        desert_tiles = grid.tiles_by_terrain(TerrainType.DESERT)
        mountain_tiles = grid.tiles_by_terrain(TerrainType.MOUNTAINS)

        avg_desert_temp = sum(t.base_temperature for t in desert_tiles) / len(desert_tiles)
        avg_mtn_temp = sum(t.base_temperature for t in mountain_tiles) / len(mountain_tiles)

        assert avg_desert_temp > avg_mtn_temp

    def test_california_slice_water_access(self):
        """River valley should have higher water access than desert."""
        grid = generate_california_slice(seed=42)
        rv_tiles = grid.tiles_by_terrain(TerrainType.RIVER_VALLEY)
        desert_tiles = grid.tiles_by_terrain(TerrainType.DESERT)

        avg_rv_water = sum(t.water_access for t in rv_tiles) / len(rv_tiles)
        avg_desert_water = sum(t.water_access for t in desert_tiles) / len(desert_tiles)

        assert avg_rv_water > avg_desert_water
        assert avg_rv_water > 0.7
        assert avg_desert_water < 0.2

    def test_california_slice_capacity_values(self):
        """River valley should have highest capacity, desert lowest."""
        grid = generate_california_slice(seed=42)
        rv_tiles = grid.tiles_by_terrain(TerrainType.RIVER_VALLEY)
        desert_tiles = grid.tiles_by_terrain(TerrainType.DESERT)

        avg_rv_cap = sum(t.capacity for t in rv_tiles) / len(rv_tiles)
        avg_desert_cap = sum(t.capacity for t in desert_tiles) / len(desert_tiles)

        assert avg_rv_cap > avg_desert_cap
        assert avg_rv_cap > 10

    def test_generate_map_factory(self):
        """Factory function creates a California slice map."""
        grid = generate_map("california_slice", width=20, height=10, seed=42)
        assert isinstance(grid, HexGrid)
        assert len(grid) == 200

    def test_generate_map_unknown(self):
        """Factory function raises KeyError for unknown generators."""
        with pytest.raises(KeyError, match="Unknown map generator"):
            generate_map("atlantis", width=10, height=10)

    def test_map_generators_registry(self):
        """MAP_GENERATORS dict contains expected generators."""
        assert "california_slice" in MAP_GENERATORS
        assert callable(MAP_GENERATORS["california_slice"])

    def test_california_slice_pathfinding(self):
        """A path should exist from coast to desert (around ocean)."""
        grid = generate_california_slice(seed=42)
        # Coast tile at (2, 5) to desert at (19, 5)
        path = grid.find_path((2, 5), (19, 5))
        assert path is not None
        assert path[0] == (2, 5)
        assert path[-1] == (19, 5)
        # Verify no ocean tiles in path
        for q, r in path:
            tile = grid.get_tile(q, r)
            assert tile.terrain_type != TerrainType.OCEAN

    def test_california_slice_no_path_through_ocean(self):
        """No path should exist into the ocean interior."""
        grid = generate_california_slice(seed=42)
        # From coast (2, 5) to deep ocean (0, 5)
        path = grid.find_path((2, 5), (0, 5))
        assert path is None
