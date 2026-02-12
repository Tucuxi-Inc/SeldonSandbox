"""
Hex grid geography system for the Seldon Sandbox.

Implements a flat-top axial coordinate hex grid with terrain types,
tile properties, pathfinding (A*), and agent placement. Designed to
model geographic cross-sections (e.g., California coast-to-desert).

Coordinate system: axial (q, r) with flat-top hexagons.
Cube coordinates derived as (q, -q-r, r).
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TerrainType(str, Enum):
    """Terrain classifications for hex tiles."""

    OCEAN = "ocean"
    COAST = "coast"
    COASTAL_VALLEY = "coastal_valley"
    FOOTHILLS = "foothills"
    MOUNTAINS = "mountains"
    HIGH_DESERT = "high_desert"
    DESERT = "desert"
    FOREST = "forest"
    PLAINS = "plains"
    RIVER_VALLEY = "river_valley"


# Movement cost per terrain type. inf = impassable.
TERRAIN_MOVEMENT_COST: dict[TerrainType, float] = {
    TerrainType.OCEAN: float("inf"),
    TerrainType.COAST: 1.5,
    TerrainType.COASTAL_VALLEY: 1.0,
    TerrainType.FOOTHILLS: 1.5,
    TerrainType.MOUNTAINS: 3.0,
    TerrainType.HIGH_DESERT: 2.0,
    TerrainType.DESERT: 2.5,
    TerrainType.FOREST: 1.8,
    TerrainType.PLAINS: 1.0,
    TerrainType.RIVER_VALLEY: 1.0,
}

# Habitability score per terrain type. 0 = uninhabitable, 1.0 = ideal.
TERRAIN_HABITABILITY: dict[TerrainType, float] = {
    TerrainType.OCEAN: 0.0,
    TerrainType.COAST: 0.6,
    TerrainType.COASTAL_VALLEY: 0.9,
    TerrainType.FOOTHILLS: 0.5,
    TerrainType.MOUNTAINS: 0.2,
    TerrainType.HIGH_DESERT: 0.15,
    TerrainType.DESERT: 0.1,
    TerrainType.FOREST: 0.4,
    TerrainType.PLAINS: 0.7,
    TerrainType.RIVER_VALLEY: 1.0,
}


@dataclass
class HexTile:
    """A single hexagonal tile in the grid.

    Uses axial coordinates (q, r). Cube coordinates are derived as
    (q, -q-r, r) which satisfies the constraint x + y + z = 0.

    Attributes:
        q: Column coordinate (axial).
        r: Row coordinate (axial).
        terrain_type: The terrain classification for this tile.
        elevation: Height in meters above sea level.
        water_access: Proximity/availability of fresh water [0, 1].
        soil_quality: Agricultural potential [0, 1].
        natural_resources: Availability of extractable resources [0, 1].
        vegetation: Plant cover density [0, 1].
        wildlife: Animal population density [0, 1].
        base_temperature: Mean annual temperature in Celsius.
        temperature_variance: Seasonal temperature swing amplitude in Celsius.
        precipitation: Annual precipitation level [0, 1].
        capacity: Maximum number of agents the tile can support.
        current_agents: List of agent IDs currently on this tile.
    """

    q: int
    r: int
    terrain_type: TerrainType
    elevation: float = 0.0
    water_access: float = 0.0
    soil_quality: float = 0.0
    natural_resources: float = 0.0
    vegetation: float = 0.0
    wildlife: float = 0.0
    base_temperature: float = 15.0
    temperature_variance: float = 10.0
    precipitation: float = 0.5
    capacity: int = 10
    current_agents: list[str] = field(default_factory=list)

    @property
    def coords(self) -> tuple[int, int]:
        """Axial coordinates as a tuple."""
        return (self.q, self.r)

    @property
    def cube_coords(self) -> tuple[int, int, int]:
        """Cube coordinates derived from axial. Satisfies x + y + z = 0."""
        return (self.q, -self.q - self.r, self.r)

    @property
    def is_habitable(self) -> bool:
        """Whether agents can live on this tile."""
        return TERRAIN_HABITABILITY[self.terrain_type] > 0

    @property
    def movement_cost(self) -> float:
        """Cost to traverse this tile."""
        return TERRAIN_MOVEMENT_COST[self.terrain_type]

    @property
    def habitability(self) -> float:
        """Base habitability score from terrain type."""
        return TERRAIN_HABITABILITY[self.terrain_type]

    def temperature_for_season(self, season_index: int) -> float:
        """Calculate temperature for a given season.

        Uses cosine wave: season 0 = summer (warmest), season 2 = winter (coldest).

        Args:
            season_index: Season number (0=summer, 1=fall, 2=winter, 3=spring).

        Returns:
            Temperature in Celsius for the given season.
        """
        return self.base_temperature + self.temperature_variance * math.cos(
            2.0 * math.pi * season_index / 4.0
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize tile to dictionary."""
        return {
            "q": self.q,
            "r": self.r,
            "terrain_type": self.terrain_type.value,
            "elevation": self.elevation,
            "water_access": self.water_access,
            "soil_quality": self.soil_quality,
            "natural_resources": self.natural_resources,
            "vegetation": self.vegetation,
            "wildlife": self.wildlife,
            "base_temperature": self.base_temperature,
            "temperature_variance": self.temperature_variance,
            "precipitation": self.precipitation,
            "capacity": self.capacity,
            "current_agents": list(self.current_agents),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> HexTile:
        """Deserialize tile from dictionary."""
        return cls(
            q=d["q"],
            r=d["r"],
            terrain_type=TerrainType(d["terrain_type"]),
            elevation=d.get("elevation", 0.0),
            water_access=d.get("water_access", 0.0),
            soil_quality=d.get("soil_quality", 0.0),
            natural_resources=d.get("natural_resources", 0.0),
            vegetation=d.get("vegetation", 0.0),
            wildlife=d.get("wildlife", 0.0),
            base_temperature=d.get("base_temperature", 15.0),
            temperature_variance=d.get("temperature_variance", 10.0),
            precipitation=d.get("precipitation", 0.5),
            capacity=d.get("capacity", 10),
            current_agents=list(d.get("current_agents", [])),
        )


class HexGrid:
    """A hex grid composed of HexTile instances in axial coordinates.

    Supports neighbor lookups, A* pathfinding, range queries, and
    agent placement/movement. Uses flat-top hex orientation with
    the standard 6 axial directions.

    Attributes:
        width: Logical width of the grid (number of columns).
        height: Logical height of the grid (number of rows).
        tiles: Mapping from (q, r) coordinates to HexTile instances.
    """

    # Flat-top axial hex directions (6 neighbors).
    DIRECTIONS: list[tuple[int, int]] = [
        (1, 0),
        (1, -1),
        (0, -1),
        (-1, 0),
        (-1, 1),
        (0, 1),
    ]

    def __init__(self, width: int, height: int) -> None:
        """Initialize an empty hex grid.

        Args:
            width: Logical width (columns).
            height: Logical height (rows).
        """
        self.width: int = width
        self.height: int = height
        self.tiles: dict[tuple[int, int], HexTile] = {}

    def add_tile(self, tile: HexTile) -> None:
        """Add a tile to the grid, keyed by its axial coordinates."""
        self.tiles[tile.coords] = tile

    def get_tile(self, q: int, r: int) -> HexTile | None:
        """Retrieve a tile by coordinates, or None if not present."""
        return self.tiles.get((q, r))

    def has_tile(self, q: int, r: int) -> bool:
        """Check whether a tile exists at the given coordinates."""
        return (q, r) in self.tiles

    def __len__(self) -> int:
        """Number of tiles in the grid."""
        return len(self.tiles)

    # ---- Neighbor queries ----

    def neighbors(self, q: int, r: int) -> list[tuple[int, int]]:
        """Return coordinates of all neighbors that exist in the grid.

        Args:
            q: Column coordinate.
            r: Row coordinate.

        Returns:
            List of (q, r) tuples for existing neighbor tiles.
        """
        result: list[tuple[int, int]] = []
        for dq, dr in self.DIRECTIONS:
            nq, nr = q + dq, r + dr
            if (nq, nr) in self.tiles:
                result.append((nq, nr))
        return result

    def valid_neighbors(self, q: int, r: int) -> list[HexTile]:
        """Return neighbor tiles that exist in the grid.

        Args:
            q: Column coordinate.
            r: Row coordinate.

        Returns:
            List of HexTile instances for existing neighbors.
        """
        return [self.tiles[coord] for coord in self.neighbors(q, r)]

    def passable_neighbors(self, q: int, r: int) -> list[HexTile]:
        """Return neighbor tiles that are habitable (movement cost < inf).

        Args:
            q: Column coordinate.
            r: Row coordinate.

        Returns:
            List of HexTile instances for passable neighbors.
        """
        return [
            tile
            for tile in self.valid_neighbors(q, r)
            if tile.movement_cost < float("inf")
        ]

    # ---- Distance ----

    @staticmethod
    def hex_distance(a: tuple[int, int], b: tuple[int, int]) -> int:
        """Calculate the hex distance between two axial coordinates.

        Converts to cube coordinates and takes the maximum absolute
        difference across the three axes.

        Args:
            a: First position as (q, r).
            b: Second position as (q, r).

        Returns:
            Integer distance in hex steps.
        """
        # Convert axial to cube: (q, -q-r, r)
        ax, az = a
        ay = -ax - az
        bx, bz = b
        by = -bx - bz
        return max(abs(ax - bx), abs(ay - by), abs(az - bz))

    # ---- Pathfinding ----

    def find_path(
        self, start: tuple[int, int], goal: tuple[int, int]
    ) -> list[tuple[int, int]] | None:
        """Find the shortest path between two tiles using A* search.

        Uses movement cost as edge weight and hex distance as heuristic.
        Only traverses passable tiles (movement cost < inf).

        Args:
            start: Starting coordinates (q, r).
            goal: Goal coordinates (q, r).

        Returns:
            List of (q, r) coordinates from start to goal inclusive,
            or None if no path exists.
        """
        if start not in self.tiles or goal not in self.tiles:
            return None

        start_tile = self.tiles[start]
        goal_tile = self.tiles[goal]

        # Impassable start or goal
        if start_tile.movement_cost == float("inf"):
            return None
        if goal_tile.movement_cost == float("inf"):
            return None

        # Priority queue: (f_score, counter, coords)
        # counter breaks ties to avoid comparing tuples
        counter = 0
        open_set: list[tuple[float, int, tuple[int, int]]] = []
        heapq.heappush(open_set, (0.0, counter, start))

        came_from: dict[tuple[int, int], tuple[int, int]] = {}
        g_score: dict[tuple[int, int], float] = {start: 0.0}

        while open_set:
            _, _, current = heapq.heappop(open_set)

            if current == goal:
                # Reconstruct path
                path: list[tuple[int, int]] = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path

            for neighbor_tile in self.passable_neighbors(*current):
                neighbor = neighbor_tile.coords
                tentative_g = g_score[current] + neighbor_tile.movement_cost

                if tentative_g < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self.hex_distance(neighbor, goal)
                    counter += 1
                    heapq.heappush(open_set, (f_score, counter, neighbor))

        return None  # No path found

    # ---- Area queries ----

    def tiles_in_range(
        self, center: tuple[int, int], radius: int
    ) -> list[HexTile]:
        """Return all tiles within a given hex distance from center.

        Args:
            center: Center coordinates (q, r).
            radius: Maximum hex distance (inclusive).

        Returns:
            List of HexTile instances within range.
        """
        result: list[HexTile] = []
        for coord, tile in self.tiles.items():
            if self.hex_distance(center, coord) <= radius:
                result.append(tile)
        return result

    def habitable_tiles(self) -> list[HexTile]:
        """Return all tiles that are habitable."""
        return [tile for tile in self.tiles.values() if tile.is_habitable]

    def tiles_by_terrain(self, terrain: TerrainType) -> list[HexTile]:
        """Return all tiles matching a specific terrain type.

        Args:
            terrain: The terrain type to filter by.

        Returns:
            List of matching HexTile instances.
        """
        return [
            tile for tile in self.tiles.values() if tile.terrain_type == terrain
        ]

    # ---- Agent management ----

    def place_agent(self, q: int, r: int, agent_id: str) -> bool:
        """Place an agent on a tile.

        Args:
            q: Column coordinate.
            r: Row coordinate.
            agent_id: Unique identifier for the agent.

        Returns:
            True if the agent was placed, False if the tile doesn't exist.
        """
        tile = self.get_tile(q, r)
        if tile is None:
            return False
        tile.current_agents.append(agent_id)
        return True

    def remove_agent(self, q: int, r: int, agent_id: str) -> bool:
        """Remove an agent from a tile.

        Args:
            q: Column coordinate.
            r: Row coordinate.
            agent_id: Unique identifier for the agent.

        Returns:
            True if the agent was removed, False if tile doesn't exist
            or agent not found on tile.
        """
        tile = self.get_tile(q, r)
        if tile is None:
            return False
        if agent_id not in tile.current_agents:
            return False
        tile.current_agents.remove(agent_id)
        return True

    def move_agent(
        self, from_q: int, from_r: int, to_q: int, to_r: int, agent_id: str
    ) -> bool:
        """Move an agent from one tile to another.

        Args:
            from_q: Source column coordinate.
            from_r: Source row coordinate.
            to_q: Destination column coordinate.
            to_r: Destination row coordinate.
            agent_id: Unique identifier for the agent.

        Returns:
            True if the move succeeded, False otherwise.
        """
        if not self.remove_agent(from_q, from_r, agent_id):
            return False
        if not self.place_agent(to_q, to_r, agent_id):
            # Restore agent to original tile on failure
            self.place_agent(from_q, from_r, agent_id)
            return False
        return True

    def agents_at(self, q: int, r: int) -> list[str]:
        """Return list of agent IDs at a tile.

        Args:
            q: Column coordinate.
            r: Row coordinate.

        Returns:
            List of agent IDs, or empty list if tile doesn't exist.
        """
        tile = self.get_tile(q, r)
        if tile is None:
            return []
        return list(tile.current_agents)

    def agent_count(self, q: int, r: int) -> int:
        """Return number of agents at a tile.

        Args:
            q: Column coordinate.
            r: Row coordinate.

        Returns:
            Agent count, or 0 if tile doesn't exist.
        """
        return len(self.agents_at(q, r))

    # ---- Serialization ----

    def to_dict(self) -> dict[str, Any]:
        """Serialize the grid to a dictionary."""
        return {
            "width": self.width,
            "height": self.height,
            "tiles": [tile.to_dict() for tile in self.tiles.values()],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> HexGrid:
        """Deserialize a grid from a dictionary."""
        grid = cls(width=d["width"], height=d["height"])
        for tile_data in d["tiles"]:
            grid.add_tile(HexTile.from_dict(tile_data))
        return grid
