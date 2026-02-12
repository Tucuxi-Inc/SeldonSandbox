"""
Map generators for the Seldon Sandbox hex grid system.

Provides procedural generation of themed hex maps. Each generator
produces a fully populated HexGrid with terrain types and tile
properties appropriate for the map theme.

All generators use seeded numpy RNG for deterministic reproduction.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from seldon.core.hex_grid import HexGrid, HexTile, TerrainType


def generate_california_slice(
    width: int = 20, height: int = 10, seed: int | None = None
) -> HexGrid:
    """Generate a California east-west cross-section map.

    Models terrain from Pacific Ocean through the Central Valley,
    Sierra Nevada, and into the high desert. Column ranges map to
    distinct terrain bands:

        Columns 0-1:   OCEAN (impassable)
        Column 2:      COAST (fishing, ports)
        Columns 3-4:   COASTAL_VALLEY (fertile farmland)
        Columns 5-7:   FOOTHILLS (timber, moderate)
        Columns 8-10:  RIVER_VALLEY (Central Valley, very fertile)
        Columns 11-13: FOREST (Sierra foothills)
        Columns 14-16: MOUNTAINS (Sierra Nevada, ore, low habitability)
        Columns 17-18: HIGH_DESERT (sparse)
        Column 19:     DESERT (harsh)

    Args:
        width: Number of columns (default 20 for full cross-section).
        height: Number of rows (default 10).
        seed: Random seed for deterministic generation.

    Returns:
        A fully populated HexGrid.
    """
    rng = np.random.default_rng(seed)
    grid = HexGrid(width, height)

    for r in range(height):
        for q in range(width):
            terrain = _column_to_terrain(q, width)
            props = _generate_tile_properties(q, width, terrain, rng)
            tile = HexTile(
                q=q,
                r=r,
                terrain_type=terrain,
                **props,
            )
            grid.add_tile(tile)

    return grid


def _column_to_terrain(q: int, width: int) -> TerrainType:
    """Map a column index to terrain type for the California slice.

    Scales column ranges proportionally if width differs from 20,
    but uses the canonical 20-column layout as the reference.

    Args:
        q: Column index.
        width: Total grid width.

    Returns:
        TerrainType for this column.
    """
    # Normalize column to 0-19 range for terrain band lookup
    normalized = q * 20.0 / width if width != 20 else float(q)

    if normalized < 2:
        return TerrainType.OCEAN
    elif normalized < 3:
        return TerrainType.COAST
    elif normalized < 5:
        return TerrainType.COASTAL_VALLEY
    elif normalized < 8:
        return TerrainType.FOOTHILLS
    elif normalized < 11:
        return TerrainType.RIVER_VALLEY
    elif normalized < 14:
        return TerrainType.FOREST
    elif normalized < 17:
        return TerrainType.MOUNTAINS
    elif normalized < 19:
        return TerrainType.HIGH_DESERT
    else:
        return TerrainType.DESERT


def _generate_tile_properties(
    q: int,
    width: int,
    terrain: TerrainType,
    rng: np.random.Generator,
) -> dict:
    """Generate environmental properties for a tile based on terrain.

    Applies base values per terrain type with +/-10-20% random noise,
    clipped to valid ranges.

    Args:
        q: Column index (used for elevation gradient).
        width: Total grid width.
        terrain: The terrain type for this tile.
        rng: Seeded random number generator.

    Returns:
        Dictionary of tile property keyword arguments.
    """
    # Terrain-specific base values
    bases = _TERRAIN_BASE_PROPERTIES[terrain]

    # Elevation follows a west-to-east profile
    elevation = _elevation_profile(q, width, terrain)

    # Apply noise (multiply by 1.0 +/- noise_factor)
    noise_factor = 0.15  # ~15% noise

    def noisy(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
        """Apply multiplicative noise and clip to [lo, hi]."""
        noise = 1.0 + rng.uniform(-noise_factor, noise_factor)
        return float(np.clip(value * noise, lo, hi))

    def noisy_int(value: int, lo: int = 0, hi: int = 100) -> int:
        """Apply multiplicative noise to integer value."""
        noise = 1.0 + rng.uniform(-noise_factor, noise_factor)
        return int(np.clip(round(value * noise), lo, hi))

    return {
        "elevation": float(np.clip(
            elevation + rng.uniform(-50, 50), -10.0, 5000.0
        )),
        "water_access": noisy(bases["water_access"]),
        "soil_quality": noisy(bases["soil_quality"]),
        "natural_resources": noisy(bases["natural_resources"]),
        "vegetation": noisy(bases["vegetation"]),
        "wildlife": noisy(bases["wildlife"]),
        "base_temperature": float(np.clip(
            bases["base_temperature"] + rng.uniform(-1.5, 1.5), -20.0, 50.0
        )),
        "temperature_variance": float(np.clip(
            bases["temperature_variance"] + rng.uniform(-1.0, 1.0), 2.0, 25.0
        )),
        "precipitation": noisy(bases["precipitation"]),
        "capacity": noisy_int(bases["capacity"], lo=0, hi=50),
    }


def _elevation_profile(
    q: int, width: int, terrain: TerrainType
) -> float:
    """Calculate base elevation for a column based on terrain type.

    Models the California cross-section: sea level at coast, rising
    through foothills to mountain peaks, then dropping to desert.

    Args:
        q: Column index.
        width: Total grid width.
        terrain: Terrain type at this position.

    Returns:
        Elevation in meters.
    """
    elevation_map: dict[TerrainType, float] = {
        TerrainType.OCEAN: 0.0,
        TerrainType.COAST: 5.0,
        TerrainType.COASTAL_VALLEY: 50.0,
        TerrainType.FOOTHILLS: 300.0,
        TerrainType.RIVER_VALLEY: 80.0,
        TerrainType.FOREST: 600.0,
        TerrainType.MOUNTAINS: 2800.0,
        TerrainType.HIGH_DESERT: 1200.0,
        TerrainType.DESERT: 800.0,
        TerrainType.PLAINS: 150.0,
    }
    return elevation_map.get(terrain, 100.0)


# Base property values for each terrain type. These are the "ideal" values
# before noise is applied.
_TERRAIN_BASE_PROPERTIES: dict[TerrainType, dict] = {
    TerrainType.OCEAN: {
        "water_access": 1.0,
        "soil_quality": 0.0,
        "natural_resources": 0.3,
        "vegetation": 0.0,
        "wildlife": 0.4,
        "base_temperature": 16.0,
        "temperature_variance": 6.0,
        "precipitation": 0.3,
        "capacity": 0,
    },
    TerrainType.COAST: {
        "water_access": 0.9,
        "soil_quality": 0.4,
        "natural_resources": 0.5,
        "vegetation": 0.3,
        "wildlife": 0.5,
        "base_temperature": 18.0,
        "temperature_variance": 8.0,
        "precipitation": 0.5,
        "capacity": 8,
    },
    TerrainType.COASTAL_VALLEY: {
        "water_access": 0.7,
        "soil_quality": 0.85,
        "natural_resources": 0.4,
        "vegetation": 0.6,
        "wildlife": 0.5,
        "base_temperature": 20.0,
        "temperature_variance": 10.0,
        "precipitation": 0.6,
        "capacity": 12,
    },
    TerrainType.FOOTHILLS: {
        "water_access": 0.5,
        "soil_quality": 0.45,
        "natural_resources": 0.5,
        "vegetation": 0.5,
        "wildlife": 0.5,
        "base_temperature": 17.0,
        "temperature_variance": 12.0,
        "precipitation": 0.5,
        "capacity": 8,
    },
    TerrainType.RIVER_VALLEY: {
        "water_access": 0.95,
        "soil_quality": 0.9,
        "natural_resources": 0.5,
        "vegetation": 0.7,
        "wildlife": 0.6,
        "base_temperature": 22.0,
        "temperature_variance": 14.0,
        "precipitation": 0.7,
        "capacity": 15,
    },
    TerrainType.FOREST: {
        "water_access": 0.55,
        "soil_quality": 0.5,
        "natural_resources": 0.7,
        "vegetation": 0.85,
        "wildlife": 0.8,
        "base_temperature": 14.0,
        "temperature_variance": 11.0,
        "precipitation": 0.65,
        "capacity": 6,
    },
    TerrainType.MOUNTAINS: {
        "water_access": 0.3,
        "soil_quality": 0.15,
        "natural_resources": 0.8,
        "vegetation": 0.2,
        "wildlife": 0.3,
        "base_temperature": 8.0,
        "temperature_variance": 12.0,
        "precipitation": 0.5,
        "capacity": 4,
    },
    TerrainType.HIGH_DESERT: {
        "water_access": 0.1,
        "soil_quality": 0.2,
        "natural_resources": 0.3,
        "vegetation": 0.15,
        "wildlife": 0.2,
        "base_temperature": 24.0,
        "temperature_variance": 15.0,
        "precipitation": 0.15,
        "capacity": 3,
    },
    TerrainType.DESERT: {
        "water_access": 0.05,
        "soil_quality": 0.1,
        "natural_resources": 0.2,
        "vegetation": 0.05,
        "wildlife": 0.1,
        "base_temperature": 28.0,
        "temperature_variance": 15.0,
        "precipitation": 0.05,
        "capacity": 2,
    },
    TerrainType.PLAINS: {
        "water_access": 0.4,
        "soil_quality": 0.55,
        "natural_resources": 0.3,
        "vegetation": 0.5,
        "wildlife": 0.5,
        "base_temperature": 20.0,
        "temperature_variance": 13.0,
        "precipitation": 0.45,
        "capacity": 10,
    },
}


# Registry of available map generators.
MAP_GENERATORS: dict[str, Callable[..., HexGrid]] = {
    "california_slice": generate_california_slice,
}


def generate_map(
    name: str, width: int, height: int, seed: int | None = None
) -> HexGrid:
    """Factory function to generate a map by name.

    Args:
        name: Name of the map generator (must be in MAP_GENERATORS).
        width: Grid width.
        height: Grid height.
        seed: Random seed for deterministic generation.

    Returns:
        A populated HexGrid.

    Raises:
        KeyError: If the generator name is not found.
    """
    if name not in MAP_GENERATORS:
        raise KeyError(
            f"Unknown map generator '{name}'. "
            f"Available: {list(MAP_GENERATORS.keys())}"
        )
    return MAP_GENERATORS[name](width=width, height=height, seed=seed)
