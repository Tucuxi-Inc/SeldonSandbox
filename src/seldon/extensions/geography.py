"""
Geography Extension — spatial simulation with locations.

Provides hex-grid locations with carrying capacities and resource richness.
Distance affects agent interaction (attraction decay).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from seldon.extensions.base import SimulationExtension

if TYPE_CHECKING:
    from seldon.core.agent import Agent
    from seldon.core.config import ExperimentConfig


@dataclass
class Location:
    """A spatial location/settlement in the simulation world."""

    id: str
    name: str
    coordinates: tuple[int, int]
    carrying_capacity: int = 50
    current_population: int = 0
    resource_richness: float = 1.0
    connections: list[str] = field(default_factory=list)


class GeographyExtension(SimulationExtension):
    """Spatial dimension: locations with carrying capacity and distance."""

    @property
    def name(self) -> str:
        return "geography"

    @property
    def description(self) -> str:
        return "Spatial simulation with locations, carrying capacity, and distance-based interaction"

    def get_default_config(self) -> dict[str, Any]:
        return {
            "map_type": "hex",
            "map_size": [10, 10],
            "starting_locations": 3,
            "base_carrying_capacity": 50,
            "capacity_variation": 0.3,
            "max_interaction_distance": 2,
            "attraction_distance_decay": 0.5,
        }

    def __init__(self) -> None:
        self.locations: dict[str, Location] = {}
        self.rng: np.random.Generator | None = None

    def _get_config(self, config: ExperimentConfig) -> dict[str, Any]:
        defaults = self.get_default_config()
        overrides = config.extensions.get("geography", {})
        defaults.update(overrides)
        return defaults

    def on_simulation_start(
        self, population: list[Agent], config: ExperimentConfig,
    ) -> None:
        """Create initial locations and assign agents."""
        geo = self._get_config(config)
        self.rng = np.random.default_rng(config.random_seed)

        n_locations = geo["starting_locations"]
        map_size = geo["map_size"]
        base_cap = geo["base_carrying_capacity"]
        variation = geo["capacity_variation"]

        self.locations = {}
        for i in range(n_locations):
            loc_id = f"loc_{i:03d}"
            x = int(self.rng.integers(0, map_size[0]))
            y = int(self.rng.integers(0, map_size[1]))
            cap = int(base_cap * (1 + self.rng.uniform(-variation, variation)))
            self.locations[loc_id] = Location(
                id=loc_id,
                name=f"Settlement {i + 1}",
                coordinates=(x, y),
                carrying_capacity=max(cap, 10),
                resource_richness=float(self.rng.uniform(0.5, 1.5)),
            )

        # Assign agents round-robin to locations
        loc_ids = list(self.locations.keys())
        for i, agent in enumerate(population):
            agent.location_id = loc_ids[i % len(loc_ids)]

        self._update_population_counts(population)

    def on_agent_created(
        self, agent: Agent, parents: tuple[Agent, Agent],
        config: ExperimentConfig,
    ) -> None:
        """New agents are born at their first parent's location."""
        if parents and parents[0].location_id:
            agent.location_id = parents[0].location_id

    def on_generation_end(
        self, generation: int, population: list[Agent],
        config: ExperimentConfig,
    ) -> None:
        """Update population counts per location."""
        self._update_population_counts(population)

    def modify_attraction(
        self, agent1: Agent, agent2: Agent, base_score: float,
        config: ExperimentConfig,
    ) -> float:
        """Decay attraction by distance between agents' locations."""
        if agent1.location_id == agent2.location_id:
            return base_score
        if agent1.location_id is None or agent2.location_id is None:
            return base_score

        geo = self._get_config(config)
        loc1 = self.locations.get(agent1.location_id)
        loc2 = self.locations.get(agent2.location_id)
        if loc1 is None or loc2 is None:
            return base_score

        dist = self.hex_distance(loc1.coordinates, loc2.coordinates)
        max_dist = geo["max_interaction_distance"]

        if dist > max_dist:
            return 0.0

        decay = geo["attraction_distance_decay"]
        modifier = max(0.0, 1.0 - decay * dist)
        return base_score * modifier

    def get_metrics(self, population: list[Agent]) -> dict[str, Any]:
        """Per-settlement population and resource data."""
        self._update_population_counts(population)
        settlements = {}
        for loc_id, loc in self.locations.items():
            settlements[loc_id] = {
                "name": loc.name,
                "population": loc.current_population,
                "carrying_capacity": loc.carrying_capacity,
                "occupancy_ratio": (
                    loc.current_population / max(loc.carrying_capacity, 1)
                ),
                "resource_richness": loc.resource_richness,
                "coordinates": list(loc.coordinates),
            }
        return {
            "settlement_count": len(self.locations),
            "settlements": settlements,
            "total_capacity": sum(
                loc.carrying_capacity for loc in self.locations.values()
            ),
        }

    # --- Helpers ---

    def _update_population_counts(self, population: list[Agent]) -> None:
        for loc in self.locations.values():
            loc.current_population = 0
        for agent in population:
            if agent.location_id and agent.location_id in self.locations:
                self.locations[agent.location_id].current_population += 1

    @staticmethod
    def hex_distance(a: tuple[int, int], b: tuple[int, int]) -> float:
        """Hex-grid distance using axial coordinates."""
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        return float(max(dx, dy, abs(dx - dy)))

    def get_location(self, location_id: str) -> Location | None:
        """Return a location by ID, or None."""
        return self.locations.get(location_id)

    def add_location(self, location: Location) -> None:
        """Add a new location (used by migration extension)."""
        self.locations[location.id] = location
