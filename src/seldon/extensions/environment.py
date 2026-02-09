"""
Environment Extension — climate, seasons, natural events, and disease.

Adds environmental dynamics: seasonal cycles, stochastic events (drought,
flood, plague, bountiful harvests), disease with trait-based resistance,
and carrying capacity modifications.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

from seldon.extensions.base import SimulationExtension

if TYPE_CHECKING:
    from seldon.core.agent import Agent
    from seldon.core.config import ExperimentConfig


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class Season(str, Enum):
    SPRING = "spring"
    SUMMER = "summer"
    AUTUMN = "autumn"
    WINTER = "winter"


class EventType(str, Enum):
    DROUGHT = "drought"
    FLOOD = "flood"
    EARTHQUAKE = "earthquake"
    PLAGUE = "plague"
    BOUNTIFUL_HARVEST = "bountiful_harvest"
    DISCOVERY = "discovery"


# Season modifiers on traits and production
SEASON_EFFECTS = {
    Season.SPRING: {"fertility_mult": 1.2, "mortality_mult": 0.9, "description": "Growth and renewal"},
    Season.SUMMER: {"fertility_mult": 1.0, "mortality_mult": 1.0, "description": "Peak activity"},
    Season.AUTUMN: {"fertility_mult": 0.8, "mortality_mult": 1.0, "description": "Harvest and preparation"},
    Season.WINTER: {"fertility_mult": 0.6, "mortality_mult": 1.3, "description": "Scarcity and hardship"},
}

# Event effects
EVENT_EFFECTS = {
    EventType.DROUGHT: {"mortality_mult": 1.5, "capacity_mult": 0.7},
    EventType.FLOOD: {"mortality_mult": 1.3, "capacity_mult": 0.8},
    EventType.EARTHQUAKE: {"mortality_mult": 1.8, "capacity_mult": 0.6},
    EventType.PLAGUE: {"mortality_mult": 2.0, "capacity_mult": 1.0},
    EventType.BOUNTIFUL_HARVEST: {"mortality_mult": 0.7, "capacity_mult": 1.2},
    EventType.DISCOVERY: {"mortality_mult": 1.0, "capacity_mult": 1.1},
}


@dataclass
class ClimateState:
    """Per-location climate conditions."""
    location_id: str
    temperature: float = 0.5  # [0=cold, 1=hot]
    rainfall: float = 0.5     # [0=dry, 1=wet]
    severity: float = 0.0     # [0=mild, 1=extreme]


@dataclass
class EnvironmentalEvent:
    """A discrete environmental event."""
    event_type: EventType
    generation: int
    location_id: str | None  # None = global
    severity: float = 1.0
    duration: int = 1         # How many generations it lasts
    remaining: int = 1


@dataclass
class Disease:
    """An active disease in the population."""
    name: str
    transmission_rate: float
    mortality_rate: float
    duration_generations: int
    resistance_trait: str  # Which trait helps resist
    infected_agents: set[str] = field(default_factory=set)
    generation_started: int = 0


class EnvironmentExtension(SimulationExtension):
    """Climate, seasons, natural events, and disease dynamics."""

    def __init__(self) -> None:
        self.current_season: Season = Season.SPRING
        self.generation_in_season: int = 0
        self.climate_states: dict[str, ClimateState] = {}
        self.active_events: list[EnvironmentalEvent] = []
        self.event_history: list[dict[str, Any]] = []
        self.active_diseases: list[Disease] = []
        self._rng: np.random.Generator | None = None
        self._global_temperature_drift: float = 0.0

    @property
    def name(self) -> str:
        return "environment"

    @property
    def description(self) -> str:
        return "Climate, seasons, natural events, and disease dynamics"

    def get_default_config(self) -> dict[str, Any]:
        return {
            "requires": ["geography"],
            "seasons_enabled": True,
            "season_length_generations": 5,
            "base_event_probability": 0.1,
            "drought_probability": 0.05,
            "flood_probability": 0.03,
            "plague_probability": 0.02,
            "bountiful_probability": 0.08,
            "discovery_probability": 0.04,
            "climate_drift_rate": 0.001,
            "disease_transmission_rate": 0.15,
            "disease_base_mortality": 0.1,
            "quarantine_effectiveness": 0.5,
        }

    def _get_config(self, config: ExperimentConfig) -> dict[str, Any]:
        defaults = self.get_default_config()
        overrides = config.extensions.get("environment", {})
        for k, v in overrides.items():
            if k != "requires":
                defaults[k] = v
        ec = config.environment_config
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
        self.current_season = Season.SPRING
        self.generation_in_season = 0

        # Initialize climate per location
        locations = {a.location_id for a in population if a.location_id}
        for loc_id in locations:
            self.climate_states[loc_id] = ClimateState(
                location_id=loc_id,
                temperature=float(self._rng.uniform(0.3, 0.7)),
                rainfall=float(self._rng.uniform(0.3, 0.7)),
            )

    def on_generation_start(
        self, generation: int, population: list[Agent],
        config: ExperimentConfig,
    ) -> None:
        """Advance season, generate events, update climate."""
        if self._rng is None:
            self._rng = np.random.default_rng(config.random_seed)

        ec = self._get_config(config)

        # Season advancement
        if ec.get("seasons_enabled", True):
            season_length = ec.get("season_length_generations", 5)
            self.generation_in_season += 1
            if self.generation_in_season >= season_length:
                self.generation_in_season = 0
                seasons = list(Season)
                current_idx = seasons.index(self.current_season)
                self.current_season = seasons[(current_idx + 1) % 4]

        # Climate drift
        drift_rate = ec.get("climate_drift_rate", 0.001)
        self._global_temperature_drift += self._rng.normal(0, drift_rate)
        for cs in self.climate_states.values():
            cs.temperature = float(np.clip(
                cs.temperature + self._rng.normal(0, 0.01) + self._global_temperature_drift,
                0.0, 1.0,
            ))
            cs.rainfall = float(np.clip(cs.rainfall + self._rng.normal(0, 0.01), 0.0, 1.0))
            cs.severity = float(np.clip(
                abs(cs.temperature - 0.5) + abs(cs.rainfall - 0.5), 0.0, 1.0,
            ))

        # Generate events
        self._generate_events(generation, population, ec)

    def on_generation_end(
        self, generation: int, population: list[Agent],
        config: ExperimentConfig,
    ) -> None:
        """Apply event consequences, update disease states."""
        ec = self._get_config(config)

        # Disease transmission
        self._update_diseases(population, config, ec)

        # Expire events
        remaining_events = []
        for event in self.active_events:
            event.remaining -= 1
            if event.remaining > 0:
                remaining_events.append(event)
        self.active_events = remaining_events

    def on_agent_created(
        self, agent: Agent, parents: tuple[Agent, Agent],
        config: ExperimentConfig,
    ) -> None:
        """Newborns may inherit disease resistance epigenetically (handled by epigenetics)."""
        pass

    # ------------------------------------------------------------------
    # Event generation
    # ------------------------------------------------------------------
    def _generate_events(
        self, generation: int, population: list[Agent],
        ec: dict[str, Any],
    ) -> None:
        """Generate stochastic events based on probabilities."""
        if self._rng is None:
            return

        locations = list(self.climate_states.keys()) or [None]

        event_probs = {
            EventType.DROUGHT: ec.get("drought_probability", 0.05),
            EventType.FLOOD: ec.get("flood_probability", 0.03),
            EventType.PLAGUE: ec.get("plague_probability", 0.02),
            EventType.BOUNTIFUL_HARVEST: ec.get("bountiful_probability", 0.08),
            EventType.DISCOVERY: ec.get("discovery_probability", 0.04),
        }

        for event_type, prob in event_probs.items():
            if self._rng.random() < prob:
                loc = locations[self._rng.integers(len(locations))] if locations[0] is not None else None
                severity = float(self._rng.uniform(0.5, 1.0))
                duration = 1 if event_type in (EventType.DISCOVERY, EventType.BOUNTIFUL_HARVEST) else self._rng.integers(1, 4)
                event = EnvironmentalEvent(
                    event_type=event_type,
                    generation=generation,
                    location_id=loc,
                    severity=severity,
                    duration=int(duration),
                    remaining=int(duration),
                )
                self.active_events.append(event)
                self.event_history.append({
                    "event_type": event_type.value,
                    "generation": generation,
                    "location_id": loc,
                    "severity": round(severity, 4),
                    "duration": int(duration),
                })

                # Plague triggers disease
                if event_type == EventType.PLAGUE:
                    self._start_disease(generation, ec)

    def _start_disease(self, generation: int, ec: dict[str, Any]) -> None:
        """Start a new disease outbreak."""
        disease = Disease(
            name=f"plague_g{generation}",
            transmission_rate=ec.get("disease_transmission_rate", 0.15),
            mortality_rate=ec.get("disease_base_mortality", 0.1),
            duration_generations=int(self._rng.integers(2, 6)) if self._rng else 3,
            resistance_trait="resilience",
            generation_started=int(generation),
        )
        self.active_diseases.append(disease)

    # ------------------------------------------------------------------
    # Disease dynamics
    # ------------------------------------------------------------------
    def _update_diseases(
        self, population: list[Agent], config: ExperimentConfig,
        ec: dict[str, Any],
    ) -> None:
        """Process disease transmission and recovery."""
        if self._rng is None:
            return

        ts = config.trait_system
        remaining_diseases = []

        for disease in self.active_diseases:
            disease.duration_generations -= 1
            if disease.duration_generations <= 0:
                # Disease ends
                disease.infected_agents.clear()
                continue

            # Transmission: infected spread to bonded agents
            newly_infected = set()
            for agent in population:
                if not agent.is_alive:
                    continue
                if agent.id in disease.infected_agents:
                    # Already infected — can spread
                    for bonded_id in agent.social_bonds:
                        if bonded_id not in disease.infected_agents:
                            if self._rng.random() < disease.transmission_rate:
                                newly_infected.add(bonded_id)
                else:
                    # Random chance of catching from proximity
                    if self._rng.random() < disease.transmission_rate * 0.1:
                        newly_infected.add(agent.id)

            disease.infected_agents.update(newly_infected)
            remaining_diseases.append(disease)

        self.active_diseases = remaining_diseases

    # ------------------------------------------------------------------
    # Modifier hooks
    # ------------------------------------------------------------------
    def modify_mortality(
        self, agent: Agent, base_rate: float, config: ExperimentConfig,
    ) -> float:
        """Season + events + disease modify mortality."""
        rate = base_rate

        # Season effects
        season_fx = SEASON_EFFECTS.get(self.current_season, {})
        rate *= season_fx.get("mortality_mult", 1.0)

        # Active events at this agent's location
        for event in self.active_events:
            if event.location_id is None or event.location_id == agent.location_id:
                fx = EVENT_EFFECTS.get(event.event_type, {})
                rate *= fx.get("mortality_mult", 1.0) * event.severity

        # Disease
        ts = config.trait_system
        for disease in self.active_diseases:
            if agent.id in disease.infected_agents:
                # Resistance from trait
                try:
                    idx = ts.trait_index(disease.resistance_trait)
                    resistance = float(agent.traits[idx])
                except KeyError:
                    resistance = 0.5
                disease_mort = disease.mortality_rate * (1.0 - resistance * 0.5)
                rate += disease_mort

        return float(np.clip(rate, 0.0, 1.0))

    def modify_decision(
        self, agent: Agent, context: str, utilities: dict[str, float],
        config: ExperimentConfig,
    ) -> dict[str, float]:
        """Harsh conditions increase migration utility."""
        # Check if agent's location has active negative events
        has_negative = False
        has_positive = False
        for event in self.active_events:
            if event.location_id is None or event.location_id == agent.location_id:
                if event.event_type in (EventType.DROUGHT, EventType.FLOOD, EventType.EARTHQUAKE, EventType.PLAGUE):
                    has_negative = True
                elif event.event_type in (EventType.BOUNTIFUL_HARVEST, EventType.DISCOVERY):
                    has_positive = True

        if has_negative and "migrate" in utilities:
            utilities["migrate"] = utilities.get("migrate", 0.0) + 0.15
        if has_positive and "stay" in utilities:
            utilities["stay"] = utilities.get("stay", 0.0) + 0.1

        return utilities

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    def get_metrics(self, population: list[Agent]) -> dict[str, Any]:
        infected_count = 0
        for disease in self.active_diseases:
            infected_count += len(disease.infected_agents)

        return {
            "current_season": self.current_season.value,
            "active_events": len(self.active_events),
            "total_events_occurred": len(self.event_history),
            "active_diseases": len(self.active_diseases),
            "infected_count": infected_count,
            "climate_states": {
                loc_id: {
                    "temperature": round(cs.temperature, 4),
                    "rainfall": round(cs.rainfall, 4),
                    "severity": round(cs.severity, 4),
                }
                for loc_id, cs in self.climate_states.items()
            },
        }

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------
    def get_event_history(self) -> list[dict[str, Any]]:
        return list(self.event_history)

    def get_events_for_generation(self, generation: int) -> list[dict[str, Any]]:
        return [e for e in self.event_history if e["generation"] == generation]

    def get_disease_status(self, population: list[Agent]) -> list[dict[str, Any]]:
        return [
            {
                "name": d.name,
                "transmission_rate": float(d.transmission_rate),
                "mortality_rate": float(d.mortality_rate),
                "duration_remaining": int(d.duration_generations),
                "infected_count": int(len(d.infected_agents)),
                "resistance_trait": d.resistance_trait,
            }
            for d in self.active_diseases
        ]
