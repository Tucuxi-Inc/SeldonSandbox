"""
In-memory session manager for simulation experiments.

Each session wraps a SimulationEngine + MetricsCollector, supporting
step-by-step execution while preserving all agents (including dead)
for family tree and lineage queries.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from seldon.core.agent import Agent
from seldon.core.config import ExperimentConfig
from seldon.core.engine import SimulationEngine
from seldon.extensions.registry import ExtensionRegistry
from seldon.metrics.collector import MetricsCollector


@dataclass
class SimulationSession:
    """A running or completed simulation session."""

    id: str
    name: str
    config: ExperimentConfig
    engine: SimulationEngine
    collector: MetricsCollector
    status: str = "created"  # created | running | completed
    current_generation: int = 0
    max_generations: int = 0
    all_agents: dict[str, Agent] = field(default_factory=dict)


class SessionManager:
    """Manages multiple simulation sessions in memory."""

    def __init__(self):
        self.sessions: dict[str, SimulationSession] = {}

    def create_session(
        self,
        config: ExperimentConfig | None = None,
        name: str | None = None,
    ) -> SimulationSession:
        """Create a new simulation session."""
        if config is None:
            config = ExperimentConfig()

        session_id = uuid.uuid4().hex[:8]
        registry = self._build_extensions(config)
        engine = SimulationEngine(config, extensions=registry)
        engine.population = engine._create_initial_population()

        # Fire on_simulation_start for extensions (run() does this but
        # we call _create_initial_population directly)
        for ext in engine.extensions.get_enabled():
            ext.on_simulation_start(engine.population, config)

        collector = MetricsCollector(config)

        session = SimulationSession(
            id=session_id,
            name=name or config.experiment_name,
            config=config,
            engine=engine,
            collector=collector,
            max_generations=config.generations_to_run,
        )

        # Store initial population
        for agent in engine.population:
            session.all_agents[agent.id] = agent

        self.sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> SimulationSession:
        """Get a session by ID. Raises KeyError if not found."""
        if session_id not in self.sessions:
            raise KeyError(f"Session '{session_id}' not found")
        return self.sessions[session_id]

    def step(self, session_id: str, n: int = 1) -> SimulationSession:
        """Advance a session by N generations."""
        session = self.get_session(session_id)

        if session.status == "completed":
            return session

        session.status = "running"

        for _ in range(n):
            if session.current_generation >= session.max_generations:
                session.status = "completed"
                break

            # Store all current agents (some may die this generation)
            for agent in session.engine.population:
                session.all_agents[agent.id] = agent

            # Run one generation
            snapshot = session.engine._run_generation(session.current_generation)
            session.engine.history.append(snapshot)

            # Collect metrics
            session.collector.collect(session.engine.population, snapshot)

            # Store any new agents (children born this generation)
            for agent in session.engine.population:
                session.all_agents[agent.id] = agent

            session.current_generation += 1

            if session.current_generation >= session.max_generations:
                session.status = "completed"

        return session

    def run_full(self, session_id: str) -> SimulationSession:
        """Run a session to completion."""
        session = self.get_session(session_id)
        remaining = session.max_generations - session.current_generation
        if remaining > 0:
            self.step(session_id, remaining)
        return session

    def reset_session(self, session_id: str) -> SimulationSession:
        """Reset a session to generation 0."""
        session = self.get_session(session_id)

        # Rebuild engine with same config + extensions
        registry = self._build_extensions(session.config)
        engine = SimulationEngine(session.config, extensions=registry)
        engine.population = engine._create_initial_population()

        for ext in engine.extensions.get_enabled():
            ext.on_simulation_start(engine.population, session.config)

        collector = MetricsCollector(session.config)

        session.engine = engine
        session.collector = collector
        session.status = "created"
        session.current_generation = 0
        session.all_agents = {}

        for agent in engine.population:
            session.all_agents[agent.id] = agent

        return session

    def delete_session(self, session_id: str) -> None:
        """Delete a session."""
        if session_id not in self.sessions:
            raise KeyError(f"Session '{session_id}' not found")
        del self.sessions[session_id]

    @staticmethod
    def _build_extensions(config: ExperimentConfig) -> ExtensionRegistry:
        """Build an ExtensionRegistry from config.extensions_enabled."""
        from seldon.extensions import (
            GeographyExtension, MigrationExtension, ResourcesExtension,
            TechnologyExtension, CultureExtension, ConflictExtension,
            SocialDynamicsExtension, DiplomacyExtension, EconomicsExtension,
            EnvironmentExtension,
        )

        registry = ExtensionRegistry()

        if not config.extensions_enabled:
            return registry

        # Register all standard extensions
        geo = GeographyExtension()
        registry.register(geo)
        registry.register(MigrationExtension(geo))
        registry.register(ResourcesExtension())
        registry.register(TechnologyExtension())
        registry.register(CultureExtension())
        registry.register(ConflictExtension())
        registry.register(SocialDynamicsExtension())
        registry.register(DiplomacyExtension(geo))
        registry.register(EconomicsExtension())
        registry.register(EnvironmentExtension())

        # Enable those requested (in order — dependency checked)
        for name in config.extensions_enabled:
            registry.enable(name)

        return registry

    def list_sessions(self) -> list[dict[str, Any]]:
        """List all sessions as summary dicts."""
        return [
            {
                "id": s.id,
                "name": s.name,
                "status": s.status,
                "current_generation": s.current_generation,
                "max_generations": s.max_generations,
                "population_size": len(s.engine.population),
            }
            for s in self.sessions.values()
        ]
