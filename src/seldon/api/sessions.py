"""
Session manager for simulation experiments with SQLite persistence.

Each session wraps a SimulationEngine + MetricsCollector, supporting
step-by-step execution while preserving all agents (including dead)
for family tree and lineage queries.

Sessions are auto-saved to SQLite after every mutation (create, step,
run, reset, delete).  On startup only metadata is loaded; full state
is deserialized lazily on first access.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from seldon.core.agent import Agent
from seldon.core.config import ExperimentConfig
from seldon.core.engine import SimulationEngine
from seldon.extensions.registry import ExtensionRegistry
from seldon.metrics.collector import MetricsCollector

logger = logging.getLogger(__name__)


@dataclass
class SimulationSession:
    """A running or completed simulation session."""

    id: str
    name: str
    config: ExperimentConfig
    engine: Any  # SimulationEngine or TickEngine
    collector: MetricsCollector
    status: str = "created"  # created | running | completed
    current_generation: int = 0
    max_generations: int = 0
    all_agents: dict[str, Agent] = field(default_factory=dict)


class SessionManager:
    """Manages multiple simulation sessions with optional SQLite persistence.

    Parameters
    ----------
    db_path : str | None
        Path to the SQLite database file.  ``None`` disables persistence
        (pure in-memory mode).  Default ``"data/seldon.db"``.
    """

    def __init__(self, db_path: str | None = "data/seldon.db"):
        self.sessions: dict[str, SimulationSession] = {}

        # Session index: metadata for sessions persisted but not yet loaded
        # into memory.  Keyed by session_id.
        self._session_index: dict[str, dict[str, Any]] = {}

        # Persistence store
        self._store = None
        if db_path is not None:
            from seldon.api.persistence import SessionStore
            self._store = SessionStore(db_path)
            self._load_index()

    def _load_index(self) -> None:
        """Populate _session_index from the database (metadata only)."""
        if self._store is None or not self._store.available:
            return
        for row in self._store.list_sessions():
            sid = row["id"]
            if sid not in self.sessions:
                self._session_index[sid] = row

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _persist_session(self, session: SimulationSession) -> None:
        """Save a session to the database (best-effort)."""
        if self._store is None or not self._store.available:
            return
        try:
            from seldon.api.persistence import build_state_blob

            # Include hex grid data if available
            hex_grid_data = None
            hex_grid = getattr(session.engine, "hex_grid", None)
            if hex_grid is not None:
                hex_grid_data = hex_grid.to_dict()

            blob = build_state_blob(
                all_agents=session.all_agents,
                living_agent_ids=[a.id for a in session.engine.population],
                metrics_history=session.collector.metrics_history,
                next_agent_id=session.engine._next_agent_id,
                previous_regions=session.collector._previous_regions,
                hex_grid_data=hex_grid_data,
            )
            self._store.save_session(
                session_id=session.id,
                name=session.name,
                status=session.status,
                current_generation=session.current_generation,
                max_generations=session.max_generations,
                population_size=len(session.engine.population),
                config=session.config,
                state_blob=blob,
            )
            # Keep index in sync
            self._session_index.pop(session.id, None)
        except Exception:
            logger.warning(
                "Failed to persist session %s", session.id, exc_info=True,
            )

    def _load_session_from_db(self, session_id: str) -> SimulationSession | None:
        """Fully load a session from the database into memory."""
        if self._store is None or not self._store.available:
            return None
        try:
            from seldon.api.persistence import restore_state
            record = self._store.load_session(session_id)
            if record is None:
                return None

            config = ExperimentConfig.from_json(record["config_json"])
            state = restore_state(record["state_blob"])

            # Rebuild engine + extensions
            registry = self._build_extensions(config)
            if config.tick_config.get("enabled", False):
                from seldon.core.tick_engine import TickEngine
                engine = TickEngine(config, extensions=registry)
            else:
                engine = SimulationEngine(config, extensions=registry)

            # Restore agent data
            all_agents: dict[str, Agent] = state["all_agents"]
            living_ids = set(state["living_agent_ids"])
            engine.population = [
                all_agents[aid] for aid in state["living_agent_ids"]
                if aid in all_agents
            ]
            engine._next_agent_id = state["next_agent_id"]

            # Restore hex grid if available
            if "hex_grid" in state and state["hex_grid"] is not None:
                hex_grid = getattr(engine, "_hex_grid", None)
                if hex_grid is None and hasattr(engine, "_hex_enabled"):
                    from seldon.core.hex_grid import HexGrid
                    engine._hex_grid = HexGrid.from_dict(state["hex_grid"])

            # Reseed RNG deterministically: seed + current_generation
            gen = record["current_generation"]
            seed = config.random_seed
            if seed is not None:
                engine.rng = np.random.default_rng(seed + gen)
            else:
                engine.rng = np.random.default_rng()

            # Fire on_simulation_start so extensions rebuild internal state
            for ext in engine.extensions.get_enabled():
                ext.on_simulation_start(engine.population, config)

            # Rebuild collector
            collector = MetricsCollector(config)
            collector.metrics_history = state["metrics_history"]
            collector._previous_regions = state["previous_regions"]

            session = SimulationSession(
                id=record["id"],
                name=record["name"],
                config=config,
                engine=engine,
                collector=collector,
                status=record["status"],
                current_generation=record["current_generation"],
                max_generations=record["max_generations"],
                all_agents=all_agents,
            )
            return session
        except Exception:
            logger.warning(
                "Failed to load session %s from database", session_id,
                exc_info=True,
            )
            return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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

        if config.tick_config.get("enabled", False):
            from seldon.core.tick_engine import TickEngine
            engine = TickEngine(config, extensions=registry)
        else:
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
        self._persist_session(session)
        return session

    def get_session(self, session_id: str) -> SimulationSession:
        """Get a session by ID. Lazy-loads from DB if needed.

        Raises KeyError if not found in memory or database.
        """
        if session_id in self.sessions:
            return self.sessions[session_id]

        # Try lazy-load from database
        if session_id in self._session_index or (
            self._store is not None and self._store.has_session(session_id)
        ):
            session = self._load_session_from_db(session_id)
            if session is not None:
                self.sessions[session_id] = session
                self._session_index.pop(session_id, None)
                return session

        raise KeyError(f"Session '{session_id}' not found")

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

        self._persist_session(session)
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
        if session.config.tick_config.get("enabled", False):
            from seldon.core.tick_engine import TickEngine
            engine = TickEngine(session.config, extensions=registry)
        else:
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

        self._persist_session(session)
        return session

    def delete_session(self, session_id: str) -> None:
        """Delete a session from memory and database."""
        in_memory = session_id in self.sessions
        in_index = session_id in self._session_index
        in_db = self._store is not None and self._store.has_session(session_id)

        if not in_memory and not in_index and not in_db:
            raise KeyError(f"Session '{session_id}' not found")

        self.sessions.pop(session_id, None)
        self._session_index.pop(session_id, None)

        if self._store is not None:
            self._store.delete_session(session_id)

    @staticmethod
    def _build_extensions(config: ExperimentConfig) -> ExtensionRegistry:
        """Build an ExtensionRegistry from config.extensions_enabled."""
        from seldon.extensions import (
            GeographyExtension, MigrationExtension, ResourcesExtension,
            TechnologyExtension, CultureExtension, ConflictExtension,
            SocialDynamicsExtension, DiplomacyExtension, EconomicsExtension,
            EnvironmentExtension, EpistemologyExtension,
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
        registry.register(EpistemologyExtension())

        # Enable those requested (in order — dependency checked)
        for name in config.extensions_enabled:
            registry.enable(name)

        return registry

    def list_sessions(self) -> list[dict[str, Any]]:
        """List all sessions as summary dicts (in-memory + persisted)."""
        seen: set[str] = set()
        result: list[dict[str, Any]] = []

        # In-memory sessions (authoritative for loaded sessions)
        for s in self.sessions.values():
            seen.add(s.id)
            result.append({
                "id": s.id,
                "name": s.name,
                "status": s.status,
                "current_generation": s.current_generation,
                "max_generations": s.max_generations,
                "population_size": len(s.engine.population),
            })

        # Persisted-but-not-loaded sessions from index
        for sid, meta in self._session_index.items():
            if sid not in seen:
                seen.add(sid)
                result.append({
                    "id": meta["id"],
                    "name": meta["name"],
                    "status": meta["status"],
                    "current_generation": meta["current_generation"],
                    "max_generations": meta["max_generations"],
                    "population_size": meta["population_size"],
                })

        return result

    def close(self) -> None:
        """Close the persistence store."""
        if self._store is not None:
            self._store.close()
