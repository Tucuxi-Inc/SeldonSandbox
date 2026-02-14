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
import threading
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

        # Track session IDs currently running in background threads
        self._running: set[str] = set()

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

        if session_id in self._running:
            return session  # Background run in progress, don't interfere

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

    def step_tick(self, session_id: str) -> dict[str, Any]:
        """Advance a tick-engine session by one tick (1/12 of a year).

        Returns a dict with the tick activity log for world-view visualization.
        Raises ValueError if the session does not use the tick engine.
        """
        session = self.get_session(session_id)

        if not hasattr(session.engine, "_run_single_tick"):
            raise ValueError(
                "Tick stepping requires tick engine (tick_config.enabled=True)"
            )

        if session.status == "completed":
            return self._build_tick_response(session)

        session.status = "running"

        # Store agents before tick (some may die at year-end)
        for agent in session.engine.population:
            session.all_agents[agent.id] = agent

        # Run one tick
        tick_log = session.engine._run_single_tick()

        # Store new agents (children born this tick)
        for agent in session.engine.population:
            session.all_agents[agent.id] = agent

        # Check if year completed
        year_complete = getattr(tick_log, "year_complete", False)
        if year_complete:
            # Collect metrics for the completed year
            latest_snapshot = session.engine.history[-1]
            session.collector.collect(session.engine.population, latest_snapshot)
            session.current_generation = tick_log.year + 1

            if session.current_generation >= session.max_generations:
                session.status = "completed"

            # Auto-save at year boundaries only
            self._persist_session(session)

        return self._build_tick_response(session, tick_log)

    def _build_tick_response(
        self, session: "SimulationSession",
        tick_log: Any = None,
    ) -> dict[str, Any]:
        """Build the JSON-serializable response for a tick step."""
        engine = session.engine

        if tick_log is None:
            # No tick log — return current state without advancing
            buffer = getattr(engine, "_tick_log_buffer", None)
            if buffer and len(buffer) > 0:
                tick_log = buffer[-1]
            else:
                return {
                    "enabled": True,
                    "year": getattr(engine, "_single_tick_year", 0),
                    "tick_in_year": getattr(engine, "_single_tick_in_year", 0),
                    "global_tick": getattr(engine, "_global_tick", 0),
                    "season": "",
                    "population_count": sum(
                        1 for a in engine.population if a.is_alive
                    ),
                    "year_complete": False,
                    "session_status": session.status,
                    "current_generation": session.current_generation,
                    "agent_activities": {},
                    "events": [],
                    "agent_names": {},
                }

        # Serialize agent activities
        activities: dict[str, dict[str, Any]] = {}
        for aid, ata in tick_log.agent_activities.items():
            activities[aid] = {
                "agent_id": ata.agent_id,
                "location": list(ata.location) if ata.location else None,
                "previous_location": (
                    list(ata.previous_location) if ata.previous_location else None
                ),
                "activity": ata.activity,
                "activity_need": ata.activity_need,
                "life_phase": ata.life_phase,
                "processing_region": ata.processing_region,
                "needs_snapshot": ata.needs_snapshot,
                "health": ata.health,
                "suffering": ata.suffering,
                "is_pregnant": ata.is_pregnant,
            }

        # Build agent names map (for event display)
        agent_names: dict[str, str] = {}
        for aid in tick_log.agent_activities:
            agent = session.all_agents.get(aid)
            if agent:
                agent_names[aid] = agent.name
        # Also include agents mentioned in events
        for evt in tick_log.events:
            for key in ("agent_id", "child_id", "mother_id", "father_id"):
                aid = evt.get(key)
                if aid and aid not in agent_names:
                    agent = session.all_agents.get(aid)
                    if agent:
                        agent_names[aid] = agent.name

        return {
            "enabled": True,
            "year": tick_log.year,
            "tick_in_year": tick_log.tick_in_year,
            "global_tick": tick_log.global_tick,
            "season": tick_log.season,
            "population_count": tick_log.population_count,
            "year_complete": getattr(tick_log, "year_complete", False),
            "session_status": session.status,
            "current_generation": session.current_generation,
            "agent_activities": activities,
            "events": tick_log.events,
            "agent_names": agent_names,
        }

    def run_full(self, session_id: str) -> SimulationSession:
        """Run a session to completion."""
        session = self.get_session(session_id)
        remaining = session.max_generations - session.current_generation
        if remaining > 0:
            self.step(session_id, remaining)
        return session

    def run_full_async(self, session_id: str) -> SimulationSession:
        """Start running a session in a background thread."""
        session = self.get_session(session_id)
        if session_id in self._running:
            return session  # Already running, no-op
        if session.status == "completed":
            return session

        session.status = "running"
        self._running.add(session_id)

        def _worker():
            try:
                remaining = session.max_generations - session.current_generation
                if remaining > 0:
                    self._step_with_periodic_save(session_id, remaining)
            except Exception:
                logger.exception("Background run failed for %s", session_id)
                session.status = "error"
            finally:
                self._running.discard(session_id)

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        return session

    def _step_with_periodic_save(
        self, session_id: str, n: int,
    ) -> SimulationSession:
        """Step N generations, persisting every 5 generations."""
        session = self.get_session(session_id)
        if session.status == "completed":
            return session
        session.status = "running"

        for i in range(n):
            if session.current_generation >= session.max_generations:
                session.status = "completed"
                break

            for agent in session.engine.population:
                session.all_agents[agent.id] = agent

            snapshot = session.engine._run_generation(session.current_generation)
            session.engine.history.append(snapshot)
            session.collector.collect(session.engine.population, snapshot)

            for agent in session.engine.population:
                session.all_agents[agent.id] = agent

            session.current_generation += 1

            if session.current_generation >= session.max_generations:
                session.status = "completed"

            # Periodic save every 5 generations
            if (i + 1) % 5 == 0:
                self._persist_session(session)

        self._persist_session(session)
        return session

    def is_running(self, session_id: str) -> bool:
        """Check if a session is running in a background thread."""
        return session_id in self._running

    def reset_session(self, session_id: str) -> SimulationSession:
        """Reset a session to generation 0."""
        if session_id in self._running:
            raise ValueError(f"Session '{session_id}' is currently running")
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
            EnvironmentExtension, EpistemologyExtension, InnerLifeExtension,
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
        registry.register(InnerLifeExtension())

        # Enable those requested — sort so dependencies come first
        dep_order = ['geography', 'migration', 'resources', 'technology',
                     'culture', 'conflict', 'social_dynamics', 'diplomacy',
                     'economics', 'environment', 'epistemology', 'inner_life']
        ordered = sorted(
            config.extensions_enabled,
            key=lambda n: dep_order.index(n) if n in dep_order else len(dep_order),
        )
        for name in ordered:
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

    def clone_session(
        self,
        source_id: str,
        config_overrides: dict[str, Any] | None = None,
        name: str | None = None,
    ) -> SimulationSession:
        """Clone a session: copy current state, optionally override config.

        The clone gets a new session ID and reseeded RNG so it will
        diverge from the source when stepped forward.
        """
        source = self.get_session(source_id)

        # Serialize source state
        from seldon.api.persistence import build_state_blob, restore_state

        hex_grid_data = None
        hex_grid = getattr(source.engine, "hex_grid", None)
        if hex_grid is not None:
            hex_grid_data = hex_grid.to_dict()

        blob = build_state_blob(
            all_agents=source.all_agents,
            living_agent_ids=[a.id for a in source.engine.population],
            metrics_history=source.collector.metrics_history,
            next_agent_id=source.engine._next_agent_id,
            previous_regions=source.collector._previous_regions,
            hex_grid_data=hex_grid_data,
        )

        # Apply config overrides
        config = source.config
        if config_overrides:
            config_dict = config.to_dict()
            config_dict.update(config_overrides)
            config = ExperimentConfig.from_dict(config_dict)

        # Restore into new engine
        state = restore_state(blob)
        registry = self._build_extensions(config)

        if config.tick_config.get("enabled", False):
            from seldon.core.tick_engine import TickEngine
            engine = TickEngine(config, extensions=registry)
        else:
            engine = SimulationEngine(config, extensions=registry)

        all_agents: dict[str, Agent] = state["all_agents"]
        engine.population = [
            all_agents[aid] for aid in state["living_agent_ids"]
            if aid in all_agents
        ]
        engine._next_agent_id = state["next_agent_id"]

        # Restore hex grid if available
        if "hex_grid" in state and state["hex_grid"] is not None:
            hex_grid_obj = getattr(engine, "_hex_grid", None)
            if hex_grid_obj is None and hasattr(engine, "_hex_enabled"):
                from seldon.core.hex_grid import HexGrid
                engine._hex_grid = HexGrid.from_dict(state["hex_grid"])

        # Reseed RNG for divergent future
        gen = source.current_generation
        seed = config.random_seed
        if seed is not None:
            # Use a different seed offset so clone diverges
            engine.rng = np.random.default_rng(seed + gen + 99999)
        else:
            engine.rng = np.random.default_rng()

        # Fire on_simulation_start for extensions
        for ext in engine.extensions.get_enabled():
            ext.on_simulation_start(engine.population, config)

        # Rebuild collector
        collector = MetricsCollector(config)
        collector.metrics_history = state["metrics_history"]
        collector._previous_regions = state["previous_regions"]

        new_id = uuid.uuid4().hex[:8]
        clone_name = name or f"{source.name} (fork)"

        session = SimulationSession(
            id=new_id,
            name=clone_name,
            config=config,
            engine=engine,
            collector=collector,
            status=source.status if source.status != "running" else "created",
            current_generation=source.current_generation,
            max_generations=source.max_generations,
            all_agents=all_agents,
        )

        self.sessions[new_id] = session
        self._persist_session(session)
        return session

    def close(self) -> None:
        """Close the persistence store."""
        if self._store is not None:
            self._store.close()
