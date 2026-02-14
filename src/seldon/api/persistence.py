"""
SQLite-backed session persistence for the Seldon Sandbox.

Stores session metadata in columns for fast listing, and full session
state (agents, metrics) as a zlib-compressed JSON blob.  Lazy loading:
only metadata is read on startup; full state is deserialized on demand.

Persistence failures are logged as warnings and never crash the app —
the system degrades gracefully to in-memory-only operation.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import zlib
from datetime import datetime, timezone
from typing import Any

import numpy as np

from seldon.core.agent import Agent
from seldon.core.config import ExperimentConfig
from seldon.core.processing import ProcessingRegion
from seldon.metrics.collector import GenerationMetrics

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def _json_fallback(obj: Any) -> Any:
    """Handle numpy types and other non-JSON-serializable objects."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, ProcessingRegion):
        return obj.value
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# ---------------------------------------------------------------------------
# Agent serialization
# ---------------------------------------------------------------------------

def serialize_agent(agent: Agent) -> dict[str, Any]:
    """Convert an Agent to a JSON-safe dict (full fidelity)."""
    return {
        # Identity
        "id": agent.id,
        "name": agent.name,
        "age": int(agent.age),
        "generation": int(agent.generation),
        "birth_order": int(agent.birth_order),
        # Traits (numpy → list)
        "traits": agent.traits.tolist(),
        "traits_at_birth": agent.traits_at_birth.tolist(),
        # Processing
        "processing_region": agent.processing_region.value,
        # State
        "suffering": float(agent.suffering),
        "burnout_level": float(agent.burnout_level),
        "is_alive": agent.is_alive,
        # Relationships
        "partner_id": agent.partner_id,
        "parent1_id": agent.parent1_id,
        "parent2_id": agent.parent2_id,
        "children_ids": list(agent.children_ids),
        "relationship_status": agent.relationship_status,
        # Social
        "social_bonds": {k: float(v) for k, v in agent.social_bonds.items()},
        # Cognitive
        "dominant_voice": agent.dominant_voice,
        # History (numpy arrays → lists)
        "trait_history": [th.tolist() for th in agent.trait_history],
        "region_history": [r.value for r in agent.region_history],
        "contribution_history": [float(c) for c in agent.contribution_history],
        "suffering_history": [float(s) for s in agent.suffering_history],
        # Lore
        "personal_memories": agent.personal_memories,
        "inherited_lore": agent.inherited_lore,
        # Decisions
        "decision_history": agent.decision_history,
        # Social hierarchy
        "social_status": float(agent.social_status),
        "mentor_id": agent.mentor_id,
        "mentee_ids": list(agent.mentee_ids),
        "social_role": agent.social_role,
        "influence_score": float(agent.influence_score),
        # Genetics
        "genome": {k: list(v) for k, v in agent.genome.items()} if agent.genome else {},
        "epigenetic_state": dict(agent.epigenetic_state) if agent.epigenetic_state else {},
        "genetic_lineage": _sanitize_lineage(agent.genetic_lineage) if agent.genetic_lineage else {},
        # Community
        "community_id": agent.community_id,
        # Economics
        "wealth": float(agent.wealth),
        "occupation": agent.occupation,
        "trade_history": agent.trade_history,
        # Extensions
        "location_id": agent.location_id,
        "resource_holdings": {k: float(v) for k, v in agent.resource_holdings.items()},
        "cultural_memes": list(agent.cultural_memes),
        "skills": {k: float(v) for k, v in agent.skills.items()},
        "extension_data": agent.extension_data,
        # Outsider
        "is_outsider": agent.is_outsider,
        "outsider_origin": agent.outsider_origin,
        "injection_generation": int(agent.injection_generation) if agent.injection_generation is not None else None,
        # Demographics
        "gender": agent.gender,
        # Fertility
        "last_birth_generation": int(agent.last_birth_generation) if agent.last_birth_generation is not None else None,
        # Tick-based / Needs (Phase A)
        "_age_ticks": int(agent._age_ticks),
        "life_phase": agent.life_phase,
        "location": list(agent.location) if agent.location else None,
        "needs": dict(agent.needs),
        "health": float(agent.health),
        "needs_history": agent.needs_history,
        "health_history": [float(h) for h in agent.health_history],
    }


def deserialize_agent(d: dict[str, Any]) -> Agent:
    """Reconstruct an Agent from a serialized dict."""
    agent = Agent(
        id=d["id"],
        name=d["name"],
        age=d["age"],
        generation=d["generation"],
        birth_order=d["birth_order"],
        traits=np.array(d["traits"]),
        traits_at_birth=np.array(d["traits_at_birth"]),
    )
    agent.processing_region = ProcessingRegion(d["processing_region"])
    agent.suffering = d["suffering"]
    agent.burnout_level = d["burnout_level"]
    agent.is_alive = d["is_alive"]
    agent.partner_id = d.get("partner_id")
    agent.parent1_id = d.get("parent1_id")
    agent.parent2_id = d.get("parent2_id")
    agent.children_ids = d.get("children_ids", [])
    agent.relationship_status = d.get("relationship_status", "single")
    agent.social_bonds = d.get("social_bonds", {})
    agent.dominant_voice = d.get("dominant_voice")
    agent.trait_history = [np.array(th) for th in d.get("trait_history", [])]
    agent.region_history = [ProcessingRegion(r) for r in d.get("region_history", [])]
    agent.contribution_history = d.get("contribution_history", [])
    agent.suffering_history = d.get("suffering_history", [])
    agent.personal_memories = d.get("personal_memories", [])
    agent.inherited_lore = d.get("inherited_lore", [])
    agent.decision_history = d.get("decision_history", [])
    agent.social_status = d.get("social_status", 0.0)
    agent.mentor_id = d.get("mentor_id")
    agent.mentee_ids = d.get("mentee_ids", [])
    agent.social_role = d.get("social_role")
    agent.influence_score = d.get("influence_score", 0.0)
    agent.genome = {k: tuple(v) for k, v in d.get("genome", {}).items()}
    agent.epigenetic_state = d.get("epigenetic_state", {})
    agent.genetic_lineage = _restore_lineage(d.get("genetic_lineage", {}))
    agent.community_id = d.get("community_id")
    agent.wealth = d.get("wealth", 0.0)
    agent.occupation = d.get("occupation")
    agent.trade_history = d.get("trade_history", [])
    agent.location_id = d.get("location_id")
    agent.resource_holdings = d.get("resource_holdings", {})
    agent.cultural_memes = d.get("cultural_memes", [])
    agent.skills = d.get("skills", {})
    agent.extension_data = d.get("extension_data", {})
    agent.is_outsider = d.get("is_outsider", False)
    agent.outsider_origin = d.get("outsider_origin")
    agent.injection_generation = d.get("injection_generation")
    agent.gender = d.get("gender")
    agent.last_birth_generation = d.get("last_birth_generation")
    # Tick-based / Needs (Phase A)
    agent._age_ticks = d.get("_age_ticks", 0)
    agent.life_phase = d.get("life_phase")
    loc = d.get("location")
    agent.location = tuple(loc) if loc else None
    agent.needs = d.get("needs", {
        "hunger": 1.0, "thirst": 1.0, "shelter": 1.0,
        "safety": 1.0, "warmth": 1.0, "rest": 1.0,
    })
    agent.health = d.get("health", 1.0)
    agent.needs_history = d.get("needs_history", [])
    agent.health_history = d.get("health_history", [])
    return agent


def _sanitize_lineage(lineage: dict) -> dict:
    """Convert genetic lineage tuples to lists for JSON."""
    result = {}
    for k, v in lineage.items():
        if isinstance(v, tuple):
            result[k] = list(v)
        elif isinstance(v, dict):
            result[k] = _sanitize_lineage(v)
        else:
            result[k] = v
    return result


def _restore_lineage(lineage: dict) -> dict:
    """Restore lineage — lists that were tuples stay as lists (acceptable)."""
    return lineage


# ---------------------------------------------------------------------------
# GenerationMetrics serialization
# ---------------------------------------------------------------------------

def serialize_metrics(m: GenerationMetrics) -> dict[str, Any]:
    """Convert a GenerationMetrics to a JSON-safe dict."""
    return {
        "generation": m.generation,
        "population_size": m.population_size,
        "births": m.births,
        "deaths": m.deaths,
        "breakthroughs": m.breakthroughs,
        "pairs_formed": m.pairs_formed,
        "trait_means": m.trait_means.tolist() if isinstance(m.trait_means, np.ndarray) else m.trait_means,
        "trait_stds": m.trait_stds.tolist() if isinstance(m.trait_stds, np.ndarray) else m.trait_stds,
        "trait_entropy": float(m.trait_entropy),
        "region_counts": m.region_counts,
        "region_fractions": {k: float(v) for k, v in m.region_fractions.items()},
        "region_transitions": m.region_transitions,
        "total_contribution": float(m.total_contribution),
        "mean_contribution": float(m.mean_contribution),
        "max_contribution": float(m.max_contribution),
        "mean_suffering": float(m.mean_suffering),
        "suffering_by_region": {k: float(v) for k, v in m.suffering_by_region.items()},
        "mean_age": float(m.mean_age),
        "age_distribution": m.age_distribution,
        "birth_order_counts": {str(k): v for k, v in m.birth_order_counts.items()},
        "total_memories": m.total_memories,
        "societal_memories": m.societal_memories,
        "myths_count": m.myths_count,
        "outsider_count": m.outsider_count,
        "outsider_descendant_count": m.outsider_descendant_count,
        "dissolutions": m.dissolutions,
        "infidelity_events": m.infidelity_events,
        "outsiders_injected": m.outsiders_injected,
        "dominant_voice_counts": m.dominant_voice_counts,
        "extension_metrics": m.extension_metrics,
    }


def deserialize_metrics(d: dict[str, Any]) -> GenerationMetrics:
    """Reconstruct a GenerationMetrics from a serialized dict."""
    # birth_order_counts keys need int conversion
    birth_order_counts = {int(k): v for k, v in d.get("birth_order_counts", {}).items()}
    return GenerationMetrics(
        generation=d["generation"],
        population_size=d["population_size"],
        births=d["births"],
        deaths=d["deaths"],
        breakthroughs=d["breakthroughs"],
        pairs_formed=d["pairs_formed"],
        trait_means=np.array(d["trait_means"]),
        trait_stds=np.array(d["trait_stds"]),
        trait_entropy=d["trait_entropy"],
        region_counts=d["region_counts"],
        region_fractions=d["region_fractions"],
        region_transitions=d["region_transitions"],
        total_contribution=d["total_contribution"],
        mean_contribution=d["mean_contribution"],
        max_contribution=d["max_contribution"],
        mean_suffering=d["mean_suffering"],
        suffering_by_region=d["suffering_by_region"],
        mean_age=d["mean_age"],
        age_distribution=d["age_distribution"],
        birth_order_counts=birth_order_counts,
        total_memories=d["total_memories"],
        societal_memories=d["societal_memories"],
        myths_count=d["myths_count"],
        outsider_count=d["outsider_count"],
        outsider_descendant_count=d["outsider_descendant_count"],
        dissolutions=d["dissolutions"],
        infidelity_events=d["infidelity_events"],
        outsiders_injected=d["outsiders_injected"],
        dominant_voice_counts=d.get("dominant_voice_counts", {}),
        extension_metrics=d.get("extension_metrics", {}),
    )


# ---------------------------------------------------------------------------
# State blob compress / decompress
# ---------------------------------------------------------------------------

def compress_state(state: dict[str, Any]) -> bytes:
    """Serialize state dict to zlib-compressed JSON bytes."""
    json_bytes = json.dumps(state, default=_json_fallback).encode("utf-8")
    return zlib.compress(json_bytes, level=6)


def decompress_state(blob: bytes) -> dict[str, Any]:
    """Decompress zlib blob and parse JSON."""
    json_bytes = zlib.decompress(blob)
    return json.loads(json_bytes.decode("utf-8"))


def build_state_blob(
    all_agents: dict[str, Agent],
    living_agent_ids: list[str],
    metrics_history: list[GenerationMetrics],
    next_agent_id: int,
    previous_regions: dict[str, str],
    hex_grid_data: dict[str, Any] | None = None,
) -> bytes:
    """Build and compress the full session state blob."""
    state = {
        "all_agents": {
            aid: serialize_agent(agent) for aid, agent in all_agents.items()
        },
        "living_agent_ids": living_agent_ids,
        "metrics_history": [serialize_metrics(m) for m in metrics_history],
        "next_agent_id": next_agent_id,
        "previous_regions": previous_regions,
    }
    if hex_grid_data is not None:
        state["hex_grid"] = hex_grid_data
    return compress_state(state)


def restore_state(blob: bytes) -> dict[str, Any]:
    """Decompress and restore state from blob.

    Returns dict with keys:
        all_agents: dict[str, Agent]
        living_agent_ids: list[str]
        metrics_history: list[GenerationMetrics]
        next_agent_id: int
        previous_regions: dict[str, str]
        hex_grid: dict | None  (if present)
    """
    raw = decompress_state(blob)
    result = {
        "all_agents": {
            aid: deserialize_agent(ad) for aid, ad in raw["all_agents"].items()
        },
        "living_agent_ids": raw["living_agent_ids"],
        "metrics_history": [deserialize_metrics(md) for md in raw["metrics_history"]],
        "next_agent_id": raw["next_agent_id"],
        "previous_regions": raw["previous_regions"],
    }
    if "hex_grid" in raw:
        result["hex_grid"] = raw["hex_grid"]
    return result


# ---------------------------------------------------------------------------
# SQLite SessionStore
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'created',
    current_generation INTEGER NOT NULL DEFAULT 0,
    max_generations INTEGER NOT NULL DEFAULT 0,
    population_size INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    config_json TEXT NOT NULL,
    state_blob BLOB
);
"""


class SessionStore:
    """SQLite-backed storage for simulation sessions.

    Thread-safety: uses ``check_same_thread=False`` so FastAPI's
    thread pool can access it.  Writes are serialized by SQLite's
    internal locking.
    """

    def __init__(self, db_path: str = "data/seldon.db"):
        self.db_path = db_path
        self._conn: sqlite3.Connection | None = None
        self._init_db()

    def _init_db(self) -> None:
        """Open connection and create table if needed."""
        try:
            self._conn = sqlite3.connect(
                self.db_path, check_same_thread=False,
            )
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute(_SCHEMA)
            self._conn.commit()
        except Exception:
            logger.warning(
                "Failed to open SQLite database at %s — "
                "falling back to in-memory only",
                self.db_path,
                exc_info=True,
            )
            self._conn = None

    @property
    def available(self) -> bool:
        """True if the database connection is open."""
        return self._conn is not None

    # ---- Write operations ----

    def save_session(
        self,
        session_id: str,
        name: str,
        status: str,
        current_generation: int,
        max_generations: int,
        population_size: int,
        config: ExperimentConfig,
        state_blob: bytes,
    ) -> None:
        """Insert or replace a full session record."""
        if not self.available:
            return
        now = datetime.now(timezone.utc).isoformat()
        try:
            self._conn.execute(  # type: ignore[union-attr]
                """
                INSERT INTO sessions
                    (id, name, status, current_generation, max_generations,
                     population_size, created_at, updated_at, config_json, state_blob)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name = excluded.name,
                    status = excluded.status,
                    current_generation = excluded.current_generation,
                    max_generations = excluded.max_generations,
                    population_size = excluded.population_size,
                    updated_at = excluded.updated_at,
                    config_json = excluded.config_json,
                    state_blob = excluded.state_blob
                """,
                (
                    session_id, name, status,
                    int(current_generation), int(max_generations), int(population_size),
                    now, now,
                    config.to_json(),
                    state_blob,
                ),
            )
            self._conn.commit()  # type: ignore[union-attr]
        except Exception:
            logger.warning(
                "Failed to save session %s to database", session_id,
                exc_info=True,
            )

    def delete_session(self, session_id: str) -> None:
        """Remove a session from the database."""
        if not self.available:
            return
        try:
            self._conn.execute(  # type: ignore[union-attr]
                "DELETE FROM sessions WHERE id = ?", (session_id,),
            )
            self._conn.commit()  # type: ignore[union-attr]
        except Exception:
            logger.warning(
                "Failed to delete session %s from database", session_id,
                exc_info=True,
            )

    # ---- Read operations ----

    def list_sessions(self) -> list[dict[str, Any]]:
        """Return metadata for all persisted sessions (no state blob)."""
        if not self.available:
            return []
        try:
            cur = self._conn.execute(  # type: ignore[union-attr]
                """
                SELECT id, name, status, current_generation,
                       max_generations, population_size, created_at, updated_at
                FROM sessions
                ORDER BY created_at DESC
                """,
            )
            rows = cur.fetchall()
            return [
                {
                    "id": r[0],
                    "name": r[1],
                    "status": r[2],
                    "current_generation": r[3],
                    "max_generations": r[4],
                    "population_size": r[5],
                    "created_at": r[6],
                    "updated_at": r[7],
                }
                for r in rows
            ]
        except Exception:
            logger.warning("Failed to list sessions from database", exc_info=True)
            return []

    def load_session(self, session_id: str) -> dict[str, Any] | None:
        """Load a full session record (metadata + config + state blob).

        Returns ``None`` if not found or on error.
        """
        if not self.available:
            return None
        try:
            cur = self._conn.execute(  # type: ignore[union-attr]
                """
                SELECT id, name, status, current_generation,
                       max_generations, population_size,
                       config_json, state_blob
                FROM sessions WHERE id = ?
                """,
                (session_id,),
            )
            row = cur.fetchone()
            if row is None:
                return None
            return {
                "id": row[0],
                "name": row[1],
                "status": row[2],
                "current_generation": row[3],
                "max_generations": row[4],
                "population_size": row[5],
                "config_json": row[6],
                "state_blob": row[7],
            }
        except Exception:
            logger.warning(
                "Failed to load session %s from database", session_id,
                exc_info=True,
            )
            return None

    def has_session(self, session_id: str) -> bool:
        """Check if a session exists in the database."""
        if not self.available:
            return False
        try:
            cur = self._conn.execute(  # type: ignore[union-attr]
                "SELECT 1 FROM sessions WHERE id = ?", (session_id,),
            )
            return cur.fetchone() is not None
        except Exception:
            return False

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None


# ---------------------------------------------------------------------------
# NarrativeCache — stores generated biographies, chronicles, etc.
# ---------------------------------------------------------------------------

_NARRATIVE_SCHEMA = """
CREATE TABLE IF NOT EXISTS narrative_cache (
    session_id TEXT NOT NULL,
    cache_type TEXT NOT NULL,
    cache_key TEXT NOT NULL,
    content TEXT NOT NULL,
    model TEXT,
    created_at TEXT NOT NULL,
    UNIQUE(session_id, cache_type, cache_key)
);
"""


class NarrativeCache:
    """Caches LLM-generated narratives in SQLite or in-memory.

    Falls back to a plain dict when no database is available.
    """

    def __init__(self, db_path: str | None = None):
        self._conn: sqlite3.Connection | None = None
        self._memory: dict[str, str] = {}
        if db_path is not None:
            try:
                self._conn = sqlite3.connect(db_path, check_same_thread=False)
                self._conn.execute(_NARRATIVE_SCHEMA)
                self._conn.commit()
            except Exception:
                logger.warning("NarrativeCache: failed to open DB, using in-memory", exc_info=True)
                self._conn = None

    def _key(self, session_id: str, cache_type: str, cache_key: str) -> str:
        return f"{session_id}:{cache_type}:{cache_key}"

    def get(self, session_id: str, cache_type: str, cache_key: str) -> str | None:
        """Retrieve a cached narrative. Returns None if not found."""
        if self._conn is not None:
            try:
                cur = self._conn.execute(
                    "SELECT content FROM narrative_cache WHERE session_id=? AND cache_type=? AND cache_key=?",
                    (session_id, cache_type, cache_key),
                )
                row = cur.fetchone()
                return row[0] if row else None
            except Exception:
                pass
        return self._memory.get(self._key(session_id, cache_type, cache_key))

    def put(
        self, session_id: str, cache_type: str, cache_key: str,
        content: str, model: str | None = None,
    ) -> None:
        """Store a narrative in the cache."""
        if self._conn is not None:
            try:
                now = datetime.now(timezone.utc).isoformat()
                self._conn.execute(
                    """INSERT INTO narrative_cache (session_id, cache_type, cache_key, content, model, created_at)
                       VALUES (?, ?, ?, ?, ?, ?)
                       ON CONFLICT(session_id, cache_type, cache_key) DO UPDATE SET
                           content=excluded.content, model=excluded.model, created_at=excluded.created_at""",
                    (session_id, cache_type, cache_key, content, model, now),
                )
                self._conn.commit()
                return
            except Exception:
                pass
        self._memory[self._key(session_id, cache_type, cache_key)] = content

    def invalidate_session(self, session_id: str) -> None:
        """Remove all cached narratives for a session."""
        if self._conn is not None:
            try:
                self._conn.execute(
                    "DELETE FROM narrative_cache WHERE session_id=?", (session_id,),
                )
                self._conn.commit()
            except Exception:
                pass
        keys_to_remove = [k for k in self._memory if k.startswith(f"{session_id}:")]
        for k in keys_to_remove:
            del self._memory[k]

    def invalidate(self, session_id: str, cache_type: str, cache_key: str) -> None:
        """Remove a specific cached narrative."""
        if self._conn is not None:
            try:
                self._conn.execute(
                    "DELETE FROM narrative_cache WHERE session_id=? AND cache_type=? AND cache_key=?",
                    (session_id, cache_type, cache_key),
                )
                self._conn.commit()
            except Exception:
                pass
        self._memory.pop(self._key(session_id, cache_type, cache_key), None)
