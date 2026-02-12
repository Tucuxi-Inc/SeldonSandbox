"""
Inner Life Extension — phenomenological layer for agent subjective experience.

Wires the ExperientialEngine into engine hooks:
  - on_simulation_start: init experiential state, seed mood from traits
  - on_generation_start: detect events, encode experiences, decay assertoric, update mood
  - on_generation_end: compute PQ, compute experiential drift, prune, cache metrics
  - on_agent_created: inherit top-k strongest experiences from each parent
  - modify_decision: 60/40 experiential modulation via recall_similar
  - modify_mortality: low PQ → slightly higher mortality; high PQ → lower
  - modify_attraction: similar mood vectors → slight attraction bonus
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from seldon.core.experiential import EXPERIENCE_DIM, ExperientialEngine
from seldon.extensions.base import SimulationExtension

if TYPE_CHECKING:
    from seldon.core.agent import Agent
    from seldon.core.config import ExperimentConfig


class InnerLifeExtension(SimulationExtension):
    """Phenomenological layer — subjective experience, recall, and drift."""

    def __init__(self) -> None:
        self._engine: ExperientialEngine | None = None
        self._rng: np.random.Generator | None = None
        self._cached_metrics: dict[str, Any] = {}

    @property
    def name(self) -> str:
        return "inner_life"

    @property
    def description(self) -> str:
        return "Experiential encoding, phenomenal quality, and experience-driven drift"

    def get_default_config(self) -> dict[str, Any]:
        return {}

    @property
    def experiential_engine(self) -> ExperientialEngine | None:
        return self._engine

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    def on_simulation_start(
        self, population: list[Agent], config: ExperimentConfig,
    ) -> None:
        """Initialize ExperientialEngine and seed mood from traits."""
        self._engine = ExperientialEngine(config)
        self._rng = np.random.default_rng(config.random_seed)

        if not config.inner_life_config.get("enabled", True):
            return

        for agent in population:
            self._engine.init_state(agent)
            self._engine.seed_mood_from_traits(agent)

    def on_generation_start(
        self, generation: int, population: list[Agent],
        config: ExperimentConfig,
    ) -> None:
        """Detect events, encode experiences, decay assertoric, update mood."""
        if self._engine is None:
            self._engine = ExperientialEngine(config)
            self._rng = np.random.default_rng(config.random_seed)

        if not config.inner_life_config.get("enabled", True):
            return

        eng = self._engine
        rng = self._rng

        for agent in population:
            if not agent.is_alive:
                continue

            state = agent.extension_data.get("inner_life")
            if state is None:
                eng.init_state(agent)
                eng.seed_mood_from_traits(agent)
                state = agent.extension_data["inner_life"]

            prev = state.get("_prev_state", {})

            # --- Detect events ---
            events: list[tuple[str, dict[str, Any] | None]] = []

            # Breakthrough: contribution above threshold (check contribution history)
            if agent.contribution_history:
                last_contrib = agent.contribution_history[-1]
                threshold = config.contribution_config.get(
                    "breakthrough_threshold", 0.95,
                )
                if last_contrib >= threshold:
                    events.append(("breakthrough", None))

            # Deep suffering: suffering > 0.6
            if agent.suffering > 0.6:
                ctx = {"processing_region": agent.processing_region}
                events.append(("deep_suffering", ctx))

            # Pair formed
            if agent.partner_id is not None and prev.get("partner_id") is None:
                events.append(("pair_formed", None))

            # Pair dissolved
            if agent.partner_id is None and prev.get("partner_id") is not None:
                # Check if partner died (bereavement) or dissolved
                old_partner_id = prev.get("partner_id")
                if old_partner_id and not self._is_partner_alive(
                    old_partner_id, population,
                ):
                    events.append(("bereavement", None))
                else:
                    events.append(("pair_dissolved", None))

            # Child born
            current_children = len(agent.children_ids)
            prev_children = prev.get("children_count", 0)
            if current_children > prev_children:
                events.append(("child_born", None))

            # Status change (if social hierarchy active)
            status = getattr(agent, "social_status", None)
            if status is not None:
                prev_status = prev.get("social_status")
                if prev_status is not None and abs(status - prev_status) > 0.2:
                    events.append(("status_change", None))

            # Migration (location changed)
            location = getattr(agent, "location", None)
            if location is not None:
                prev_loc = prev.get("location")
                if prev_loc is not None and location != prev_loc:
                    events.append(("migration", None))

            # Always encode routine if no other events
            if not events:
                events.append(("routine", None))

            # Encode all detected events
            for event_type, ctx in events:
                eng.encode_experience(agent, event_type, generation, rng, ctx)

            # Decay assertoric strength
            eng.decay_assertoric_strength(agent)

            # Update mood
            eng.update_mood(agent)

            # Update previous state
            state["_prev_state"] = {
                "partner_id": agent.partner_id,
                "children_count": len(agent.children_ids),
                "suffering": agent.suffering,
                "social_status": getattr(agent, "social_status", None),
                "location": getattr(agent, "location", None),
            }

    def on_generation_end(
        self, generation: int, population: list[Agent],
        config: ExperimentConfig,
    ) -> None:
        """Compute PQ, experiential drift, prune, cache metrics."""
        if self._engine is None or not config.inner_life_config.get("enabled", True):
            return

        eng = self._engine

        for agent in population:
            if not agent.is_alive:
                continue

            state = agent.extension_data.get("inner_life")
            if state is None:
                continue

            # Compute phenomenal quality
            eng.compute_phenomenal_quality(agent)

            # Compute and apply experiential drift
            drift = eng.compute_experiential_drift(agent)
            if drift:
                eng.apply_drift(agent, drift)

            # Prune old experiences
            eng.prune_experiences(agent)

        # Cache metrics
        self._cached_metrics = eng.get_metrics(population)

    def on_agent_created(
        self, agent: Agent, parents: tuple[Agent, Agent],
        config: ExperimentConfig,
    ) -> None:
        """Inherit top-k strongest experiences from each parent."""
        if self._engine is None or not config.inner_life_config.get("enabled", True):
            return

        eng = self._engine
        eng.init_state(agent)
        eng.seed_mood_from_traits(agent)
        eng.inherit_experiences(parents[0], parents[1], agent)

    # ------------------------------------------------------------------
    # Modifier hooks
    # ------------------------------------------------------------------

    def modify_decision(
        self, agent: Agent, context: str, utilities: dict[str, float],
        config: ExperimentConfig,
    ) -> dict[str, float]:
        """Apply experiential recall to modify decision utilities (60/40 split)."""
        if self._engine is None or not config.inner_life_config.get("enabled", True):
            return utilities
        return self._engine.compute_experiential_modulation(agent, context, utilities)

    def modify_mortality(
        self, agent: Agent, base_rate: float,
        config: ExperimentConfig,
    ) -> float:
        """Low PQ → slightly higher mortality (despair); high PQ → lower (will to live)."""
        if self._engine is None or not config.inner_life_config.get("enabled", True):
            return base_rate

        state = agent.extension_data.get("inner_life")
        if state is None:
            return base_rate

        pq = state.get("phenomenal_quality", 0.5)
        scale = config.inner_life_config.get("pq_mortality_scale", 0.03)

        # PQ 0.5 = neutral, <0.5 = increased mortality, >0.5 = decreased
        mortality_modifier = (0.5 - pq) * scale
        return max(0.0, base_rate + mortality_modifier)

    def modify_attraction(
        self, agent1: Agent, agent2: Agent, base_score: float,
        config: ExperimentConfig,
    ) -> float:
        """Similar mood vectors → slight attraction bonus."""
        if self._engine is None or not config.inner_life_config.get("enabled", True):
            return base_score

        s1 = agent1.extension_data.get("inner_life")
        s2 = agent2.extension_data.get("inner_life")
        if s1 is None or s2 is None:
            return base_score

        mood1 = np.array(s1.get("mood", [0.0] * EXPERIENCE_DIM))
        mood2 = np.array(s2.get("mood", [0.0] * EXPERIENCE_DIM))

        n1 = np.linalg.norm(mood1)
        n2 = np.linalg.norm(mood2)
        if n1 < 1e-8 or n2 < 1e-8:
            return base_score

        similarity = float(np.dot(mood1, mood2) / (n1 * n2))
        bonus = config.inner_life_config.get("mood_attraction_bonus", 0.1)

        return base_score + max(0.0, similarity) * bonus

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def get_metrics(self, population: list[Agent]) -> dict[str, Any]:
        """Return cached experiential metrics."""
        return dict(self._cached_metrics)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_partner_alive(partner_id: str, population: list[Agent]) -> bool:
        """Check if a partner is still alive in the population."""
        for agent in population:
            if agent.id == partner_id:
                return agent.is_alive
        return False
