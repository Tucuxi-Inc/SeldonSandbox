"""
Epistemology Extension — belief formation, propagation, and accuracy dynamics.

Wires the BeliefSystem into engine hooks:
  - on_simulation_start: seed initial beliefs from existing memories
  - on_generation_start: form beliefs from new memories, update ground truths
  - on_generation_end: propagate beliefs, update accuracy, evolve societal beliefs
  - on_agent_created: transmit parent beliefs to child
  - modify_decision: apply belief decision_effects to action utilities
  - modify_mortality: accurate danger beliefs slightly reduce mortality
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from seldon.extensions.base import SimulationExtension
from seldon.social.beliefs import BeliefDomain, BeliefSystem

if TYPE_CHECKING:
    from seldon.core.agent import Agent
    from seldon.core.config import ExperimentConfig


class EpistemologyExtension(SimulationExtension):
    """Belief formation, propagation, accuracy dynamics, and societal beliefs."""

    def __init__(self) -> None:
        self._belief_system: BeliefSystem | None = None
        self._rng: np.random.Generator | None = None
        self._propagation_metrics: dict[str, int] = {}
        self._belief_metrics: dict[str, Any] = {}

    @property
    def name(self) -> str:
        return "epistemology"

    @property
    def description(self) -> str:
        return "Belief formation, propagation, and accuracy dynamics"

    def get_default_config(self) -> dict[str, Any]:
        return {
            "danger_belief_mortality_reduction": 0.05,
        }

    def _get_config(self, config: ExperimentConfig) -> dict[str, Any]:
        defaults = self.get_default_config()
        overrides = config.extensions.get("epistemology", {})
        defaults.update(overrides)
        return defaults

    @property
    def belief_system(self) -> BeliefSystem | None:
        return self._belief_system

    def on_simulation_start(
        self, population: list[Agent], config: ExperimentConfig,
    ) -> None:
        """Initialize BeliefSystem and seed initial beliefs from existing memories."""
        self._belief_system = BeliefSystem(config)
        self._rng = np.random.default_rng(config.random_seed)

        if not config.belief_config.get("enabled", True):
            return

        # Seed initial beliefs from existing agent memories
        for agent in population:
            all_memories = agent.personal_memories + agent.inherited_lore
            for mem_dict in all_memories:
                belief = self._belief_system.form_belief_from_memory(
                    mem_dict, agent, 0, self._rng,
                )
                if belief is not None:
                    BeliefSystem._add_belief_direct(agent, belief)

    def on_generation_start(
        self, generation: int, population: list[Agent],
        config: ExperimentConfig,
    ) -> None:
        """Form beliefs from new memories, update ground truths."""
        if self._belief_system is None:
            self._belief_system = BeliefSystem(config)
            self._rng = np.random.default_rng(config.random_seed)

        if not config.belief_config.get("enabled", True):
            return

        bs = self._belief_system
        rng = self._rng

        # Update ground truths from simulation state
        bs.update_ground_truths_from_simulation(population, config)

        # Form beliefs from recent memories
        for agent in population:
            if not agent.is_alive:
                continue
            all_memories = agent.personal_memories + agent.inherited_lore
            for mem_dict in all_memories:
                # Only consider memories from recent generation
                if mem_dict.get("created_generation", -1) == generation - 1:
                    belief = bs.form_belief_from_memory(
                        mem_dict, agent, generation, rng,
                    )
                    if belief is not None:
                        BeliefSystem._add_belief_direct(agent, belief)

    def on_generation_end(
        self, generation: int, population: list[Agent],
        config: ExperimentConfig,
    ) -> None:
        """Propagate beliefs, update accuracy, evolve societal beliefs."""
        if self._belief_system is None or not config.belief_config.get("enabled", True):
            return

        bs = self._belief_system
        rng = self._rng

        # Propagate through social bonds
        self._propagation_metrics = bs.propagate_beliefs(
            population, generation, rng,
        )

        # Update accuracy dynamics
        bs.update_belief_accuracy(population, generation, rng)

        # Evolve societal beliefs
        bs.evolve_societal_beliefs(population, generation)

        # Cache metrics
        self._belief_metrics = bs.get_metrics(population)

    def on_agent_created(
        self, agent: Agent, parents: tuple[Agent, Agent],
        config: ExperimentConfig,
    ) -> None:
        """Transmit parent beliefs to child."""
        if self._belief_system is None or not config.belief_config.get("enabled", True):
            return

        rng = self._rng
        self._belief_system.transmit_to_child(
            parents[0], parents[1], agent, rng,
        )

    def modify_decision(
        self, agent: Agent, context: str, utilities: dict[str, float],
        config: ExperimentConfig,
    ) -> dict[str, float]:
        """Apply belief decision_effects to action utilities."""
        if self._belief_system is None or not config.belief_config.get("enabled", True):
            return utilities

        beliefs = BeliefSystem._get_beliefs(agent)
        for belief in beliefs:
            for key, modifier in belief.decision_effects.items():
                # key format: "CONTEXT:action"
                parts = key.split(":", 1)
                if len(parts) != 2:
                    continue
                belief_context, action = parts
                if belief_context == context and action in utilities:
                    utilities[action] = utilities[action] + modifier * belief.conviction

        return utilities

    def modify_mortality(
        self, agent: Agent, base_rate: float,
        config: ExperimentConfig,
    ) -> float:
        """Accurate danger beliefs slightly reduce mortality."""
        if self._belief_system is None or not config.belief_config.get("enabled", True):
            return base_rate

        ext_cfg = self._get_config(config)
        beliefs = BeliefSystem._get_beliefs(agent)

        for belief in beliefs:
            if belief.domain == BeliefDomain.DANGER and belief.accuracy > 0.5:
                # Accurate danger knowledge helps survival
                reduction = (
                    ext_cfg["danger_belief_mortality_reduction"]
                    * belief.accuracy
                    * belief.conviction
                )
                base_rate = max(0.0, base_rate - reduction)

        return base_rate

    def get_metrics(self, population: list[Agent]) -> dict[str, Any]:
        """Return cached belief metrics plus propagation stats."""
        metrics = dict(self._belief_metrics)
        metrics.update({
            f"propagation_{k}": v
            for k, v in self._propagation_metrics.items()
        })
        return metrics
