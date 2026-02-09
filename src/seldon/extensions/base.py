"""
Base class for simulation extensions.

All extensions implement this ABC. Default hook implementations are
no-ops/pass-throughs so extensions only override what they need.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from seldon.core.agent import Agent
    from seldon.core.config import ExperimentConfig


class SimulationExtension(ABC):
    """
    Abstract base for optional simulation extensions.

    Extensions hook into the engine lifecycle to add spatial, economic,
    cultural, or conflict mechanics without modifying the core loop.

    Lifecycle hooks fire in this order each generation:
        on_generation_start → [core phases] → on_generation_end

    Modifier hooks are called inline during core phases:
        modify_attraction  — during pairing
        modify_mortality   — during death checks
        modify_decision    — before softmax action selection
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this extension."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description."""

    @abstractmethod
    def get_default_config(self) -> dict[str, Any]:
        """
        Return default configuration for this extension.

        May include a ``"requires"`` key listing extension names that
        must be enabled first (dependency resolution).
        """

    # --- Lifecycle hooks (no-op by default) ---

    def on_simulation_start(
        self, population: list[Agent], config: ExperimentConfig,
    ) -> None:
        """Called once before the generation loop begins."""

    def on_generation_start(
        self, generation: int, population: list[Agent],
        config: ExperimentConfig,
    ) -> None:
        """Called at the start of each generation, before any phases."""

    def on_agent_created(
        self, agent: Agent, parents: tuple[Agent, Agent],
        config: ExperimentConfig,
    ) -> None:
        """Called when a child agent is created during reproduction."""

    def on_generation_end(
        self, generation: int, population: list[Agent],
        config: ExperimentConfig,
    ) -> None:
        """Called at the end of each generation, before snapshot."""

    # --- Modifier hooks (pass-through by default) ---

    def modify_attraction(
        self, agent1: Agent, agent2: Agent, base_score: float,
        config: ExperimentConfig,
    ) -> float:
        """Modify attraction score between two agents."""
        return base_score

    def modify_mortality(
        self, agent: Agent, base_rate: float,
        config: ExperimentConfig,
    ) -> float:
        """Modify mortality rate for an agent."""
        return base_rate

    def modify_decision(
        self, agent: Agent, context: str, utilities: dict[str, float],
        config: ExperimentConfig,
    ) -> dict[str, float]:
        """Modify utility scores before softmax action selection."""
        return utilities

    # --- Metrics ---

    def get_metrics(self, population: list[Agent]) -> dict[str, Any]:
        """Return extension-specific metrics for the current generation."""
        return {}
