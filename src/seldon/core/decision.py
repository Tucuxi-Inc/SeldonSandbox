"""
Utility-based decision model.

All agent decisions flow through this model:
  U(action | personality, context) = personality^T · W_action · context + b_action

Action selection via softmax with configurable temperature.
Every decision produces a DecisionResult with per-trait explainability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from seldon.core.agent import Agent
    from seldon.core.config import ExperimentConfig


class DecisionContext(Enum):
    """Types of decisions agents make."""
    PAIRING = "pairing"
    REPRODUCTION = "reproduction"
    MIGRATION = "migration"
    CONFLICT = "conflict"
    CONTRIBUTION = "contribution"
    SOCIAL = "social"


@dataclass
class DecisionResult:
    """Result of a utility-based decision with explainability data."""
    chosen_action: str
    probabilities: dict[str, float]
    trait_contributions: np.ndarray  # Per-trait influence on chosen action
    utilities: dict[str, float]     # Raw utility per action
    context_type: DecisionContext

    def explain(self, trait_names: list[str]) -> dict[str, float]:
        """Return a readable mapping of trait name -> influence score."""
        return {
            name: float(self.trait_contributions[i])
            for i, name in enumerate(trait_names)
            if abs(self.trait_contributions[i]) > 0.01
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "chosen_action": self.chosen_action,
            "probabilities": self.probabilities,
            "utilities": self.utilities,
            "context": self.context_type.value,
        }


class DecisionModel:
    """
    Unified utility-based decision engine.

    For Phase 1, uses a simplified model where trait-context interaction
    is computed via configurable weight vectors per action. The full
    matrix form (P^T · W · x) is available for Phase 2 when richer
    context vectors are needed.
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.ts = config.trait_system
        self.temperature = config.decision_temperature
        self._extension_modifiers: list = []

    def set_extension_modifiers(self, modifiers: list) -> None:
        """Set extension modifier callbacks for decision utilities."""
        self._extension_modifiers = modifiers

    def decide(
        self,
        agent: Agent,
        context: DecisionContext,
        situation: dict[str, float],
        actions: list[str],
        action_weights: dict[str, np.ndarray] | None = None,
        action_biases: dict[str, float] | None = None,
        rng: np.random.Generator | None = None,
    ) -> DecisionResult:
        """
        Compute utility for each action and select one via softmax.

        Parameters
        ----------
        agent : Agent
            The deciding agent (personality = traits vector).
        context : DecisionContext
            Type of decision being made.
        situation : dict[str, float]
            Context variables (e.g., resource_level, overcrowding).
        actions : list[str]
            Available actions to choose from.
        action_weights : dict[str, np.ndarray], optional
            Per-action trait weight vectors, shape (trait_count,).
            If None, uses uniform weights (pure trait-magnitude based).
        action_biases : dict[str, float], optional
            Per-action bias terms. Default 0.
        rng : Generator, optional

        Returns
        -------
        DecisionResult with chosen action, probabilities, and explainability.
        """
        rng = rng or np.random.default_rng()

        if not actions:
            raise ValueError("At least one action required")

        if len(actions) == 1:
            return DecisionResult(
                chosen_action=actions[0],
                probabilities={actions[0]: 1.0},
                trait_contributions=agent.traits.copy(),
                utilities={actions[0]: 1.0},
                context_type=context,
            )

        # Compute situation magnitude for scaling
        sit_values = np.array(list(situation.values())) if situation else np.array([1.0])
        sit_magnitude = float(np.mean(np.abs(sit_values))) if len(sit_values) > 0 else 1.0

        # Compute utility for each action
        utilities: dict[str, float] = {}
        contributions_by_action: dict[str, np.ndarray] = {}

        for action in actions:
            if action_weights and action in action_weights:
                w = action_weights[action]
            else:
                # Default: uniform weight vector
                w = np.ones(self.ts.count) / self.ts.count

            bias = (action_biases or {}).get(action, 0.0)

            # U(a|P,x) = P · w * sit_magnitude + b
            contributions = agent.traits * w * sit_magnitude
            utility = float(contributions.sum()) + bias

            utilities[action] = utility
            contributions_by_action[action] = contributions

        # Extension modifier hooks (before softmax)
        for modifier in self._extension_modifiers:
            utilities = modifier(agent, context.value, utilities, self.config)

        # Softmax with temperature
        values = np.array([utilities[a] for a in actions])
        probabilities = self._softmax(values, self.temperature)
        prob_dict = dict(zip(actions, probabilities.tolist()))

        # Select action
        chosen_idx = rng.choice(len(actions), p=probabilities)
        chosen = actions[chosen_idx]

        return DecisionResult(
            chosen_action=chosen,
            probabilities=prob_dict,
            trait_contributions=contributions_by_action[chosen],
            utilities=utilities,
            context_type=context,
        )

    @staticmethod
    def _softmax(values: np.ndarray, temperature: float) -> np.ndarray:
        """Numerically stable softmax with temperature scaling."""
        if temperature <= 0:
            # Deterministic: pick the max
            result = np.zeros_like(values)
            result[np.argmax(values)] = 1.0
            return result

        scaled = (values - values.max()) / temperature
        exp_vals = np.exp(scaled)
        return exp_vals / exp_vals.sum()
