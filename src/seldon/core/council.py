"""
Cognitive Council — 8-voice modulation layer.

Each sub-agent maps to personality traits via configurable weights.
The council is optional (disabled by default) and modulates decision-making
by amplifying or dampening trait influences based on which cognitive voices
are strongest for a given agent.

Sub-agents:
    Cortex      — Analytical reasoning (openness, conscientiousness)
    Seer        — Pattern recognition, intuition (creativity, depth_drive)
    Oracle      — Future orientation, prediction (risk_taking, ambition)
    House       — Safety, stability, home (trust, self_control, agreeableness)
    Prudence    — Caution, risk evaluation (conscientiousness, self_control)
    Hypothalamus — Drives, needs, arousal (extraversion, dominance)
    Amygdala    — Threat detection, fear (neuroticism, resilience inverse)
    Conscience  — Moral reasoning, ethics (agreeableness, empathy)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from seldon.core.agent import Agent
from seldon.core.config import ExperimentConfig


# ---------------------------------------------------------------------------
# Default council weight map
# ---------------------------------------------------------------------------
# Maps sub-agent name -> list of (trait_name, weight) tuples.
# Weights determine how much each trait contributes to a sub-agent's activation.
DEFAULT_COUNCIL_WEIGHTS: dict[str, list[tuple[str, float]]] = {
    "cortex": [
        ("openness", 0.4),
        ("conscientiousness", 0.3),
        ("adaptability", 0.15),
        ("self_control", 0.15),
    ],
    "seer": [
        ("creativity", 0.4),
        ("depth_drive", 0.35),
        ("openness", 0.25),
    ],
    "oracle": [
        ("risk_taking", 0.3),
        ("ambition", 0.35),
        ("adaptability", 0.2),
        ("openness", 0.15),
    ],
    "house": [
        ("trust", 0.3),
        ("self_control", 0.25),
        ("agreeableness", 0.25),
        ("empathy", 0.2),
    ],
    "prudence": [
        ("conscientiousness", 0.35),
        ("self_control", 0.35),
        ("neuroticism", 0.15),
        ("trust", 0.15),
    ],
    "hypothalamus": [
        ("extraversion", 0.35),
        ("dominance", 0.3),
        ("ambition", 0.2),
        ("risk_taking", 0.15),
    ],
    "amygdala": [
        ("neuroticism", 0.45),
        ("resilience", -0.3),
        ("trust", -0.25),
    ],
    "conscience": [
        ("agreeableness", 0.3),
        ("empathy", 0.35),
        ("trust", 0.2),
        ("self_control", 0.15),
    ],
}

SUB_AGENT_NAMES = list(DEFAULT_COUNCIL_WEIGHTS.keys())


# ---------------------------------------------------------------------------
# CognitiveCouncil
# ---------------------------------------------------------------------------
class CognitiveCouncil:
    """
    8-voice cognitive modulation layer.

    Each sub-agent has an activation level based on the agent's traits.
    The dominant voice influences behavior; the full council can produce
    a modulation vector that amplifies/dampens trait contributions.

    Disabled by default (cognitive_council_enabled=False); returns None
    when disabled.
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.ts = config.trait_system
        self.enabled = config.cognitive_council_enabled

        # Use custom weights from config if provided, else defaults
        raw_weights = getattr(config, "cognitive_council_weights", None)
        if raw_weights is None:
            raw_weights = DEFAULT_COUNCIL_WEIGHTS

        # Pre-compute weight vectors for each sub-agent
        # Each sub-agent gets a vector of shape (trait_count,) mapping traits to weights
        self._weight_vectors: dict[str, np.ndarray] = {}
        self._valid_agents: list[str] = []

        for agent_name, trait_weights in raw_weights.items():
            vec = np.zeros(self.ts.count)
            valid = False
            for trait_name, weight in trait_weights:
                try:
                    idx = self.ts.trait_index(trait_name)
                    vec[idx] = weight
                    valid = True
                except KeyError:
                    # Trait not in current preset — skip silently
                    pass
            if valid:
                self._weight_vectors[agent_name] = vec
                self._valid_agents.append(agent_name)

    def get_activations(self, agent: Agent) -> dict[str, float]:
        """
        Compute activation level for each sub-agent based on agent's traits.

        Returns dict mapping sub-agent name -> activation (can be negative
        for agents with negative weights like amygdala).
        """
        activations = {}
        for name in self._valid_agents:
            activations[name] = float(np.dot(self._weight_vectors[name], agent.traits))
        return activations

    def get_dominant_voice(self, agent: Agent) -> str | None:
        """
        Determine which sub-agent is strongest for this agent.

        Returns None if the council is disabled.
        """
        if not self.enabled:
            return None

        activations = self.get_activations(agent)
        if not activations:
            return None

        return max(activations, key=activations.get)

    def compute_council_modulation(
        self, agent: Agent, context: dict[str, Any] | None = None,
    ) -> np.ndarray | None:
        """
        Compute a modulation vector that can be applied to decision weights.

        The modulation amplifies traits associated with the dominant voice
        and slightly dampens traits from opposing voices.

        Returns None if the council is disabled.
        Returns ndarray of shape (trait_count,) with values centered around 1.0.
        Values > 1 amplify that trait's influence; < 1 dampen it.
        """
        if not self.enabled:
            return None

        activations = self.get_activations(agent)
        if not activations:
            return None

        # Normalize activations to [0, 1] range for modulation
        act_values = np.array(list(activations.values()))
        if act_values.max() == act_values.min():
            return np.ones(self.ts.count)

        act_min = act_values.min()
        act_range = act_values.max() - act_min
        normalized = {
            name: (val - act_min) / act_range
            for name, val in activations.items()
        }

        # Build modulation vector: weighted sum of sub-agent weight vectors
        # scaled by their normalized activation
        modulation = np.ones(self.ts.count)
        for name, norm_act in normalized.items():
            weight_vec = self._weight_vectors[name]
            # Modulation: amplify by 0.2 * (norm_act - 0.5) so dominant voices
            # get up to +10% and weak voices get up to -10%
            influence = 0.2 * (norm_act - 0.5)
            # Only modulate traits this sub-agent cares about
            mask = weight_vec != 0
            modulation[mask] += influence * np.abs(weight_vec[mask])

        # Clamp modulation to reasonable range [0.5, 1.5]
        return np.clip(modulation, 0.5, 1.5)
