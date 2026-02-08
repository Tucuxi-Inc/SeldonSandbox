"""
Configurable trait system for the Seldon Sandbox.

Supports compact (15 traits), full (50 traits), or custom trait definitions.
Trait count is NEVER hardcoded — always use trait_system.count.
"""

from __future__ import annotations

from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Compact trait set (15 traits) — default for quick experiments
# ---------------------------------------------------------------------------
COMPACT_TRAITS: list[dict[str, Any]] = [
    {"name": "openness",         "desirability": 1,  "stability": 0.6, "description": "Curiosity, creativity, willingness to explore"},
    {"name": "conscientiousness","desirability": 1,  "stability": 0.7, "description": "Organization, discipline, reliability"},
    {"name": "extraversion",     "desirability": 0,  "stability": 0.5, "description": "Social energy (neutral)"},
    {"name": "agreeableness",    "desirability": 1,  "stability": 0.6, "description": "Cooperation, trust, empathy"},
    {"name": "neuroticism",      "desirability": -1, "stability": 0.4, "description": "Anxiety, emotional instability"},
    {"name": "creativity",       "desirability": 1,  "stability": 0.5, "description": "Novel idea generation"},
    {"name": "resilience",       "desirability": 1,  "stability": 0.6, "description": "Recovery from adversity"},
    {"name": "ambition",         "desirability": 0,  "stability": 0.5, "description": "Drive to achieve (neutral)"},
    {"name": "empathy",          "desirability": 1,  "stability": 0.6, "description": "Understanding others' emotions"},
    {"name": "dominance",        "desirability": 0,  "stability": 0.5, "description": "Leadership assertion (neutral)"},
    {"name": "trust",            "desirability": 1,  "stability": 0.5, "description": "Willingness to rely on others"},
    {"name": "risk_taking",      "desirability": 0,  "stability": 0.4, "description": "Tolerance for uncertainty (neutral)"},
    {"name": "adaptability",     "desirability": 1,  "stability": 0.5, "description": "Flexibility in changing circumstances"},
    {"name": "self_control",     "desirability": 1,  "stability": 0.6, "description": "Impulse regulation"},
    {"name": "depth_drive",      "desirability": 0,  "stability": 0.4, "description": "Tendency toward deep processing (key for RSH)"},
]

# ---------------------------------------------------------------------------
# Full trait set (50 traits) — rich personality modeling from ChatGPT taxonomy
# ---------------------------------------------------------------------------
FULL_TRAITS: list[dict[str, Any]] = [
    # Core Domains (0-4)
    {"name": "openness",              "desirability": 1,  "stability": 0.6, "description": "Curiosity and willingness to explore"},
    {"name": "conscientiousness",     "desirability": 1,  "stability": 0.7, "description": "Organization, discipline, reliability"},
    {"name": "extraversion",          "desirability": 0,  "stability": 0.5, "description": "Social energy"},
    {"name": "agreeableness",         "desirability": 1,  "stability": 0.6, "description": "Cooperation, trust"},
    {"name": "neuroticism",           "desirability": -1, "stability": 0.4, "description": "Anxiety, emotional instability"},
    # Cognitive & Innovation (5-11)
    {"name": "curiosity",             "desirability": 1,  "stability": 0.5, "description": "Drive to learn and discover"},
    {"name": "creativity",            "desirability": 1,  "stability": 0.5, "description": "Novel idea generation"},
    {"name": "innovativeness",        "desirability": 1,  "stability": 0.5, "description": "Applying novel ideas practically"},
    {"name": "intellectual_curiosity", "desirability": 1,  "stability": 0.6, "description": "Deep intellectual exploration"},
    {"name": "systematic_thinking",   "desirability": 1,  "stability": 0.7, "description": "Structured analytical approach"},
    {"name": "open_mindedness",       "desirability": 1,  "stability": 0.5, "description": "Receptivity to new perspectives"},
    {"name": "resourcefulness",       "desirability": 1,  "stability": 0.5, "description": "Creative problem-solving with constraints"},
    # Adaptation & Resilience (12-15)
    {"name": "adaptability",          "desirability": 1,  "stability": 0.5, "description": "Flexibility in changing circumstances"},
    {"name": "resilience",            "desirability": 1,  "stability": 0.6, "description": "Recovery from adversity"},
    {"name": "perseverance",          "desirability": 1,  "stability": 0.7, "description": "Sustained effort despite obstacles"},
    {"name": "focus",                 "desirability": 1,  "stability": 0.6, "description": "Concentration on task"},
    # Social & Interpersonal (16-22)
    {"name": "empathy",               "desirability": 1,  "stability": 0.6, "description": "Understanding others' emotions"},
    {"name": "assertiveness",         "desirability": 0,  "stability": 0.5, "description": "Directness in expressing needs"},
    {"name": "altruism",              "desirability": 1,  "stability": 0.6, "description": "Selfless concern for others"},
    {"name": "tolerance",             "desirability": 1,  "stability": 0.5, "description": "Acceptance of differences"},
    {"name": "trust",                 "desirability": 1,  "stability": 0.5, "description": "Willingness to rely on others"},
    {"name": "sociability",           "desirability": 0,  "stability": 0.5, "description": "Enjoyment of social interaction"},
    {"name": "collaboration",         "desirability": 1,  "stability": 0.5, "description": "Working effectively with others"},
    # Self-Regulation & Drive (23-31)
    {"name": "patience",              "desirability": 1,  "stability": 0.6, "description": "Tolerance for delay"},
    {"name": "self_efficacy",         "desirability": 1,  "stability": 0.5, "description": "Belief in own capabilities"},
    {"name": "optimism",              "desirability": 1,  "stability": 0.5, "description": "Positive expectations"},
    {"name": "ambition",              "desirability": 0,  "stability": 0.5, "description": "Drive to achieve"},
    {"name": "confidence",            "desirability": 0,  "stability": 0.5, "description": "Self-assurance"},
    {"name": "self_control",          "desirability": 1,  "stability": 0.6, "description": "Impulse regulation"},
    {"name": "decisiveness",          "desirability": 1,  "stability": 0.5, "description": "Speed and firmness of decisions"},
    {"name": "integrity",             "desirability": 1,  "stability": 0.8, "description": "Adherence to moral principles"},
    {"name": "humility",              "desirability": 1,  "stability": 0.6, "description": "Accurate self-assessment"},
    # Emotional Processing (32-37)
    {"name": "emotional_stability",   "desirability": 1,  "stability": 0.5, "description": "Evenness of emotional response"},
    {"name": "emotional_expressiveness", "desirability": 0, "stability": 0.4, "description": "Outward display of emotion"},
    {"name": "reflectiveness",        "desirability": 1,  "stability": 0.6, "description": "Tendency for introspection"},
    {"name": "self_awareness",        "desirability": 1,  "stability": 0.6, "description": "Understanding own mental states"},
    {"name": "empathic_accuracy",     "desirability": 1,  "stability": 0.5, "description": "Precision in reading others"},
    {"name": "mindfulness",           "desirability": 1,  "stability": 0.5, "description": "Present-moment awareness"},
    # Pragmatic & Style (38-49)
    {"name": "risk_taking",           "desirability": 0,  "stability": 0.4, "description": "Tolerance for uncertainty"},
    {"name": "pragmatism",            "desirability": 0,  "stability": 0.6, "description": "Practical over theoretical"},
    {"name": "independence",          "desirability": 0,  "stability": 0.5, "description": "Self-reliance"},
    {"name": "competitiveness",       "desirability": 0,  "stability": 0.5, "description": "Drive to outperform others"},
    {"name": "detail_orientation",    "desirability": 0,  "stability": 0.6, "description": "Attention to specifics"},
    {"name": "big_picture_thinking",  "desirability": 0,  "stability": 0.5, "description": "Focus on overarching patterns"},
    {"name": "enthusiasm",            "desirability": 1,  "stability": 0.4, "description": "Energetic engagement"},
    {"name": "humor",                 "desirability": 1,  "stability": 0.5, "description": "Use of levity"},
    {"name": "caution",               "desirability": 0,  "stability": 0.6, "description": "Careful risk evaluation"},
    {"name": "altruistic_leadership", "desirability": 1,  "stability": 0.6, "description": "Leading for collective benefit"},
    {"name": "ethical_reasoning",     "desirability": 1,  "stability": 0.7, "description": "Moral judgment sophistication"},
    {"name": "depth_drive",           "desirability": 0,  "stability": 0.4, "description": "Tendency toward deep processing (key for RSH)"},
]


# ---------------------------------------------------------------------------
# Preset registry
# ---------------------------------------------------------------------------
PRESETS: dict[str, list[dict[str, Any]]] = {
    "compact": COMPACT_TRAITS,
    "full": FULL_TRAITS,
}


class TraitSystem:
    """
    Configurable trait definitions.

    Trait count is determined by the active trait preset, not hardcoded.
    Use trait_system.count for array shapes and trait_system.TRAIT_NAME
    for named indexing.
    """

    def __init__(self, preset: str = "compact",
                 custom_traits: list[dict[str, Any]] | None = None):
        if preset == "custom":
            if not custom_traits:
                raise ValueError("custom_traits required when preset='custom'")
            self.traits = list(custom_traits)
        else:
            if preset not in PRESETS:
                raise ValueError(f"Unknown preset '{preset}'. Choose from: {list(PRESETS)}")
            self.traits = list(PRESETS[preset])

        self.preset = preset
        self.count = len(self.traits)

        # Build name -> index lookup
        self._name_to_index: dict[str, int] = {}
        for i, trait_def in enumerate(self.traits):
            name = trait_def["name"]
            self._name_to_index[name] = i
            # Dynamic attribute: trait_system.OPENNESS, trait_system.DEPTH_DRIVE, etc.
            attr_name = name.upper()
            setattr(self, attr_name, i)

        # Pre-compute vectors
        self._desirability = np.array(
            [t["desirability"] for t in self.traits], dtype=np.float64
        )
        self._stability = np.array(
            [t["stability"] for t in self.traits], dtype=np.float64
        )

    @property
    def desirability(self) -> np.ndarray:
        """Desirability vector for inheritance calculations. Shape (count,)."""
        return self._desirability

    @property
    def stability(self) -> np.ndarray:
        """Stability vector for drift calculations. Shape (count,)."""
        return self._stability

    def trait_name(self, index: int) -> str:
        """Get trait name by index."""
        if 0 <= index < self.count:
            return self.traits[index]["name"]
        raise IndexError(f"Trait index {index} out of range [0, {self.count})")

    def trait_index(self, name: str) -> int:
        """Get trait index by name."""
        idx = self._name_to_index.get(name)
        if idx is None:
            raise KeyError(f"Unknown trait: '{name}'")
        return idx

    def trait_description(self, name_or_index: str | int) -> str:
        """Get trait description by name or index."""
        if isinstance(name_or_index, int):
            return self.traits[name_or_index]["description"]
        return self.traits[self.trait_index(name_or_index)]["description"]

    def random_traits(self, rng: np.random.Generator | None = None) -> np.ndarray:
        """Generate a random trait vector. Values in [0, 1], shape (count,)."""
        rng = rng or np.random.default_rng()
        return rng.uniform(0.0, 1.0, size=self.count)

    def random_population_traits(self, n: int,
                                  rng: np.random.Generator | None = None) -> np.ndarray:
        """Generate trait vectors for n agents. Shape (n, count)."""
        rng = rng or np.random.default_rng()
        return rng.uniform(0.0, 1.0, size=(n, self.count))

    def names(self) -> list[str]:
        """List of all trait names in order."""
        return [t["name"] for t in self.traits]

    def __repr__(self) -> str:
        return f"TraitSystem(preset='{self.preset}', count={self.count})"
