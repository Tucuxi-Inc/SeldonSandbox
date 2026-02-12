"""
Experiential Engine — phenomenological layer for agent inner life.

Events create 6-dimensional "felt quality" vectors colored by agent
personality.  These experiences accumulate, influence decisions through
similarity-based recall, drive directional trait drift, and produce a
"phenomenal quality of life" score for subjective well-being.

The 6 experience dimensions:
    valence        [-1, 1]  Positive/negative emotional tone
    arousal        [0,  1]  Intensity of the experience
    social_quality [-1, 1]  Isolation vs connection
    agency         [0,  1]  Control vs helplessness
    novelty        [0,  1]  Familiar vs new
    meaning        [0,  1]  Sense of purpose
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import numpy as np

from seldon.core.config import ExperimentConfig


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXPERIENCE_DIM = 6
EXPERIENCE_LABELS = ["valence", "arousal", "social_quality", "agency", "novelty", "meaning"]

# Trait → experience dimension modulation weights.
# Missing traits in compact mode are silently skipped.
DEFAULT_TRAIT_EXPERIENCE_WEIGHTS: dict[str, dict[str, float]] = {
    "valence":        {"resilience": 0.3, "neuroticism": -0.3},
    "arousal":        {"extraversion": 0.3, "self_control": -0.2},
    "social_quality": {"empathy": 0.3, "agreeableness": 0.3},
    "agency":         {"self_control": 0.3, "dominance": 0.2, "ambition": 0.2},
    "novelty":        {"openness": 0.3, "adaptability": 0.3},
    "meaning":        {"depth_drive": 0.3, "conscientiousness": 0.2, "creativity": 0.2},
}

#                          valence  arousal  social  agency  novelty  meaning
BASE_EVENT_VECTORS: dict[str, list[float]] = {
    "breakthrough":    [ 0.8,  0.7,  0.1,  0.8,  0.6,  0.9],
    "deep_suffering":  [-0.7,  0.8, -0.2, -0.3,  0.3,  0.4],
    "pair_formed":     [ 0.7,  0.5,  0.8,  0.5,  0.4,  0.6],
    "pair_dissolved":  [-0.6,  0.5, -0.6, -0.2,  0.3,  0.2],
    "child_born":      [ 0.8,  0.6,  0.7,  0.4,  0.5,  0.8],
    "bereavement":     [-0.9,  0.8, -0.5, -0.4,  0.2,  0.3],
    "status_change":   [ 0.3,  0.4,  0.6,  0.5,  0.4,  0.5],
    "migration":       [ 0.1,  0.5, -0.3,  0.6,  0.8,  0.4],
    "routine":         [ 0.1,  0.2,  0.3,  0.4,  0.1,  0.2],
}

# Experience dimension → trait drift mapping.
# Positive dimension value pushes listed traits; negative pushes opposite.
EXPERIENCE_DRIFT_MAP: dict[str, dict[str, float]] = {
    "valence":        {"resilience": 0.3, "neuroticism": -0.3},
    "arousal":        {},  # Arousal doesn't directly drift traits
    "social_quality": {"agreeableness": 0.2, "empathy": 0.2, "trust": 0.2},
    "agency":         {"resilience": 0.3, "self_control": 0.2},
    "novelty":        {"openness": 0.3, "adaptability": 0.2},
    "meaning":        {"depth_drive": 0.3, "conscientiousness": 0.2},
}


# ---------------------------------------------------------------------------
# ExperientialEngine
# ---------------------------------------------------------------------------

class ExperientialEngine:
    """Manages experiential encoding, recall, phenomenal quality, and drift."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.ts = config.trait_system
        self._cfg = self._get_config(config)

        # Pre-compute trait index vectors for experience modulation
        self._trait_mod_indices: dict[str, list[tuple[int, float]]] = {}
        tw = self._cfg.get("trait_experience_weights") or DEFAULT_TRAIT_EXPERIENCE_WEIGHTS
        for dim_name, trait_weights in tw.items():
            indices = []
            for trait_name, weight in trait_weights.items():
                try:
                    idx = self.ts.trait_index(trait_name)
                    indices.append((idx, weight))
                except (KeyError, ValueError):
                    pass
            self._trait_mod_indices[dim_name] = indices

        # Pre-compute trait indices for drift
        self._drift_indices: dict[str, list[tuple[int, float]]] = {}
        for dim_name, trait_weights in EXPERIENCE_DRIFT_MAP.items():
            indices = []
            for trait_name, weight in trait_weights.items():
                try:
                    idx = self.ts.trait_index(trait_name)
                    indices.append((idx, weight))
                except (KeyError, ValueError):
                    pass
            self._drift_indices[dim_name] = indices

    @staticmethod
    def _get_config(config: ExperimentConfig) -> dict[str, Any]:
        defaults = {
            "enabled": True,
            "modulation_strength": 0.15,
            "max_experiences_per_agent": 50,
            "recall_similarity_threshold": 0.6,
            "recall_top_k": 5,
            "experiential_weight": 0.4,
            "positive_recall_boost": 0.15,
            "negative_recall_penalty": 0.15,
            "initial_assertoric_strength": 0.9,
            "assertoric_decay_rate": 0.05,
            "assertoric_min_floor": 0.05,
            "arousal_decay_protection": 0.3,
            "pq_valence_weight": 0.3,
            "pq_social_weight": 0.2,
            "pq_agency_weight": 0.2,
            "pq_meaning_weight": 0.2,
            "pq_needs_weight": 0.1,
            "pq_lookback_generations": 5,
            "experiential_drift_rate": 0.005,
            "drift_lookback_generations": 3,
            "mood_decay": 0.7,
            "pq_mortality_scale": 0.03,
            "mood_attraction_bonus": 0.1,
            "inheritance_top_k": 3,
            "inheritance_strength_decay": 0.5,
            "trait_experience_weights": None,
        }
        overrides = config.inner_life_config
        defaults.update(overrides)
        return defaults

    # ------------------------------------------------------------------
    # State initialization
    # ------------------------------------------------------------------

    def init_state(self, agent: Any) -> None:
        """Initialize inner life state on an agent."""
        agent.extension_data["inner_life"] = {
            "experiences": [],
            "phenomenal_quality": 0.5,
            "pq_history": [],
            "mood": [0.0] * EXPERIENCE_DIM,
            "experiential_drift_applied": {},
            "_prev_state": {
                "partner_id": agent.partner_id,
                "children_count": len(agent.children_ids),
                "suffering": agent.suffering,
            },
        }

    def seed_mood_from_traits(self, agent: Any) -> None:
        """Set initial mood based on personality traits."""
        mood = [0.0] * EXPERIENCE_DIM
        for i, dim_name in enumerate(EXPERIENCE_LABELS):
            for idx, weight in self._trait_mod_indices.get(dim_name, []):
                val = float(agent.traits[idx])
                mood[i] += (val - 0.5) * weight * 0.5
        # Clip mood dimensions
        mood[0] = float(np.clip(mood[0], -1.0, 1.0))  # valence
        mood[2] = float(np.clip(mood[2], -1.0, 1.0))  # social
        for i in [1, 3, 4, 5]:
            mood[i] = float(np.clip(mood[i], 0.0, 1.0))
        state = agent.extension_data.get("inner_life")
        if state is not None:
            state["mood"] = mood

    # ------------------------------------------------------------------
    # Experience encoding
    # ------------------------------------------------------------------

    def encode_experience(
        self,
        agent: Any,
        event_type: str,
        generation: int,
        rng: np.random.Generator,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Encode an event into a trait-colored felt quality vector."""
        base = list(BASE_EVENT_VECTORS.get(event_type, BASE_EVENT_VECTORS["routine"]))

        # Special case: R4 suffering has high meaning, R5 has low
        if event_type == "deep_suffering" and context:
            from seldon.core.processing import ProcessingRegion
            region = context.get("processing_region")
            if region == ProcessingRegion.SACRIFICIAL:
                base[5] = 0.7  # meaning is high (suffering has purpose)
            elif region == ProcessingRegion.PATHOLOGICAL:
                base[5] = 0.1  # meaning is low (suffering is empty)

        # Routine modulation by needs satisfaction and social bonds
        if event_type == "routine":
            needs = getattr(agent, "needs", {})
            if needs:
                avg_needs = sum(needs.values()) / max(len(needs), 1)
                base[0] += (avg_needs - 0.5) * 0.3  # needs satisfaction → valence
            bonds = getattr(agent, "social_bonds", {})
            if bonds:
                avg_bond = sum(bonds.values()) / max(len(bonds), 1)
                base[2] += avg_bond * 0.3  # bonds → social quality

        # Apply trait modulation
        mod_strength = self._cfg["modulation_strength"]
        felt = list(base)
        for i, dim_name in enumerate(EXPERIENCE_LABELS):
            for idx, weight in self._trait_mod_indices.get(dim_name, []):
                trait_val = float(agent.traits[idx])
                felt[i] += (trait_val - 0.5) * weight * mod_strength

        # Clip to valid ranges
        felt[0] = float(np.clip(felt[0], -1.0, 1.0))  # valence
        felt[2] = float(np.clip(felt[2], -1.0, 1.0))  # social_quality
        for i in [1, 3, 4, 5]:
            felt[i] = float(np.clip(felt[i], 0.0, 1.0))

        experience = {
            "id": str(uuid4())[:8],
            "generation": generation,
            "event_type": event_type,
            "felt_vector": felt,
            "assertoric_strength": self._cfg["initial_assertoric_strength"],
            "recall_count": 0,
        }

        # Store
        state = agent.extension_data.get("inner_life")
        if state is not None:
            state["experiences"].append(experience)

        return experience

    # ------------------------------------------------------------------
    # Similarity-based recall
    # ------------------------------------------------------------------

    def recall_similar(
        self,
        agent: Any,
        query_vector: list[float],
        top_k: int | None = None,
    ) -> list[tuple[dict[str, Any], float]]:
        """Find past experiences similar to a query vector."""
        top_k = top_k or self._cfg["recall_top_k"]
        threshold = self._cfg["recall_similarity_threshold"]

        state = agent.extension_data.get("inner_life")
        if state is None:
            return []

        experiences = state.get("experiences", [])
        if not experiences:
            return []

        query = np.array(query_vector, dtype=float)
        query_norm = np.linalg.norm(query)
        if query_norm < 1e-8:
            return []

        scored: list[tuple[dict[str, Any], float]] = []
        for exp in experiences:
            felt = np.array(exp["felt_vector"], dtype=float)
            felt_norm = np.linalg.norm(felt)
            if felt_norm < 1e-8:
                continue
            cosine_sim = float(np.dot(query, felt) / (query_norm * felt_norm))
            # Weight by assertoric strength
            weighted_sim = cosine_sim * exp.get("assertoric_strength", 0.5)
            if weighted_sim >= threshold:
                scored.append((exp, weighted_sim))

        # Sort descending by similarity
        scored.sort(key=lambda x: x[1], reverse=True)

        # Increment recall count on returned experiences
        result = scored[:top_k]
        for exp, _ in result:
            exp["recall_count"] = exp.get("recall_count", 0) + 1

        return result

    # ------------------------------------------------------------------
    # Decision modulation (60/40 split)
    # ------------------------------------------------------------------

    def compute_experiential_modulation(
        self,
        agent: Any,
        context: str,
        utilities: dict[str, float],
    ) -> dict[str, float]:
        """Apply experiential recall to modify decision utilities."""
        state = agent.extension_data.get("inner_life")
        if state is None:
            return utilities

        mood = state.get("mood", [0.0] * EXPERIENCE_DIM)
        recalled = self.recall_similar(agent, mood)
        if not recalled:
            return utilities

        ew = self._cfg["experiential_weight"]
        boost = self._cfg["positive_recall_boost"]
        penalty = self._cfg["negative_recall_penalty"]

        # Compute experiential adjustment per action
        exp_adjustments: dict[str, float] = {a: 0.0 for a in utilities}
        for exp, sim in recalled:
            valence = exp["felt_vector"][0]  # valence dimension
            for action in utilities:
                if valence > 0:
                    exp_adjustments[action] += sim * boost * valence
                else:
                    exp_adjustments[action] += sim * penalty * valence

        # Apply 60/40 blend
        modified = {}
        for action, base_util in utilities.items():
            adj = exp_adjustments.get(action, 0.0)
            modified[action] = base_util + ew * adj

        return modified

    # ------------------------------------------------------------------
    # Phenomenal quality
    # ------------------------------------------------------------------

    def compute_phenomenal_quality(self, agent: Any) -> float:
        """Compute subjective well-being from recent experiences."""
        state = agent.extension_data.get("inner_life")
        if state is None:
            return 0.5

        experiences = state.get("experiences", [])
        lookback = self._cfg["pq_lookback_generations"]

        # Get recent experiences
        if experiences:
            max_gen = max(e["generation"] for e in experiences)
            recent = [
                e for e in experiences
                if e["generation"] > max_gen - lookback
            ]
        else:
            recent = []

        if not recent:
            pq = 0.5
        else:
            # Weighted average of experience dimensions
            valences = [e["felt_vector"][0] for e in recent]
            socials = [e["felt_vector"][2] for e in recent]
            agencies = [e["felt_vector"][3] for e in recent]
            meanings = [e["felt_vector"][5] for e in recent]

            w = self._cfg
            # Normalize valence and social from [-1,1] to [0,1]
            pq = (
                w["pq_valence_weight"] * (np.mean(valences) + 1.0) / 2.0
                + w["pq_social_weight"] * (np.mean(socials) + 1.0) / 2.0
                + w["pq_agency_weight"] * np.mean(agencies)
                + w["pq_meaning_weight"] * np.mean(meanings)
            )

            # Needs contribution
            needs = getattr(agent, "needs", {})
            if needs:
                avg_needs = sum(needs.values()) / max(len(needs), 1)
                pq += w["pq_needs_weight"] * avg_needs

        pq = float(np.clip(pq, 0.0, 1.0))
        state["phenomenal_quality"] = pq
        state["pq_history"].append(pq)
        return pq

    # ------------------------------------------------------------------
    # Experience-driven trait drift
    # ------------------------------------------------------------------

    def compute_experiential_drift(
        self, agent: Any,
    ) -> dict[str, float]:
        """Compute trait drift from recent experiences (additive to random drift)."""
        state = agent.extension_data.get("inner_life")
        if state is None:
            return {}

        experiences = state.get("experiences", [])
        lookback = self._cfg["drift_lookback_generations"]
        rate = self._cfg["experiential_drift_rate"]

        if not experiences:
            return {}

        max_gen = max(e["generation"] for e in experiences)
        recent = [
            e for e in experiences
            if e["generation"] > max_gen - lookback
        ]
        if not recent:
            return {}

        # Aggregate felt quality across recent experiences,
        # weighted by assertoric strength
        weighted_dims = np.zeros(EXPERIENCE_DIM)
        total_weight = 0.0
        for exp in recent:
            strength = exp.get("assertoric_strength", 0.5)
            weighted_dims += np.array(exp["felt_vector"]) * strength
            total_weight += strength

        if total_weight < 1e-8:
            return {}

        avg_dims = weighted_dims / total_weight

        # Map experience dimensions → trait drift
        drift: dict[str, float] = {}
        for i, dim_name in enumerate(EXPERIENCE_LABELS):
            dim_val = avg_dims[i]
            for idx, weight in self._drift_indices.get(dim_name, []):
                trait_name = self.ts.trait_name(idx)
                delta = dim_val * weight * rate
                drift[trait_name] = drift.get(trait_name, 0.0) + delta

        state["experiential_drift_applied"] = drift
        return drift

    def apply_drift(self, agent: Any, drift: dict[str, float]) -> None:
        """Apply experiential drift to agent traits."""
        for trait_name, delta in drift.items():
            try:
                idx = self.ts.trait_index(trait_name)
                agent.traits[idx] = float(np.clip(
                    agent.traits[idx] + delta, 0.0, 1.0,
                ))
            except (KeyError, ValueError):
                pass

    # ------------------------------------------------------------------
    # Mood
    # ------------------------------------------------------------------

    def update_mood(self, agent: Any) -> None:
        """Update mood as exponential moving average of recent experience vectors."""
        state = agent.extension_data.get("inner_life")
        if state is None:
            return

        experiences = state.get("experiences", [])
        if not experiences:
            return

        latest = experiences[-1]["felt_vector"]
        decay = self._cfg["mood_decay"]
        old_mood = state.get("mood", [0.0] * EXPERIENCE_DIM)

        new_mood = []
        for i in range(EXPERIENCE_DIM):
            new_mood.append(decay * old_mood[i] + (1.0 - decay) * latest[i])

        # Clip
        new_mood[0] = float(np.clip(new_mood[0], -1.0, 1.0))
        new_mood[2] = float(np.clip(new_mood[2], -1.0, 1.0))
        for i in [1, 3, 4, 5]:
            new_mood[i] = float(np.clip(new_mood[i], 0.0, 1.0))

        state["mood"] = new_mood

    # ------------------------------------------------------------------
    # Assertoric decay
    # ------------------------------------------------------------------

    def decay_assertoric_strength(self, agent: Any) -> None:
        """Decay assertoric strength; high-arousal experiences decay slower."""
        state = agent.extension_data.get("inner_life")
        if state is None:
            return

        decay_rate = self._cfg["assertoric_decay_rate"]
        min_floor = self._cfg["assertoric_min_floor"]
        arousal_protection = self._cfg["arousal_decay_protection"]

        for exp in state.get("experiences", []):
            arousal = exp["felt_vector"][1]  # arousal dimension
            effective_decay = decay_rate * (1.0 - arousal * arousal_protection)
            effective_decay = max(0.0, effective_decay)
            exp["assertoric_strength"] = max(
                min_floor,
                exp["assertoric_strength"] * (1.0 - effective_decay),
            )

    # ------------------------------------------------------------------
    # Pruning
    # ------------------------------------------------------------------

    def prune_experiences(self, agent: Any) -> None:
        """Remove oldest experiences beyond max capacity."""
        state = agent.extension_data.get("inner_life")
        if state is None:
            return

        max_exp = self._cfg["max_experiences_per_agent"]
        experiences = state.get("experiences", [])
        if len(experiences) > max_exp:
            # Keep most recent
            state["experiences"] = experiences[-max_exp:]

    # ------------------------------------------------------------------
    # Experiential inheritance
    # ------------------------------------------------------------------

    def inherit_experiences(
        self,
        parent1: Any,
        parent2: Any,
        child: Any,
    ) -> int:
        """Inherit strongest experiences from parents at reduced strength."""
        top_k = self._cfg["inheritance_top_k"]
        strength_decay = self._cfg["inheritance_strength_decay"]
        count = 0

        child_state = child.extension_data.get("inner_life")
        if child_state is None:
            return 0

        for parent in (parent1, parent2):
            p_state = parent.extension_data.get("inner_life")
            if p_state is None:
                continue

            experiences = list(p_state.get("experiences", []))
            # Sort by assertoric strength descending
            experiences.sort(
                key=lambda e: e.get("assertoric_strength", 0.0),
                reverse=True,
            )

            for exp in experiences[:top_k]:
                inherited = {
                    "id": str(uuid4())[:8],
                    "generation": exp["generation"],
                    "event_type": exp["event_type"],
                    "felt_vector": list(exp["felt_vector"]),
                    "assertoric_strength": exp["assertoric_strength"] * strength_decay,
                    "recall_count": 0,
                }
                child_state["experiences"].append(inherited)
                count += 1

        return count

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def get_metrics(self, population: list[Any]) -> dict[str, Any]:
        """Compute population-level experiential metrics."""
        pqs: list[float] = []
        exp_counts: list[int] = []
        event_type_counts: dict[str, int] = {}
        mood_sum = np.zeros(EXPERIENCE_DIM)
        drift_magnitudes: list[float] = []
        alive_count = 0

        for agent in population:
            if not agent.is_alive:
                continue
            alive_count += 1
            state = agent.extension_data.get("inner_life")
            if state is None:
                continue

            pqs.append(state.get("phenomenal_quality", 0.5))
            exps = state.get("experiences", [])
            exp_counts.append(len(exps))

            for exp in exps:
                et = exp.get("event_type", "unknown")
                event_type_counts[et] = event_type_counts.get(et, 0) + 1

            mood = state.get("mood", [0.0] * EXPERIENCE_DIM)
            mood_sum += np.array(mood)

            drift = state.get("experiential_drift_applied", {})
            if drift:
                drift_magnitudes.append(
                    sum(abs(v) for v in drift.values())
                )

        # PQ distribution buckets
        pq_dist = {"0.0-0.2": 0, "0.2-0.4": 0, "0.4-0.6": 0, "0.6-0.8": 0, "0.8-1.0": 0}
        for pq in pqs:
            if pq < 0.2:
                pq_dist["0.0-0.2"] += 1
            elif pq < 0.4:
                pq_dist["0.2-0.4"] += 1
            elif pq < 0.6:
                pq_dist["0.4-0.6"] += 1
            elif pq < 0.8:
                pq_dist["0.6-0.8"] += 1
            else:
                pq_dist["0.8-1.0"] += 1

        mean_mood = (mood_sum / max(alive_count, 1)).tolist()

        return {
            "mean_phenomenal_quality": round(float(np.mean(pqs)), 4) if pqs else 0.5,
            "pq_distribution": pq_dist,
            "mean_experience_count": round(float(np.mean(exp_counts)), 2) if exp_counts else 0.0,
            "event_type_counts": event_type_counts,
            "population_mood": [round(m, 4) for m in mean_mood],
            "mean_drift_magnitude": (
                round(float(np.mean(drift_magnitudes)), 6)
                if drift_magnitudes else 0.0
            ),
        }
