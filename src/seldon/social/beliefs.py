"""
Belief System — epistemological layer for agent knowledge.

Beliefs are formed from memories with strong emotional content and
classified by epistemology (how the agent justifies the knowledge):

    EMPIRICAL    — From direct observation; self-corrects toward truth
    TRADITIONAL  — From authority/ancestors; resistant to change
    SACRED       — Hardened traditional; immune to counter-evidence
    MYTHICAL     — From distorted memory; drifting accuracy

Key insight: ``accuracy`` is objective ground truth the simulation knows
but agents don't.  ``conviction`` drives behavioral impact.  A strongly-
convicted wrong belief (high conviction, low accuracy) is the core of
the sacred/empirical tension.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from uuid import uuid4

import numpy as np

from seldon.core.config import ExperimentConfig


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class EpistemologyType(str, Enum):
    EMPIRICAL = "empirical"
    TRADITIONAL = "traditional"
    SACRED = "sacred"
    MYTHICAL = "mythical"


class BeliefDomain(str, Enum):
    RESOURCE = "resource"
    DANGER = "danger"
    SOCIAL = "social"
    PRODUCTIVITY = "productivity"
    MIGRATION = "migration"
    REPRODUCTION = "reproduction"


# Keyword → domain mapping for inference from memory content
_DOMAIN_KEYWORDS: dict[str, BeliefDomain] = {
    "crop": BeliefDomain.RESOURCE,
    "food": BeliefDomain.RESOURCE,
    "harvest": BeliefDomain.RESOURCE,
    "resource": BeliefDomain.RESOURCE,
    "river": BeliefDomain.RESOURCE,
    "danger": BeliefDomain.DANGER,
    "deadly": BeliefDomain.DANGER,
    "death": BeliefDomain.DANGER,
    "suffering": BeliefDomain.DANGER,
    "winter": BeliefDomain.DANGER,
    "mountain": BeliefDomain.DANGER,
    "leader": BeliefDomain.SOCIAL,
    "first-born": BeliefDomain.SOCIAL,
    "community": BeliefDomain.SOCIAL,
    "status": BeliefDomain.SOCIAL,
    "breakthrough": BeliefDomain.PRODUCTIVITY,
    "contribution": BeliefDomain.PRODUCTIVITY,
    "creativity": BeliefDomain.PRODUCTIVITY,
    "greatness": BeliefDomain.PRODUCTIVITY,
    "migrate": BeliefDomain.MIGRATION,
    "settle": BeliefDomain.MIGRATION,
    "east": BeliefDomain.MIGRATION,
    "land": BeliefDomain.MIGRATION,
    "fertile": BeliefDomain.MIGRATION,
    "children": BeliefDomain.REPRODUCTION,
    "birth": BeliefDomain.REPRODUCTION,
    "family": BeliefDomain.REPRODUCTION,
    "fertility": BeliefDomain.REPRODUCTION,
}


# ---------------------------------------------------------------------------
# Belief dataclass
# ---------------------------------------------------------------------------

@dataclass
class Belief:
    """A single belief with epistemological classification."""

    id: str
    content: str
    domain: BeliefDomain
    epistemology: EpistemologyType
    accuracy: float          # 0-1, ground truth (hidden from agent)
    conviction: float        # 0-1, how strongly held
    decision_effects: dict[str, float]    # "context:action" → utility modifier
    contribution_modifier: float          # Additive modifier on contribution
    source_memory_id: str | None = None
    source_agent_id: str | None = None
    created_generation: int = 0
    transmission_count: int = 0
    challenge_count: int = 0
    reinforcement_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "domain": self.domain.value,
            "epistemology": self.epistemology.value,
            "accuracy": self.accuracy,
            "conviction": self.conviction,
            "decision_effects": dict(self.decision_effects),
            "contribution_modifier": self.contribution_modifier,
            "source_memory_id": self.source_memory_id,
            "source_agent_id": self.source_agent_id,
            "created_generation": self.created_generation,
            "transmission_count": self.transmission_count,
            "challenge_count": self.challenge_count,
            "reinforcement_count": self.reinforcement_count,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Belief:
        return cls(
            id=d["id"],
            content=d["content"],
            domain=BeliefDomain(d["domain"]),
            epistemology=EpistemologyType(d["epistemology"]),
            accuracy=d["accuracy"],
            conviction=d["conviction"],
            decision_effects=dict(d.get("decision_effects", {})),
            contribution_modifier=d.get("contribution_modifier", 0.0),
            source_memory_id=d.get("source_memory_id"),
            source_agent_id=d.get("source_agent_id"),
            created_generation=d.get("created_generation", 0),
            transmission_count=d.get("transmission_count", 0),
            challenge_count=d.get("challenge_count", 0),
            reinforcement_count=d.get("reinforcement_count", 0),
        )


# ---------------------------------------------------------------------------
# BeliefSystem manager
# ---------------------------------------------------------------------------

class BeliefSystem:
    """Manages belief formation, propagation, conflict, and accuracy dynamics."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.ts = config.trait_system
        self._cfg = self._get_belief_config(config)

        # Ground truth registry: domain → truth value
        self.ground_truths: dict[str, float] = {}

        # Beliefs held by ≥ societal threshold of population
        self.societal_beliefs: list[Belief] = []

        # Cached metrics from last update
        self._cached_metrics: dict[str, Any] = {}

    @staticmethod
    def _get_belief_config(config: ExperimentConfig) -> dict[str, Any]:
        defaults = {
            "enabled": True,
            "memory_to_belief_threshold": 0.5,
            "empirical_min_fidelity": 0.7,
            "traditional_min_fidelity": 0.3,
            "sacred_conviction_threshold": 0.8,
            "sacred_min_age_generations": 3,
            "propagation_rate": 0.15,
            "bond_strength_weight": 0.5,
            "conviction_transmission_decay": 0.1,
            "parent_transmission_rate": 0.6,
            "empirical_correction_rate": 0.1,
            "sacred_resistance": 0.95,
            "traditional_resistance": 0.5,
            "mythical_drift_rate": 0.05,
            "conflict_same_domain_threshold": 0.3,
            "evidence_weight": 0.6,
            "social_pressure_weight": 0.4,
            "max_beliefs_per_agent": 12,
            "contribution_modifier_scale": 0.1,
            "decision_effect_scale": 0.15,
            "reinforcement_conviction_boost": 0.05,
            "challenge_conviction_decay": 0.03,
            "base_conviction_decay": 0.005,
        }
        overrides = config.belief_config
        defaults.update(overrides)
        return defaults

    # ------------------------------------------------------------------
    # Formation
    # ------------------------------------------------------------------

    def form_belief_from_memory(
        self,
        memory_dict: dict[str, Any],
        agent: Any,
        generation: int,
        rng: np.random.Generator,
    ) -> Belief | None:
        """
        Attempt to form a belief from a memory.

        Only memories with sufficient emotional valence become beliefs.
        Epistemology depends on memory fidelity.
        """
        valence = memory_dict.get("emotional_valence", 0.0)
        fidelity = memory_dict.get("fidelity", 1.0)
        threshold = self._cfg["memory_to_belief_threshold"]

        if abs(valence) < threshold:
            return None

        # Check max beliefs
        beliefs = agent.extension_data.get("beliefs", [])
        if len(beliefs) >= self._cfg["max_beliefs_per_agent"]:
            return None

        # Classify epistemology from fidelity
        if fidelity >= self._cfg["empirical_min_fidelity"]:
            epistemology = EpistemologyType.EMPIRICAL
        elif fidelity >= self._cfg["traditional_min_fidelity"]:
            epistemology = EpistemologyType.TRADITIONAL
        else:
            epistemology = EpistemologyType.MYTHICAL

        # Infer domain from memory content keywords
        domain = self._infer_domain(memory_dict.get("content", ""))

        # Conviction driven by |valence| × (1 - openness)
        openness = self._get_openness(agent)
        conviction = min(1.0, abs(valence) * (1.0 - openness * 0.5))

        # Accuracy: empirical starts near 0.7, traditional 0.5, mythical random
        if epistemology == EpistemologyType.EMPIRICAL:
            accuracy = 0.5 + rng.uniform(0.1, 0.3)
        elif epistemology == EpistemologyType.TRADITIONAL:
            accuracy = 0.3 + rng.uniform(0.0, 0.3)
        else:
            accuracy = rng.uniform(0.1, 0.6)

        # Decision effects based on domain and valence
        decision_effects = self._generate_decision_effects(domain, valence, rng)

        # Contribution modifier: productivity beliefs affect output
        contrib_mod = 0.0
        if domain == BeliefDomain.PRODUCTIVITY:
            contrib_mod = valence * self._cfg["contribution_modifier_scale"]

        belief = Belief(
            id=str(uuid4())[:8],
            content=memory_dict.get("content", "Unknown belief"),
            domain=domain,
            epistemology=epistemology,
            accuracy=accuracy,
            conviction=conviction,
            decision_effects=decision_effects,
            contribution_modifier=contrib_mod,
            source_memory_id=memory_dict.get("id"),
            source_agent_id=agent.id,
            created_generation=generation,
        )
        return belief

    def _infer_domain(self, content: str) -> BeliefDomain:
        """Infer belief domain from memory content keywords."""
        content_lower = content.lower()
        for keyword, domain in _DOMAIN_KEYWORDS.items():
            if keyword in content_lower:
                return domain
        return BeliefDomain.SOCIAL  # default fallback

    def _get_openness(self, agent: Any) -> float:
        """Get agent's openness trait value, defaulting to 0.5."""
        try:
            idx = self.ts.trait_index("openness")
            return float(agent.traits[idx])
        except (KeyError, ValueError, IndexError):
            return 0.5

    def _generate_decision_effects(
        self,
        domain: BeliefDomain,
        valence: float,
        rng: np.random.Generator,
    ) -> dict[str, float]:
        """Generate decision utility modifiers from belief domain and valence."""
        scale = self._cfg["decision_effect_scale"]
        effects: dict[str, float] = {}

        if domain == BeliefDomain.MIGRATION:
            # Positive valence → encourage migration, negative → discourage
            effects["MIGRATION:migrate"] = valence * scale
            effects["MIGRATION:stay"] = -valence * scale * 0.5
        elif domain == BeliefDomain.DANGER:
            # Danger beliefs make agents cautious about risky actions
            effects["MIGRATION:stay"] = abs(valence) * scale
        elif domain == BeliefDomain.RESOURCE:
            effects["MIGRATION:stay"] = valence * scale * 0.5
        elif domain == BeliefDomain.REPRODUCTION:
            effects["REPRODUCTION:reproduce"] = valence * scale

        return effects

    # ------------------------------------------------------------------
    # Propagation
    # ------------------------------------------------------------------

    def propagate_beliefs(
        self,
        population: list[Any],
        generation: int,
        rng: np.random.Generator,
    ) -> dict[str, int]:
        """Propagate beliefs through social bonds."""
        transmitted = 0
        conflicts = 0
        reinforced = 0

        for agent in population:
            if not agent.is_alive:
                continue
            agent_beliefs = self._get_beliefs(agent)
            if not agent_beliefs:
                continue

            for bonded_id, strength in agent.social_bonds.items():
                # Find bonded agent in population
                receiver = None
                for a in population:
                    if a.id == bonded_id and a.is_alive:
                        receiver = a
                        break
                if receiver is None:
                    continue

                for belief in agent_beliefs:
                    prob = (
                        self._cfg["propagation_rate"]
                        * belief.conviction
                        * (1.0 - self._cfg["bond_strength_weight"]
                           + self._cfg["bond_strength_weight"] * strength)
                    )
                    if rng.random() >= prob:
                        continue

                    receiver_beliefs = self._get_beliefs(receiver)

                    # Check for same-domain conflict
                    existing = self._find_same_domain(receiver_beliefs, belief.domain)
                    if existing is not None:
                        winner = self._resolve_conflict(
                            belief, existing, population, rng,
                        )
                        conflicts += 1
                        if winner.id == existing.id:
                            existing.reinforcement_count += 1
                            existing.conviction = min(
                                1.0,
                                existing.conviction
                                + self._cfg["reinforcement_conviction_boost"],
                            )
                            reinforced += 1
                        else:
                            # Replace existing with transmitted belief
                            self._replace_belief(receiver, existing.id, belief, generation)
                            transmitted += 1
                    else:
                        if len(receiver_beliefs) < self._cfg["max_beliefs_per_agent"]:
                            self._add_belief(receiver, belief, generation)
                            transmitted += 1

        return {
            "transmitted": transmitted,
            "conflicts": conflicts,
            "reinforced": reinforced,
        }

    def transmit_to_child(
        self,
        parent1: Any,
        parent2: Any,
        child: Any,
        rng: np.random.Generator,
    ) -> int:
        """Transmit beliefs from parents to a newborn child."""
        count = 0
        rate = self._cfg["parent_transmission_rate"]

        for parent in (parent1, parent2):
            for belief in self._get_beliefs(parent):
                if rng.random() >= rate * belief.conviction:
                    continue

                child_beliefs = self._get_beliefs(child)
                if len(child_beliefs) >= self._cfg["max_beliefs_per_agent"]:
                    break

                # Check for same-domain conflict with already-transmitted beliefs
                existing = self._find_same_domain(child_beliefs, belief.domain)
                if existing is not None:
                    # Keep the one with higher conviction
                    if belief.conviction > existing.conviction:
                        self._replace_belief(child, existing.id, belief, belief.created_generation)
                        count += 1
                    continue

                # Transmitted belief becomes TRADITIONAL (child didn't observe it)
                new_belief = Belief(
                    id=str(uuid4())[:8],
                    content=belief.content,
                    domain=belief.domain,
                    epistemology=(
                        EpistemologyType.TRADITIONAL
                        if belief.epistemology == EpistemologyType.EMPIRICAL
                        else belief.epistemology
                    ),
                    accuracy=belief.accuracy,
                    conviction=max(
                        0.0,
                        belief.conviction - self._cfg["conviction_transmission_decay"],
                    ),
                    decision_effects=dict(belief.decision_effects),
                    contribution_modifier=belief.contribution_modifier,
                    source_memory_id=belief.source_memory_id,
                    source_agent_id=belief.source_agent_id,
                    created_generation=belief.created_generation,
                    transmission_count=belief.transmission_count + 1,
                )
                self._add_belief_direct(child, new_belief)
                count += 1

        return count

    # ------------------------------------------------------------------
    # Accuracy dynamics
    # ------------------------------------------------------------------

    def update_belief_accuracy(
        self,
        population: list[Any],
        generation: int,
        rng: np.random.Generator,
    ) -> None:
        """Update belief accuracy and conviction based on epistemology type."""
        cfg = self._cfg
        prune_ids: list[tuple[Any, str]] = []

        for agent in population:
            if not agent.is_alive:
                continue
            beliefs = self._get_beliefs(agent)

            for belief in beliefs:
                # EMPIRICAL: drift toward ground truth
                if belief.epistemology == EpistemologyType.EMPIRICAL:
                    truth = self.ground_truths.get(belief.domain.value)
                    if truth is not None:
                        diff = truth - belief.accuracy
                        belief.accuracy += diff * cfg["empirical_correction_rate"]
                        belief.accuracy = float(np.clip(belief.accuracy, 0.0, 1.0))

                # TRADITIONAL: check promotion to SACRED
                elif belief.epistemology == EpistemologyType.TRADITIONAL:
                    age = generation - belief.created_generation
                    if (
                        belief.conviction >= cfg["sacred_conviction_threshold"]
                        and age >= cfg["sacred_min_age_generations"]
                    ):
                        belief.epistemology = EpistemologyType.SACRED

                # SACRED: immune to accuracy change (core property)
                elif belief.epistemology == EpistemologyType.SACRED:
                    pass  # No accuracy change

                # MYTHICAL: random accuracy drift
                elif belief.epistemology == EpistemologyType.MYTHICAL:
                    drift = rng.normal(0, cfg["mythical_drift_rate"])
                    belief.accuracy += drift
                    belief.accuracy = float(np.clip(belief.accuracy, 0.0, 1.0))

                # Passive conviction decay for all beliefs
                belief.conviction -= cfg["base_conviction_decay"]
                belief.conviction = max(0.0, belief.conviction)

                # Prune near-zero conviction
                if belief.conviction < 0.01:
                    prune_ids.append((agent, belief.id))

            # Write back
            self._set_beliefs(agent, beliefs)

        # Prune dead beliefs
        for agent, bid in prune_ids:
            beliefs = self._get_beliefs(agent)
            beliefs = [b for b in beliefs if b.id != bid]
            self._set_beliefs(agent, beliefs)

    # ------------------------------------------------------------------
    # Conflict resolution
    # ------------------------------------------------------------------

    def _resolve_conflict(
        self,
        challenger: Belief,
        defender: Belief,
        population: list[Any],
        rng: np.random.Generator,
    ) -> Belief:
        """
        Resolve conflict between two beliefs in the same domain.

        Sacred beliefs resist challenge with high probability.
        Traditional beliefs resist with moderate probability.
        Otherwise: weighted score of evidence + social pressure.
        """
        cfg = self._cfg

        # Sacred beliefs are nearly immune to challenge
        if defender.epistemology == EpistemologyType.SACRED:
            if rng.random() < cfg["sacred_resistance"]:
                defender.challenge_count += 1
                return defender

        # Traditional beliefs resist with moderate probability
        if defender.epistemology == EpistemologyType.TRADITIONAL:
            if rng.random() < cfg["traditional_resistance"]:
                defender.challenge_count += 1
                return defender

        # Evidence score: accuracy × conviction
        challenger_evidence = challenger.accuracy * challenger.conviction
        defender_evidence = defender.accuracy * defender.conviction

        # Social pressure: count holders in population
        challenger_holders = self._count_belief_holders(
            population, challenger.content,
        )
        defender_holders = self._count_belief_holders(
            population, defender.content,
        )
        total_holders = max(challenger_holders + defender_holders, 1)
        challenger_social = challenger_holders / total_holders
        defender_social = defender_holders / total_holders

        # Weighted score
        ew = cfg["evidence_weight"]
        sw = cfg["social_pressure_weight"]
        challenger_score = ew * challenger_evidence + sw * challenger_social
        defender_score = ew * defender_evidence + sw * defender_social

        total = challenger_score + defender_score
        if total <= 0:
            return defender

        challenger_prob = challenger_score / total
        if rng.random() < challenger_prob:
            challenger.challenge_count += 1
            return challenger
        else:
            defender.challenge_count += 1
            return defender

    def _count_belief_holders(
        self, population: list[Any], content: str,
    ) -> int:
        """Count how many agents hold a belief with matching content."""
        count = 0
        for agent in population:
            if not agent.is_alive:
                continue
            for belief in self._get_beliefs(agent):
                if belief.content == content:
                    count += 1
                    break
        return count

    # ------------------------------------------------------------------
    # Ground truth
    # ------------------------------------------------------------------

    def update_ground_truths_from_simulation(
        self, population: list[Any], config: ExperimentConfig,
    ) -> None:
        """Derive ground truths from current simulation state."""
        if not population:
            return

        from seldon.core.processing import ProcessingRegion

        # Productivity truth: are R4 (sacrificial) agents actually more
        # productive than R2 (optimal)?
        r4_contrib = []
        r2_contrib = []
        for agent in population:
            if not agent.is_alive or not agent.contribution_history:
                continue
            last = agent.contribution_history[-1]
            if agent.processing_region == ProcessingRegion.SACRIFICIAL:
                r4_contrib.append(last)
            elif agent.processing_region == ProcessingRegion.OPTIMAL:
                r2_contrib.append(last)

        if r4_contrib and r2_contrib:
            r4_mean = np.mean(r4_contrib)
            r2_mean = np.mean(r2_contrib)
            # Truth: how much more productive is R4 vs R2 (normalized)
            if r2_mean > 0:
                ratio = min(2.0, r4_mean / r2_mean) / 2.0
            else:
                ratio = 0.5
            self.ground_truths[BeliefDomain.PRODUCTIVITY.value] = float(ratio)

        # Danger truth: what fraction of population died recently
        alive = sum(1 for a in population if a.is_alive)
        total = max(len(population), 1)
        mortality_rate = 1.0 - (alive / total)
        self.ground_truths[BeliefDomain.DANGER.value] = float(
            min(1.0, mortality_rate * 5.0)  # Scale up for visibility
        )

    # ------------------------------------------------------------------
    # Societal beliefs
    # ------------------------------------------------------------------

    def evolve_societal_beliefs(
        self, population: list[Any], generation: int,
    ) -> list[Belief]:
        """Promote widely-held beliefs to societal status."""
        content_counts: dict[str, int] = {}
        content_to_belief: dict[str, Belief] = {}
        alive_count = 0

        for agent in population:
            if not agent.is_alive:
                continue
            alive_count += 1
            for belief in self._get_beliefs(agent):
                content_counts[belief.content] = (
                    content_counts.get(belief.content, 0) + 1
                )
                content_to_belief[belief.content] = belief

        # Threshold: 20% of living population
        threshold = max(2, int(alive_count * 0.2))

        existing_contents = {b.content for b in self.societal_beliefs}
        for content, count in content_counts.items():
            if count >= threshold and content not in existing_contents:
                self.societal_beliefs.append(content_to_belief[content])

        # Prune societal beliefs no longer widely held
        self.societal_beliefs = [
            b for b in self.societal_beliefs
            if content_counts.get(b.content, 0) >= max(1, threshold // 2)
        ]

        return self.societal_beliefs

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def get_metrics(self, population: list[Any]) -> dict[str, Any]:
        """Return belief system metrics for the current generation."""
        total_beliefs = 0
        epistemology_counts: dict[str, int] = {e.value: 0 for e in EpistemologyType}
        domain_counts: dict[str, int] = {d.value: 0 for d in BeliefDomain}
        accuracies: list[float] = []
        convictions: list[float] = []
        agents_with_beliefs = 0

        for agent in population:
            if not agent.is_alive:
                continue
            beliefs = self._get_beliefs(agent)
            if beliefs:
                agents_with_beliefs += 1
            for belief in beliefs:
                total_beliefs += 1
                epistemology_counts[belief.epistemology.value] += 1
                domain_counts[belief.domain.value] += 1
                accuracies.append(belief.accuracy)
                convictions.append(belief.conviction)

        self._cached_metrics = {
            "total_beliefs": total_beliefs,
            "agents_with_beliefs": agents_with_beliefs,
            "beliefs_per_agent": (
                round(total_beliefs / max(agents_with_beliefs, 1), 2)
            ),
            "epistemology_distribution": epistemology_counts,
            "domain_distribution": domain_counts,
            "mean_accuracy": (
                round(float(np.mean(accuracies)), 4) if accuracies else 0.0
            ),
            "mean_conviction": (
                round(float(np.mean(convictions)), 4) if convictions else 0.0
            ),
            "societal_belief_count": len(self.societal_beliefs),
        }
        return self._cached_metrics

    # ------------------------------------------------------------------
    # Agent belief helpers (beliefs stored in extension_data)
    # ------------------------------------------------------------------

    @staticmethod
    def _get_beliefs(agent: Any) -> list[Belief]:
        """Read beliefs from agent.extension_data."""
        raw = agent.extension_data.get("beliefs", [])
        if not raw:
            return []
        if isinstance(raw[0], Belief):
            return raw
        return [Belief.from_dict(d) for d in raw]

    @staticmethod
    def _set_beliefs(agent: Any, beliefs: list[Belief]) -> None:
        """Write beliefs to agent.extension_data as dicts."""
        agent.extension_data["beliefs"] = [b.to_dict() for b in beliefs]

    def _add_belief(
        self, agent: Any, belief: Belief, generation: int,
    ) -> None:
        """Add a transmitted copy of a belief to an agent."""
        new_belief = Belief(
            id=str(uuid4())[:8],
            content=belief.content,
            domain=belief.domain,
            epistemology=(
                EpistemologyType.TRADITIONAL
                if belief.epistemology == EpistemologyType.EMPIRICAL
                else belief.epistemology
            ),
            accuracy=belief.accuracy,
            conviction=max(
                0.0,
                belief.conviction - self._cfg["conviction_transmission_decay"],
            ),
            decision_effects=dict(belief.decision_effects),
            contribution_modifier=belief.contribution_modifier,
            source_memory_id=belief.source_memory_id,
            source_agent_id=belief.source_agent_id,
            created_generation=belief.created_generation,
            transmission_count=belief.transmission_count + 1,
        )
        self._add_belief_direct(agent, new_belief)

    @staticmethod
    def _add_belief_direct(agent: Any, belief: Belief) -> None:
        """Append a belief object directly to agent's extension_data."""
        beliefs = agent.extension_data.get("beliefs", [])
        beliefs.append(belief.to_dict())
        agent.extension_data["beliefs"] = beliefs

    def _replace_belief(
        self, agent: Any, old_id: str, new_belief: Belief, generation: int,
    ) -> None:
        """Replace a belief by ID in agent's extension_data."""
        beliefs = self._get_beliefs(agent)
        beliefs = [b for b in beliefs if b.id != old_id]
        replacement = Belief(
            id=str(uuid4())[:8],
            content=new_belief.content,
            domain=new_belief.domain,
            epistemology=(
                EpistemologyType.TRADITIONAL
                if new_belief.epistemology == EpistemologyType.EMPIRICAL
                else new_belief.epistemology
            ),
            accuracy=new_belief.accuracy,
            conviction=max(
                0.0,
                new_belief.conviction - self._cfg["conviction_transmission_decay"],
            ),
            decision_effects=dict(new_belief.decision_effects),
            contribution_modifier=new_belief.contribution_modifier,
            source_memory_id=new_belief.source_memory_id,
            source_agent_id=new_belief.source_agent_id,
            created_generation=new_belief.created_generation,
            transmission_count=new_belief.transmission_count + 1,
        )
        beliefs.append(replacement)
        self._set_beliefs(agent, beliefs)

    @staticmethod
    def _find_same_domain(
        beliefs: list[Belief], domain: BeliefDomain,
    ) -> Belief | None:
        """Find an existing belief in the same domain."""
        for b in beliefs:
            if b.domain == domain:
                return b
        return None
