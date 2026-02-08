"""
Main simulation engine.

Runs the multi-generational simulation with 7 phases per generation,
extension hooks, outsider injection support, and full metrics collection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

import numpy as np

from seldon.core.agent import Agent
from seldon.core.attraction import AttractionModel
from seldon.core.config import ExperimentConfig
from seldon.core.decision import DecisionContext, DecisionModel
from seldon.core.drift import TraitDriftEngine
from seldon.core.inheritance import InheritanceEngine
from seldon.core.processing import ProcessingClassifier, ProcessingRegion


# ---------------------------------------------------------------------------
# Name generation (simple deterministic names for agents)
# ---------------------------------------------------------------------------
_FIRST_NAMES = [
    "Ada", "Alan", "Alice", "Anya", "Atlas", "Bea", "Blake", "Cleo",
    "Cole", "Dana", "Eli", "Elara", "Eve", "Finn", "Grace", "Hugo",
    "Iris", "Jade", "Kael", "Kai", "Leo", "Luna", "Mae", "Max",
    "Nia", "Noah", "Ora", "Owen", "Pia", "Quinn", "Reed", "Rena",
    "Sage", "Sol", "Tara", "Troy", "Uma", "Vale", "Wren", "Zara",
]


def _generate_name(generation: int, index: int) -> str:
    name = _FIRST_NAMES[index % len(_FIRST_NAMES)]
    return f"{name}-G{generation}"


# ---------------------------------------------------------------------------
# Generation metrics (lightweight snapshot)
# ---------------------------------------------------------------------------
@dataclass
class GenerationSnapshot:
    """Per-generation metrics snapshot."""
    generation: int
    population_size: int
    births: int
    deaths: int
    breakthroughs: int
    pairs_formed: int

    # Trait stats
    trait_means: np.ndarray
    trait_stds: np.ndarray

    # Region distribution
    region_counts: dict[str, int]

    # Contribution
    total_contribution: float
    mean_contribution: float

    # Suffering
    mean_suffering: float

    # Demographics
    mean_age: float

    # Birth order analysis
    birth_order_counts: dict[int, int]

    # Raw events dict for extension data
    events: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Simulation Engine
# ---------------------------------------------------------------------------
class SimulationEngine:
    """
    Main simulation loop.

    Phases per generation:
    1. Age & trait drift
    2. Processing region updates
    3. Contribution & breakthroughs
    4. Relationship dynamics (pairing)
    5. Reproduction
    6. Mortality
    7. Record metrics
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.ts = config.trait_system
        self.rng = np.random.default_rng(config.random_seed)

        # Core components
        self.inheritance = InheritanceEngine(config)
        self.classifier = ProcessingClassifier(config)
        self.drift_engine = TraitDriftEngine(config)
        self.attraction = AttractionModel(config)
        self.decision_model = DecisionModel(config)

        # State
        self.population: list[Agent] = []
        self.history: list[GenerationSnapshot] = []
        self._next_agent_id = 0

    def run(self, generations: int | None = None) -> list[GenerationSnapshot]:
        """Run the simulation for the specified number of generations."""
        generations = generations or self.config.generations_to_run
        self.population = self._create_initial_population()
        self.history = []

        for gen in range(generations):
            snapshot = self._run_generation(gen)
            self.history.append(snapshot)

        return self.history

    # ------------------------------------------------------------------
    # Initial population
    # ------------------------------------------------------------------
    def _create_initial_population(self) -> list[Agent]:
        """Generate the founding population."""
        pop: list[Agent] = []
        for i in range(self.config.initial_population):
            traits = self.ts.random_traits(self.rng)
            agent = Agent(
                id=self._new_id(),
                name=_generate_name(0, i),
                age=self.rng.integers(16, 35),
                generation=0,
                birth_order=self.rng.integers(1, 4),
                traits=traits,
                traits_at_birth=traits.copy(),
            )
            agent.processing_region = self.classifier.classify(agent)
            pop.append(agent)
        return pop

    # ------------------------------------------------------------------
    # Generation loop
    # ------------------------------------------------------------------
    def _run_generation(self, generation: int) -> GenerationSnapshot:
        events: dict[str, Any] = {
            "births": 0, "deaths": 0, "breakthroughs": 0, "pairs_formed": 0,
        }

        # === Phase 1: Age and trait drift ===
        for agent in self.population:
            agent.age += 1
            agent.traits = self.drift_engine.drift_traits(agent, self.rng)
            agent.traits = self.drift_engine.apply_region_effects(agent)

        # === Phase 2: Processing region updates ===
        for agent in self.population:
            agent.processing_region = self.classifier.classify(agent)
            self.drift_engine.update_suffering(agent)
            self.drift_engine.update_burnout(agent)

        # === Phase 3: Contribution and breakthroughs ===
        generation_contributions: list[float] = []
        for agent in self.population:
            contribution, breakthrough = self._calculate_contribution(agent)
            agent.record_generation(contribution)
            generation_contributions.append(contribution)
            if breakthrough:
                events["breakthroughs"] += 1

        # === Phase 4: Pairing ===
        min_age = self.config.relationship_config["pairing_min_age"]
        eligible = [a for a in self.population if a.is_eligible_for_pairing(min_age)]
        pairs = self._form_pairs(eligible)
        events["pairs_formed"] = len(pairs)

        # === Phase 5: Reproduction ===
        new_children: list[Agent] = []
        paired_agents = [
            (a, self._find_agent(a.partner_id))
            for a in self.population
            if a.partner_id is not None
        ]

        for p1, p2 in paired_agents:
            if p2 is None or not p1.is_alive or not p2.is_alive:
                continue
            if self._will_reproduce(p1, p2, generation):
                child = self._create_child(p1, p2, generation)
                if child is not None:
                    new_children.append(child)
                    events["births"] += 1

        self.population.extend(new_children)

        # === Phase 6: Mortality ===
        for agent in self.population:
            if not agent.is_alive:
                continue
            death_rate = self._mortality_rate(agent)
            if self.rng.random() < death_rate:
                agent.is_alive = False
                events["deaths"] += 1
                # Widen partner
                if agent.partner_id:
                    partner = self._find_agent(agent.partner_id)
                    if partner:
                        partner.partner_id = None
                        partner.relationship_status = "widowed"

        # Remove dead agents
        self.population = [a for a in self.population if a.is_alive]

        # === Phase 7: Build snapshot ===
        return self._build_snapshot(generation, events, generation_contributions)

    # ------------------------------------------------------------------
    # Contribution & breakthroughs
    # ------------------------------------------------------------------
    def _calculate_contribution(self, agent: Agent) -> tuple[float, bool]:
        """Calculate agent contribution and check for breakthrough."""
        cc = self.config.contribution_config
        region_mult = cc["region_multipliers"].get(
            agent.processing_region.value, 0.5
        )

        # Base contribution from traits
        creativity = agent.traits[self.ts.trait_index("creativity")]
        resilience = agent.traits[self.ts.trait_index("resilience")]
        conscientiousness = agent.traits[self.ts.trait_index("conscientiousness")]

        trait_contribution = (
            creativity * cc["creativity_weight"]
            + resilience * cc["resilience_weight"]
            + conscientiousness * cc["conscientiousness_weight"]
        )

        contribution = cc["base_contribution"] * region_mult * (0.5 + trait_contribution)
        contribution = float(np.clip(contribution, 0.0, 2.0))

        # Breakthrough check (only R3/R4 can break through)
        breakthrough = False
        if agent.processing_region in (ProcessingRegion.DEEP, ProcessingRegion.SACRIFICIAL):
            if contribution > cc["breakthrough_threshold"]:
                if self.rng.random() < cc["breakthrough_base_probability"]:
                    breakthrough = True

        return contribution, breakthrough

    # ------------------------------------------------------------------
    # Pairing
    # ------------------------------------------------------------------
    def _form_pairs(self, eligible: list[Agent]) -> list[tuple[Agent, Agent]]:
        """Form pairs from eligible agents, weighted by attraction."""
        pairs: list[tuple[Agent, Agent]] = []
        unpaired = list(eligible)
        self.rng.shuffle(unpaired)

        while len(unpaired) >= 2:
            a1 = unpaired.pop(0)

            # Calculate attraction to all remaining
            scores = np.array([
                self.attraction.calculate(a1, a2, self.rng) for a2 in unpaired
            ])
            scores = np.maximum(scores, 0.0)

            if scores.sum() <= 0:
                continue

            # Weighted random selection
            probs = scores / scores.sum()
            idx = self.rng.choice(len(unpaired), p=probs)
            partner = unpaired.pop(idx)

            # Link them
            a1.partner_id = partner.id
            partner.partner_id = a1.id
            a1.relationship_status = "paired"
            partner.relationship_status = "paired"

            pairs.append((a1, partner))

        return pairs

    # ------------------------------------------------------------------
    # Reproduction
    # ------------------------------------------------------------------
    def _will_reproduce(self, p1: Agent, p2: Agent, generation: int) -> bool:
        """Determine if a pair reproduces this generation."""
        fc = self.config.fertility_config

        # Check birth spacing
        for p in (p1, p2):
            if p.last_birth_generation is not None:
                gap = generation - p.last_birth_generation
                if gap < fc["min_birth_spacing_generations"]:
                    return False

        # Base probability influenced by societal pressure and traits
        base_prob = 0.3 + fc["societal_fertility_pressure"] * 0.2
        return bool(self.rng.random() < base_prob)

    def _create_child(
        self, parent1: Agent, parent2: Agent, generation: int,
    ) -> Agent | None:
        """Create a child agent via the inheritance engine."""
        fc = self.config.fertility_config

        # Child mortality check
        if self.rng.random() < fc["child_mortality_rate"]:
            # Child dies at birth — still counts for birth order
            parent1.children_ids.append("stillborn")
            parent2.children_ids.append("stillborn")
            return None

        # Determine birth order (count living + dead children of this pair)
        shared_children = set(parent1.children_ids) & set(parent2.children_ids)
        birth_order = len(shared_children) + 1

        # Generate traits via inheritance
        child_traits = self.inheritance.inherit(
            parent1, parent2, birth_order, self.population, self.rng
        )

        child = Agent(
            id=self._new_id(),
            name=_generate_name(generation, len(self.population)),
            age=0,
            generation=generation,
            birth_order=birth_order,
            traits=child_traits,
            traits_at_birth=child_traits.copy(),
            parent1_id=parent1.id,
            parent2_id=parent2.id,
        )
        child.processing_region = self.classifier.classify(child)

        # Record parentage
        parent1.children_ids.append(child.id)
        parent2.children_ids.append(child.id)
        parent1.last_birth_generation = generation
        parent2.last_birth_generation = generation

        # Mark outsider ancestry
        if parent1.is_descendant_of_outsider or parent2.is_descendant_of_outsider:
            child.extension_data["outsider_ancestor"] = True

        return child

    # ------------------------------------------------------------------
    # Mortality
    # ------------------------------------------------------------------
    def _mortality_rate(self, agent: Agent) -> float:
        """Calculate death probability for this generation."""
        base = self.config.base_mortality_rate
        age_component = agent.age * self.config.age_mortality_factor
        burnout_component = agent.burnout_level * self.config.burnout_mortality_factor
        return float(np.clip(base + age_component + burnout_component, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _find_agent(self, agent_id: str | None) -> Agent | None:
        if agent_id is None:
            return None
        for a in self.population:
            if a.id == agent_id:
                return a
        return None

    def _new_id(self) -> str:
        self._next_agent_id += 1
        return f"agent_{self._next_agent_id:06d}"

    def _build_snapshot(
        self, generation: int, events: dict[str, Any],
        contributions: list[float],
    ) -> GenerationSnapshot:
        pop = self.population
        if not pop:
            return GenerationSnapshot(
                generation=generation, population_size=0,
                births=events["births"], deaths=events["deaths"],
                breakthroughs=events["breakthroughs"],
                pairs_formed=events["pairs_formed"],
                trait_means=np.zeros(self.ts.count),
                trait_stds=np.zeros(self.ts.count),
                region_counts={r.value: 0 for r in ProcessingRegion},
                total_contribution=0, mean_contribution=0,
                mean_suffering=0, mean_age=0,
                birth_order_counts={}, events=events,
            )

        trait_matrix = np.array([a.traits for a in pop])
        ages = [a.age for a in pop]

        region_counts: dict[str, int] = {r.value: 0 for r in ProcessingRegion}
        birth_order_counts: dict[int, int] = {}
        for a in pop:
            region_counts[a.processing_region.value] += 1
            birth_order_counts[a.birth_order] = birth_order_counts.get(a.birth_order, 0) + 1

        total_contrib = sum(contributions) if contributions else 0.0

        return GenerationSnapshot(
            generation=generation,
            population_size=len(pop),
            births=events["births"],
            deaths=events["deaths"],
            breakthroughs=events["breakthroughs"],
            pairs_formed=events["pairs_formed"],
            trait_means=trait_matrix.mean(axis=0),
            trait_stds=trait_matrix.std(axis=0),
            region_counts=region_counts,
            total_contribution=total_contrib,
            mean_contribution=total_contrib / len(pop) if pop else 0,
            mean_suffering=float(np.mean([a.suffering for a in pop])),
            mean_age=float(np.mean(ages)),
            birth_order_counts=birth_order_counts,
            events=events,
        )
