"""
Main simulation engine.

Runs the multi-generational simulation with phases per generation,
extension hooks, outsider injection support, and full metrics collection.

Phase 2 enhancements:
- CognitiveCouncil voice updates
- Memory creation on breakthroughs/suffering
- Dissolution and infidelity via RelationshipManager
- FertilityManager for reproduction decisions
- Lore transmission and evolution
- Outsider injection and ripple tracking
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from seldon.core.agent import Agent
from seldon.core.attraction import AttractionModel
from seldon.core.config import ExperimentConfig
from seldon.core.council import CognitiveCouncil
from seldon.core.decision import DecisionContext, DecisionModel
from seldon.core.drift import TraitDriftEngine
from seldon.core.epigenetics import EpigeneticModel
from seldon.core.genetic_attribution import GeneticAttribution
from seldon.core.genetics import GeneticModel
from seldon.core.inheritance import InheritanceEngine
from seldon.core.processing import ProcessingClassifier, ProcessingRegion
from seldon.experiment.outsider import OutsiderInterface, RippleTracker
from seldon.extensions.registry import ExtensionRegistry
from seldon.social.fertility import FertilityManager
from seldon.social.lore import LoreEngine
from seldon.social.relationships import RelationshipManager


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
    2. Processing region updates + council voice
    3. Contribution & breakthroughs + memory creation
    4. Relationship dynamics (dissolution, infidelity, pairing)
    5. Reproduction + lore transmission
    6. Lore evolution
    7. Outsider injection + ripple tracking
    8. Mortality
    9. Record metrics
    """

    def __init__(
        self, config: ExperimentConfig,
        extensions: ExtensionRegistry | None = None,
    ):
        self.config = config
        self.ts = config.trait_system
        self.rng = np.random.default_rng(config.random_seed)

        # Extensions (empty registry by default — zero behavioral change)
        self.extensions: ExtensionRegistry = extensions or ExtensionRegistry()

        # Core components
        self.inheritance = InheritanceEngine(config)
        self.classifier = ProcessingClassifier(config)
        self.drift_engine = TraitDriftEngine(config)
        self.attraction = AttractionModel(config)
        self.decision_model = DecisionModel(config)

        # Phase 2 components
        self.council = CognitiveCouncil(config)
        self.lore_engine = LoreEngine(config)
        self.relationship_manager = RelationshipManager(config, self.attraction)
        self.fertility_manager = FertilityManager(config)
        self.outsider_interface = OutsiderInterface(config, self.classifier)
        self.ripple_tracker = RippleTracker(config)

        # Phase 8: Genetics/Epigenetics
        self.genetic_model = GeneticModel(config)
        self.epigenetic_model = EpigeneticModel(config)
        self.genetic_attribution = GeneticAttribution(config)

        # Wire extension modifier hooks into attraction and decision models
        self.attraction.set_extension_modifiers(
            [ext.modify_attraction for ext in self.extensions.get_enabled()]
        )
        self.decision_model.set_extension_modifiers(
            [ext.modify_decision for ext in self.extensions.get_enabled()]
        )

        # State
        self.population: list[Agent] = []
        self.history: list[GenerationSnapshot] = []
        self._next_agent_id = 0

    def run(self, generations: int | None = None) -> list[GenerationSnapshot]:
        """Run the simulation for the specified number of generations."""
        generations = generations or self.config.generations_to_run
        self.population = self._create_initial_population()
        self.history = []

        # Extension hook: simulation start
        for ext in self.extensions.get_enabled():
            ext.on_simulation_start(self.population, self.config)

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
            # Phase 8: Generate initial genome and epigenetic state
            if self.genetic_model.enabled:
                agent.genome = self.genetic_model.generate_initial_genome(
                    self.ts, agent.traits, self.rng,
                )
            if self.epigenetic_model.enabled:
                agent.epigenetic_state = self.epigenetic_model.initialize_state()
            pop.append(agent)
        return pop

    # ------------------------------------------------------------------
    # Generation loop
    # ------------------------------------------------------------------
    def _run_generation(self, generation: int) -> GenerationSnapshot:
        events: dict[str, Any] = {
            "births": 0, "deaths": 0, "breakthroughs": 0, "pairs_formed": 0,
            "dissolutions": 0, "infidelity_events": 0, "outsiders_injected": 0,
            "memories_created": 0, "lore_transmitted": 0,
        }

        # Extension hook: generation start
        for ext in self.extensions.get_enabled():
            ext.on_generation_start(generation, self.population, self.config)

        # === Phase 1: Age and trait drift ===
        for agent in self.population:
            agent.age += 1
            agent.traits = self.drift_engine.drift_traits(agent, self.rng)
            agent.traits = self.drift_engine.apply_region_effects(agent)

        # === Phase 1.5: Epigenetic updates ===
        if self.epigenetic_model.enabled:
            for agent in self.population:
                self.epigenetic_model.update_epigenetic_state(agent, self.ts)
                self.epigenetic_model.apply_epigenetic_modifiers(agent, self.ts)

        # === Phase 2: Processing region updates + council voice ===
        for agent in self.population:
            agent.processing_region = self.classifier.classify(agent)
            self.drift_engine.update_suffering(agent)
            self.drift_engine.update_burnout(agent)
            # Council voice update (no-op if disabled)
            agent.dominant_voice = self.council.get_dominant_voice(agent)

        # === Phase 3: Contribution and breakthroughs + memory ===
        generation_contributions: list[float] = []
        for agent in self.population:
            contribution, breakthrough = self._calculate_contribution(agent)
            agent.record_generation(contribution)
            generation_contributions.append(contribution)
            if breakthrough:
                events["breakthroughs"] += 1
                # Create breakthrough memory
                if self.config.lore_enabled:
                    mem = self.lore_engine.create_breakthrough_memory(
                        agent.id, generation, contribution,
                    )
                    agent.personal_memories.append(mem.to_dict())
                    events["memories_created"] += 1

            # Create suffering memory for agents in R4/R5
            if (self.config.lore_enabled
                    and agent.suffering > 0.6
                    and agent.processing_region in (
                        ProcessingRegion.SACRIFICIAL,
                        ProcessingRegion.PATHOLOGICAL,
                    )):
                mem = self.lore_engine.create_suffering_memory(
                    agent.id, generation, agent.suffering,
                )
                agent.personal_memories.append(mem.to_dict())
                events["memories_created"] += 1

        # === Phase 4: Relationship dynamics ===
        # 4a: Dissolution
        dissolved = self.relationship_manager.process_dissolutions(
            self.population, generation, self.rng,
        )
        events["dissolutions"] = len(dissolved)

        # 4b: Infidelity
        infidelity = self.relationship_manager.check_infidelity(
            self.population, generation, self.rng,
        )
        events["infidelity_events"] = len(infidelity)

        # 4c: Pairing (via RelationshipManager)
        pairs = self.relationship_manager.form_pairs(
            self.population, generation, self.rng,
        )
        events["pairs_formed"] = len(pairs)

        # === Phase 5: Reproduction + lore transmission ===
        new_children: list[Agent] = []
        paired_agents = [
            (a, self._find_agent(a.partner_id))
            for a in self.population
            if a.partner_id is not None
        ]

        for p1, p2 in paired_agents:
            if p2 is None or not p1.is_alive or not p2.is_alive:
                continue
            if self.fertility_manager.will_reproduce(p1, p2, generation, self.rng):
                child = self._create_child(p1, p2, generation)
                if child is not None:
                    new_children.append(child)
                    events["births"] += 1

                    # Extension hook: agent created
                    for ext in self.extensions.get_enabled():
                        ext.on_agent_created(child, (p1, p2), self.config)

                    # Lore transmission
                    if self.config.lore_enabled:
                        lore_count = self._transmit_lore(p1, p2, child)
                        events["lore_transmitted"] += lore_count

        self.population.extend(new_children)

        # === Phase 6: Lore evolution ===
        if self.config.lore_enabled:
            pop_memories = [
                a.personal_memories + a.inherited_lore
                for a in self.population
            ]
            self.lore_engine.evolve_societal_lore(pop_memories, generation, self.rng)

        # === Phase 7: Outsider injection + ripple tracking ===
        outsiders = self.outsider_interface.process_scheduled_injections(
            generation, self.rng,
        )
        for outsider in outsiders:
            self.population.append(outsider)
            self.ripple_tracker.track_injection(
                self.outsider_interface.injections[-1]
            )
        events["outsiders_injected"] = len(outsiders)

        self.ripple_tracker.track_generation(self.population, generation)

        # === Phase 8: Mortality ===
        for agent in self.population:
            if not agent.is_alive:
                continue
            death_rate = self._mortality_rate(agent)
            # Extension modifier hooks
            for ext in self.extensions.get_enabled():
                death_rate = ext.modify_mortality(agent, death_rate, self.config)
            death_rate = float(np.clip(death_rate, 0.0, 1.0))
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

        # Extension hooks: generation end + metrics
        for ext in self.extensions.get_enabled():
            ext.on_generation_end(generation, self.population, self.config)
        for ext in self.extensions.get_enabled():
            ext_metrics = ext.get_metrics(self.population)
            if ext_metrics:
                events[f"ext_{ext.name}"] = ext_metrics

        # === Phase 9: Build snapshot ===
        return self._build_snapshot(generation, events, generation_contributions)

    # ------------------------------------------------------------------
    # Lore transmission
    # ------------------------------------------------------------------
    def _transmit_lore(
        self, parent1: Agent, parent2: Agent, child: Agent,
    ) -> int:
        """Transmit memories from parents to child. Returns count transmitted."""
        count = 0
        for parent in (parent1, parent2):
            all_memories = parent.personal_memories + parent.inherited_lore
            for mem_dict in all_memories:
                from seldon.social.lore import Memory
                mem = Memory.from_dict(mem_dict)
                if self.lore_engine.should_transmit(mem, self.rng):
                    child_mem = self.lore_engine.transmit_to_child(mem, self.rng)
                    child.inherited_lore.append(child_mem.to_dict())
                    count += 1
        return count

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
    # Reproduction
    # ------------------------------------------------------------------
    def _create_child(
        self, parent1: Agent, parent2: Agent, generation: int,
    ) -> Agent | None:
        """Create a child agent via the inheritance engine."""
        # Child mortality check
        if self.fertility_manager.check_child_mortality(self.rng):
            # Child dies at birth — still counts for birth order
            parent1.children_ids.append("stillborn")
            parent2.children_ids.append("stillborn")
            return None

        # Maternal mortality check
        if self.fertility_manager.check_maternal_mortality(self.rng):
            # Parent dies in childbirth
            parent1.is_alive = False
            if parent2.partner_id == parent1.id:
                parent2.partner_id = None
                parent2.relationship_status = "widowed"

        # Determine birth order (count living + dead children of this pair)
        shared_children = set(parent1.children_ids) & set(parent2.children_ids)
        birth_order = len(shared_children) + 1

        # Generate traits via inheritance (with genetic integration if enabled)
        if self.genetic_model.enabled:
            child_traits, genome, epi_state, lineage = (
                self.inheritance.inherit_with_genetics(
                    parent1, parent2, birth_order, self.population, self.rng,
                )
            )
        else:
            child_traits = self.inheritance.inherit(
                parent1, parent2, birth_order, self.population, self.rng,
            )
            genome = {}
            epi_state = {}
            lineage = {}

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
            genome=genome,
            epigenetic_state=epi_state,
            genetic_lineage=lineage,
        )
        child.processing_region = self.classifier.classify(child)

        # Track genetic attribution
        if genome:
            self.genetic_attribution.track_inheritance(child, parent1, parent2)

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
