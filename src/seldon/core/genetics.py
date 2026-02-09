"""
Genetic model for the Seldon Sandbox.

Provides discrete allele-based genetics with Mendelian inheritance,
crossover, mutation, and trait expression. Each trait can have an
associated gene locus with two alleles (one from each parent).

Only a subset of traits have genetic components — the rest are
purely environmental/developmental. Gene-trait influence is
configurable via ``genetics_config["gene_trait_influence"]``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from seldon.core.agent import Agent
    from seldon.core.config import ExperimentConfig
    from seldon.core.traits import TraitSystem


# ---------------------------------------------------------------------------
# Gene locus definitions — maps locus name to trait name
# ---------------------------------------------------------------------------
# Not all traits have genetic components. Only ~10 of 15 compact traits are
# genetically linked; the rest are purely environmental.
GENE_LOCI: dict[str, str] = {
    "CREA_1": "creativity",
    "RESI_1": "resilience",
    "NEUR_1": "neuroticism",
    "EXTR_1": "extraversion",
    "CONS_1": "conscientiousness",
    "OPEN_1": "openness",
    "DEPT_1": "depth_drive",
    "AGRE_1": "agreeableness",
    "AMBI_1": "ambition",
    "EMPA_1": "empathy",
}

# Allele labels: uppercase = dominant, lowercase = recessive
DOMINANT = "A"
RECESSIVE = "a"


def _allele_expression(allele1: str, allele2: str) -> float:
    """
    Compute expression value for an allele pair.

    - AA (homozygous dominant): +modifier
    - Aa or aA (heterozygous): +modifier * 0.5
    - aa (homozygous recessive): -modifier
    """
    dominant_count = sum(1 for a in (allele1, allele2) if a == DOMINANT)
    if dominant_count == 2:
        return 1.0
    if dominant_count == 1:
        return 0.5
    return -1.0


class GeneticModel:
    """Handles allele-based genetics, crossover, mutation, and trait expression."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self._gc = config.genetics_config

    @property
    def enabled(self) -> bool:
        return self._gc.get("enabled", True)

    @property
    def mutation_rate(self) -> float:
        return self._gc.get("mutation_rate", 0.001)

    @property
    def crossover_rate(self) -> float:
        return self._gc.get("crossover_rate", 0.5)

    @property
    def dominance_modifier(self) -> float:
        return self._gc.get("dominance_modifier", 0.1)

    @property
    def gene_trait_influence(self) -> float:
        return self._gc.get("gene_trait_influence", 0.3)

    # ------------------------------------------------------------------
    # Genome generation
    # ------------------------------------------------------------------
    def generate_initial_genome(
        self, trait_system: TraitSystem, traits: np.ndarray,
        rng: np.random.Generator,
    ) -> dict[str, tuple[str, str]]:
        """
        Generate an initial genome consistent with the given trait values.

        For each gene locus, allele pair is chosen probabilistically based
        on the trait value: high trait → more likely to have dominant alleles.
        """
        if not self.enabled:
            return {}

        genome: dict[str, tuple[str, str]] = {}
        for locus, trait_name in GENE_LOCI.items():
            try:
                idx = trait_system.trait_index(trait_name)
            except KeyError:
                continue
            trait_val = float(traits[idx])
            # Map trait value [0,1] to probability of dominant allele
            p_dominant = np.clip(trait_val, 0.1, 0.9)
            allele1 = DOMINANT if rng.random() < p_dominant else RECESSIVE
            allele2 = DOMINANT if rng.random() < p_dominant else RECESSIVE
            genome[locus] = (allele1, allele2)
        return genome

    # ------------------------------------------------------------------
    # Crossover (reproduction)
    # ------------------------------------------------------------------
    def crossover(
        self,
        parent1_genome: dict[str, tuple[str, str]],
        parent2_genome: dict[str, tuple[str, str]],
        rng: np.random.Generator,
    ) -> dict[str, tuple[str, str]]:
        """
        Mendelian crossover: each parent contributes one allele per locus.

        With ``crossover_rate`` probability, adjacent loci swap linkage
        (simulating chromosomal crossover). Mutation flips individual alleles
        with ``mutation_rate`` probability.
        """
        if not self.enabled:
            return {}

        child_genome: dict[str, tuple[str, str]] = {}
        loci = list(GENE_LOCI.keys())
        swap = False

        for i, locus in enumerate(loci):
            # Crossover event between loci
            if i > 0 and rng.random() < self.crossover_rate:
                swap = not swap

            p1 = parent1_genome.get(locus, (RECESSIVE, RECESSIVE))
            p2 = parent2_genome.get(locus, (RECESSIVE, RECESSIVE))

            if swap:
                # Swap which parent contributes which position
                allele_from_p1 = p1[1]  # Second allele from parent1
                allele_from_p2 = p2[0]  # First allele from parent2
            else:
                allele_from_p1 = p1[rng.integers(0, 2)]
                allele_from_p2 = p2[rng.integers(0, 2)]

            # Mutation
            allele_from_p1 = self._maybe_mutate(allele_from_p1, rng)
            allele_from_p2 = self._maybe_mutate(allele_from_p2, rng)

            child_genome[locus] = (allele_from_p1, allele_from_p2)

        return child_genome

    def _maybe_mutate(self, allele: str, rng: np.random.Generator) -> str:
        """Flip an allele with mutation_rate probability."""
        if rng.random() < self.mutation_rate:
            return RECESSIVE if allele == DOMINANT else DOMINANT
        return allele

    # ------------------------------------------------------------------
    # Trait expression
    # ------------------------------------------------------------------
    def express_traits(
        self, genome: dict[str, tuple[str, str]],
        base_traits: np.ndarray,
        trait_system: TraitSystem,
    ) -> np.ndarray:
        """
        Modify base trait values based on allele expression.

        For each gene locus, compute allele expression and apply
        ``dominance_modifier * gene_trait_influence`` adjustment.
        """
        if not self.enabled or not genome:
            return base_traits.copy()

        modified = base_traits.copy()
        for locus, trait_name in GENE_LOCI.items():
            if locus not in genome:
                continue
            try:
                idx = trait_system.trait_index(trait_name)
            except KeyError:
                continue
            allele1, allele2 = genome[locus]
            expression = _allele_expression(allele1, allele2)
            modifier = expression * self.dominance_modifier * self.gene_trait_influence
            modified[idx] = np.clip(modified[idx] + modifier, 0.0, 1.0)

        return modified

    # ------------------------------------------------------------------
    # Analysis helpers
    # ------------------------------------------------------------------
    def get_allele_frequencies(
        self, population: list[Agent],
    ) -> dict[str, dict[str, float]]:
        """
        Compute allele frequencies across population for each locus.

        Returns dict[locus_name, {"dominant_freq": float, "recessive_freq": float}].
        """
        freqs: dict[str, dict[str, float]] = {}
        for locus in GENE_LOCI:
            total = 0
            dominant_count = 0
            for agent in population:
                if not agent.is_alive or locus not in agent.genome:
                    continue
                a1, a2 = agent.genome[locus]
                total += 2
                dominant_count += sum(1 for a in (a1, a2) if a == DOMINANT)
            if total > 0:
                freqs[locus] = {
                    "dominant_freq": dominant_count / total,
                    "recessive_freq": (total - dominant_count) / total,
                }
            else:
                freqs[locus] = {"dominant_freq": 0.0, "recessive_freq": 0.0}
        return freqs
