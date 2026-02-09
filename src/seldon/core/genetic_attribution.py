"""
Genetic attribution for the Seldon Sandbox.

Tracks which genetic markers came from which ancestor, computes
trait-gene correlations, and generates ancestry reports explaining
the genetic vs environmental vs epigenetic sources of agent traits.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from seldon.core.genetics import GENE_LOCI, _allele_expression

if TYPE_CHECKING:
    from seldon.core.agent import Agent
    from seldon.core.config import ExperimentConfig
    from seldon.core.traits import TraitSystem


class GeneticAttribution:
    """Tracks inheritance pathways and computes trait-gene correlations."""

    def __init__(self, config: ExperimentConfig):
        self.config = config

    # ------------------------------------------------------------------
    # Lineage tracking
    # ------------------------------------------------------------------
    def track_inheritance(
        self,
        child: Agent,
        parent1: Agent,
        parent2: Agent,
    ) -> dict[str, Any]:
        """
        Record which alleles came from which parent for each locus.

        Returns a lineage dict stored in ``child.genetic_lineage``.
        """
        lineage: dict[str, Any] = {
            "parent1_id": parent1.id,
            "parent2_id": parent2.id,
            "locus_origins": {},
        }

        for locus in GENE_LOCI:
            child_alleles = child.genome.get(locus)
            p1_alleles = parent1.genome.get(locus)
            p2_alleles = parent2.genome.get(locus)

            if child_alleles and p1_alleles and p2_alleles:
                lineage["locus_origins"][locus] = {
                    "child": child_alleles,
                    "parent1_alleles": p1_alleles,
                    "parent2_alleles": p2_alleles,
                }

        # Carry forward grandparent info if available
        if parent1.genetic_lineage.get("parent1_id"):
            lineage["grandparent1_ids"] = (
                parent1.genetic_lineage.get("parent1_id"),
                parent1.genetic_lineage.get("parent2_id"),
            )
        if parent2.genetic_lineage.get("parent1_id"):
            lineage["grandparent2_ids"] = (
                parent2.genetic_lineage.get("parent1_id"),
                parent2.genetic_lineage.get("parent2_id"),
            )

        child.genetic_lineage = lineage
        return lineage

    # ------------------------------------------------------------------
    # Trait-gene correlation
    # ------------------------------------------------------------------
    def compute_trait_gene_correlation(
        self,
        population: list[Agent],
        trait_system: TraitSystem,
    ) -> dict[str, dict[str, float]]:
        """
        For each gene locus, compute correlation between allele expression
        and actual trait values across the population.

        Returns dict[locus_name, {"correlation": float, "n_samples": int}].
        """
        alive = [a for a in population if a.is_alive and a.genome]
        if len(alive) < 3:
            return {}

        results: dict[str, dict[str, float]] = {}

        for locus, trait_name in GENE_LOCI.items():
            try:
                idx = trait_system.trait_index(trait_name)
            except KeyError:
                continue

            expressions = []
            trait_vals = []
            for agent in alive:
                if locus not in agent.genome:
                    continue
                a1, a2 = agent.genome[locus]
                expressions.append(_allele_expression(a1, a2))
                trait_vals.append(float(agent.traits[idx]))

            if len(expressions) < 3:
                continue

            expr_arr = np.array(expressions)
            trait_arr = np.array(trait_vals)

            # Pearson correlation
            if np.std(expr_arr) > 0 and np.std(trait_arr) > 0:
                corr = float(np.corrcoef(expr_arr, trait_arr)[0, 1])
            else:
                corr = 0.0

            results[locus] = {
                "trait": trait_name,
                "correlation": round(corr, 4),
                "n_samples": len(expressions),
            }

        return results

    # ------------------------------------------------------------------
    # Ancestry report
    # ------------------------------------------------------------------
    def get_ancestry_report(
        self,
        agent: Agent,
        all_agents: dict[str, Agent],
        trait_system: TraitSystem,
    ) -> dict[str, Any]:
        """
        Generate an ancestry report explaining trait sources.

        Breaks down each genetically-linked trait into:
        - genetic_factor: contribution from allele expression
        - epigenetic_factor: contribution from active markers
        - environmental_factor: remainder (drift, experience)
        """
        report: dict[str, Any] = {
            "agent_id": agent.id,
            "trait_breakdown": {},
        }

        gc = self.config.genetics_config
        gene_influence = gc.get("gene_trait_influence", 0.3)
        dominance_mod = gc.get("dominance_modifier", 0.1)

        for locus, trait_name in GENE_LOCI.items():
            try:
                idx = trait_system.trait_index(trait_name)
            except KeyError:
                continue

            current_val = float(agent.traits[idx])
            birth_val = float(agent.traits_at_birth[idx])

            # Genetic factor
            genetic_factor = 0.0
            if locus in agent.genome:
                a1, a2 = agent.genome[locus]
                expression = _allele_expression(a1, a2)
                genetic_factor = abs(expression * dominance_mod * gene_influence)

            # Epigenetic factor
            epigenetic_factor = 0.0
            for marker_name, active in agent.epigenetic_state.items():
                if not active:
                    continue
                from seldon.core.epigenetics import DEFAULT_MARKERS
                for m in DEFAULT_MARKERS:
                    if m.name == marker_name and m.target_trait == trait_name:
                        epigenetic_factor += abs(m.modifier)

            # Environmental factor = everything else
            total_explained = genetic_factor + epigenetic_factor
            drift = abs(current_val - birth_val)
            environmental_factor = max(0.0, drift - total_explained)

            # Normalize to fractions
            total = genetic_factor + epigenetic_factor + environmental_factor
            if total > 0:
                genetic_frac = genetic_factor / total
                epigenetic_frac = epigenetic_factor / total
                environmental_frac = environmental_factor / total
            else:
                genetic_frac = gene_influence
                epigenetic_frac = 0.0
                environmental_frac = 1.0 - gene_influence

            # Parent attribution
            parent_info = {}
            if agent.genetic_lineage.get("locus_origins", {}).get(locus):
                origin = agent.genetic_lineage["locus_origins"][locus]
                parent_info = {
                    "parent1_alleles": origin.get("parent1_alleles"),
                    "parent2_alleles": origin.get("parent2_alleles"),
                }

            report["trait_breakdown"][trait_name] = {
                "current_value": round(current_val, 4),
                "birth_value": round(birth_val, 4),
                "alleles": agent.genome.get(locus),
                "genetic_factor": round(genetic_frac, 4),
                "epigenetic_factor": round(epigenetic_frac, 4),
                "environmental_factor": round(environmental_frac, 4),
                **parent_info,
            }

        return report
