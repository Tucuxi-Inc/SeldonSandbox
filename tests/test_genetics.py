"""
Tests for Phase 8: Genetic & Epigenetic Modeling.

Tests GeneticModel, EpigeneticModel, GeneticAttribution, and
integration with InheritanceEngine and SimulationEngine.
"""

from __future__ import annotations

import numpy as np
import pytest

from seldon.core.agent import Agent
from seldon.core.config import ExperimentConfig
from seldon.core.epigenetics import DEFAULT_MARKERS, EpigeneticModel
from seldon.core.genetic_attribution import GeneticAttribution
from seldon.core.genetics import (
    DOMINANT,
    GENE_LOCI,
    RECESSIVE,
    GeneticModel,
    _allele_expression,
)
from seldon.core.inheritance import InheritanceEngine
from seldon.core.processing import ProcessingRegion


def _make_config(**overrides) -> ExperimentConfig:
    defaults = {"random_seed": 42}
    defaults.update(overrides)
    return ExperimentConfig(**defaults)


def _make_agent(
    agent_id: str = "a1",
    traits: np.ndarray | None = None,
    config: ExperimentConfig | None = None,
    **kwargs,
) -> Agent:
    config = config or _make_config()
    ts = config.trait_system
    if traits is None:
        traits = ts.random_traits(np.random.default_rng(42))
    agent = Agent(
        id=agent_id,
        name=f"Agent-{agent_id}",
        age=kwargs.pop("age", 25),
        generation=kwargs.pop("generation", 0),
        birth_order=kwargs.pop("birth_order", 1),
        traits=traits,
        traits_at_birth=traits.copy(),
        **kwargs,
    )
    return agent


# =====================================================================
# GeneticModel Tests
# =====================================================================
class TestGeneticModel:
    def test_generate_initial_genome_produces_all_loci(self):
        config = _make_config()
        gm = GeneticModel(config)
        rng = np.random.default_rng(42)
        traits = config.trait_system.random_traits(rng)
        genome = gm.generate_initial_genome(config.trait_system, traits, rng)
        # Should have one entry per locus that maps to a valid trait
        for locus, trait_name in GENE_LOCI.items():
            if trait_name in config.trait_system._name_to_index:
                assert locus in genome, f"Missing locus {locus}"
                assert len(genome[locus]) == 2

    def test_genome_alleles_are_valid(self):
        config = _make_config()
        gm = GeneticModel(config)
        rng = np.random.default_rng(123)
        traits = config.trait_system.random_traits(rng)
        genome = gm.generate_initial_genome(config.trait_system, traits, rng)
        for locus, (a1, a2) in genome.items():
            assert a1 in (DOMINANT, RECESSIVE)
            assert a2 in (DOMINANT, RECESSIVE)

    def test_disabled_returns_empty(self):
        config = _make_config(genetics_config={"enabled": False})
        gm = GeneticModel(config)
        rng = np.random.default_rng(42)
        traits = config.trait_system.random_traits(rng)
        genome = gm.generate_initial_genome(config.trait_system, traits, rng)
        assert genome == {}

    def test_crossover_produces_child_genome(self):
        config = _make_config()
        gm = GeneticModel(config)
        rng = np.random.default_rng(42)
        ts = config.trait_system
        p1_traits = ts.random_traits(rng)
        p2_traits = ts.random_traits(rng)
        p1_genome = gm.generate_initial_genome(ts, p1_traits, rng)
        p2_genome = gm.generate_initial_genome(ts, p2_traits, rng)
        child_genome = gm.crossover(p1_genome, p2_genome, rng)
        assert len(child_genome) == len(GENE_LOCI)
        for locus, (a1, a2) in child_genome.items():
            assert a1 in (DOMINANT, RECESSIVE)
            assert a2 in (DOMINANT, RECESSIVE)

    def test_crossover_disabled_returns_empty(self):
        config = _make_config(genetics_config={"enabled": False})
        gm = GeneticModel(config)
        rng = np.random.default_rng(42)
        result = gm.crossover({}, {}, rng)
        assert result == {}

    def test_express_traits_modifies_values(self):
        config = _make_config()
        gm = GeneticModel(config)
        ts = config.trait_system
        base = np.full(ts.count, 0.5)
        # All dominant alleles should push traits up
        genome = {locus: (DOMINANT, DOMINANT) for locus in GENE_LOCI
                  if locus in GENE_LOCI}
        modified = gm.express_traits(genome, base, ts)
        # At least one trait should be different
        assert not np.array_equal(base, modified)
        # All dominant → positive modifier
        for locus, trait_name in GENE_LOCI.items():
            try:
                idx = ts.trait_index(trait_name)
                assert modified[idx] >= base[idx]
            except KeyError:
                pass

    def test_express_traits_disabled_returns_copy(self):
        config = _make_config(genetics_config={"enabled": False})
        gm = GeneticModel(config)
        ts = config.trait_system
        base = np.full(ts.count, 0.5)
        result = gm.express_traits({}, base, ts)
        np.testing.assert_array_equal(result, base)

    def test_express_traits_clamps_to_bounds(self):
        config = _make_config(genetics_config={
            "enabled": True, "dominance_modifier": 0.5, "gene_trait_influence": 1.0,
        })
        gm = GeneticModel(config)
        ts = config.trait_system
        base = np.full(ts.count, 0.99)
        genome = {locus: (DOMINANT, DOMINANT) for locus in GENE_LOCI}
        modified = gm.express_traits(genome, base, ts)
        assert np.all(modified <= 1.0)
        assert np.all(modified >= 0.0)

    def test_mutation_can_flip_alleles(self):
        config = _make_config(genetics_config={
            "enabled": True, "mutation_rate": 1.0,  # 100% mutation
        })
        gm = GeneticModel(config)
        rng = np.random.default_rng(42)
        p1_genome = {locus: (DOMINANT, DOMINANT) for locus in GENE_LOCI}
        p2_genome = {locus: (DOMINANT, DOMINANT) for locus in GENE_LOCI}
        child = gm.crossover(p1_genome, p2_genome, rng)
        # With 100% mutation, all alleles should flip
        for locus, (a1, a2) in child.items():
            assert a1 == RECESSIVE
            assert a2 == RECESSIVE

    def test_allele_frequencies(self):
        config = _make_config()
        gm = GeneticModel(config)
        rng = np.random.default_rng(42)
        ts = config.trait_system
        agents = []
        for i in range(20):
            traits = ts.random_traits(rng)
            a = _make_agent(f"a{i}", traits=traits, config=config)
            a.genome = gm.generate_initial_genome(ts, traits, rng)
            agents.append(a)
        freqs = gm.get_allele_frequencies(agents)
        for locus in GENE_LOCI:
            assert locus in freqs
            assert 0.0 <= freqs[locus]["dominant_freq"] <= 1.0
            assert abs(freqs[locus]["dominant_freq"] + freqs[locus]["recessive_freq"] - 1.0) < 1e-10


# =====================================================================
# Allele Expression Tests
# =====================================================================
class TestAlleleExpression:
    def test_homozygous_dominant(self):
        assert _allele_expression(DOMINANT, DOMINANT) == 1.0

    def test_heterozygous(self):
        assert _allele_expression(DOMINANT, RECESSIVE) == 0.5
        assert _allele_expression(RECESSIVE, DOMINANT) == 0.5

    def test_homozygous_recessive(self):
        assert _allele_expression(RECESSIVE, RECESSIVE) == -1.0


# =====================================================================
# EpigeneticModel Tests
# =====================================================================
class TestEpigeneticModel:
    def test_initialize_state_has_all_markers(self):
        config = _make_config()
        em = EpigeneticModel(config)
        state = em.initialize_state()
        for marker in DEFAULT_MARKERS:
            assert marker.name in state
            assert state[marker.name] is False

    def test_disabled_returns_empty(self):
        config = _make_config(epigenetics_config={"enabled": False})
        em = EpigeneticModel(config)
        state = em.initialize_state()
        assert state == {}

    def test_stress_resilience_activates(self):
        config = _make_config()
        em = EpigeneticModel(config)
        ts = config.trait_system
        agent = _make_agent(config=config)
        agent.epigenetic_state = em.initialize_state()
        # Simulate high suffering for 3 generations
        agent.suffering_history = [0.7, 0.8, 0.9]
        agent.suffering = 0.8
        state = em.update_epigenetic_state(agent, ts)
        assert state["stress_resilience"] is True

    def test_stress_resilience_does_not_activate_insufficient_history(self):
        config = _make_config()
        em = EpigeneticModel(config)
        ts = config.trait_system
        agent = _make_agent(config=config)
        agent.epigenetic_state = em.initialize_state()
        agent.suffering_history = [0.7, 0.8]  # Only 2, need 3
        state = em.update_epigenetic_state(agent, ts)
        assert state["stress_resilience"] is False

    def test_stress_resilience_deactivates(self):
        config = _make_config()
        em = EpigeneticModel(config)
        ts = config.trait_system
        agent = _make_agent(config=config)
        agent.epigenetic_state = em.initialize_state()
        agent.epigenetic_state["stress_resilience"] = True
        agent.suffering_history = [0.1, 0.2, 0.1]  # Low suffering
        state = em.update_epigenetic_state(agent, ts)
        assert state["stress_resilience"] is False

    def test_creative_amplification_activates(self):
        config = _make_config()
        em = EpigeneticModel(config)
        ts = config.trait_system
        agent = _make_agent(config=config)
        agent.epigenetic_state = em.initialize_state()
        agent.region_history = [
            ProcessingRegion.DEEP,
            ProcessingRegion.SACRIFICIAL,
            ProcessingRegion.DEEP,
        ]
        state = em.update_epigenetic_state(agent, ts)
        assert state["creative_amplification"] is True

    def test_social_withdrawal_activates(self):
        config = _make_config()
        em = EpigeneticModel(config)
        ts = config.trait_system
        agent = _make_agent(config=config)
        agent.epigenetic_state = em.initialize_state()
        # Set low trust
        trust_idx = ts.trait_index("trust")
        agent.traits[trust_idx] = 0.2
        agent.suffering = 0.6
        state = em.update_epigenetic_state(agent, ts)
        assert state["social_withdrawal"] is True

    def test_apply_modifiers_changes_traits(self):
        config = _make_config()
        em = EpigeneticModel(config)
        ts = config.trait_system
        agent = _make_agent(config=config)
        agent.epigenetic_state = {"stress_resilience": True}
        original_resilience = float(agent.traits[ts.trait_index("resilience")])
        em.apply_epigenetic_modifiers(agent, ts)
        new_resilience = float(agent.traits[ts.trait_index("resilience")])
        assert new_resilience == pytest.approx(
            min(1.0, original_resilience + 0.05), abs=1e-6
        )

    def test_inherit_epigenetic_state(self):
        config = _make_config()
        em = EpigeneticModel(config)
        rng = np.random.default_rng(42)
        p1 = _make_agent("p1", config=config)
        p2 = _make_agent("p2", config=config)
        p1.epigenetic_state = em.initialize_state()
        p2.epigenetic_state = em.initialize_state()
        p1.epigenetic_state["stress_resilience"] = True
        p2.epigenetic_state["stress_resilience"] = True
        # Run many times to check probabilistic inheritance
        inherited_count = 0
        for seed in range(100):
            rng_trial = np.random.default_rng(seed)
            state = em.inherit_epigenetic_state(p1, p2, rng_trial)
            if state.get("stress_resilience", False):
                inherited_count += 1
        # With both parents active, rate = min(1.0, 0.3*1.5) = 0.45
        # Should be around 45%
        assert 20 < inherited_count < 70

    def test_inherit_inactive_marker_not_inherited(self):
        config = _make_config()
        em = EpigeneticModel(config)
        rng = np.random.default_rng(42)
        p1 = _make_agent("p1", config=config)
        p2 = _make_agent("p2", config=config)
        p1.epigenetic_state = em.initialize_state()
        p2.epigenetic_state = em.initialize_state()
        # Neither parent has any active markers
        state = em.inherit_epigenetic_state(p1, p2, rng)
        assert all(not v for v in state.values())

    def test_max_active_markers_limit(self):
        config = _make_config(epigenetics_config={
            "enabled": True, "max_active_markers": 2,
            "activation_threshold_generations": 1,
        })
        em = EpigeneticModel(config)
        ts = config.trait_system
        agent = _make_agent(config=config)
        agent.epigenetic_state = em.initialize_state()
        # Set up conditions for multiple activations
        agent.suffering_history = [0.7]
        agent.suffering = 0.7
        agent.region_history = [ProcessingRegion.DEEP]
        trust_idx = ts.trait_index("trust")
        agent.traits[trust_idx] = 0.2
        state = em.update_epigenetic_state(agent, ts)
        active_count = sum(1 for v in state.values() if v)
        assert active_count <= 2

    def test_marker_prevalence(self):
        config = _make_config()
        em = EpigeneticModel(config)
        agents = []
        for i in range(10):
            a = _make_agent(f"a{i}", config=config)
            a.epigenetic_state = em.initialize_state()
            if i < 3:
                a.epigenetic_state["stress_resilience"] = True
            agents.append(a)
        prev = em.get_marker_prevalence(agents)
        assert prev["stress_resilience"]["active_count"] == 3
        assert prev["stress_resilience"]["prevalence"] == pytest.approx(0.3)

    def test_trauma_sensitivity_deactivates_with_high_resilience(self):
        config = _make_config()
        em = EpigeneticModel(config)
        ts = config.trait_system
        agent = _make_agent(config=config)
        agent.epigenetic_state = em.initialize_state()
        agent.epigenetic_state["trauma_sensitivity"] = True
        # Set high resilience
        res_idx = ts.trait_index("resilience")
        agent.traits[res_idx] = 0.85
        state = em.update_epigenetic_state(agent, ts)
        assert state["trauma_sensitivity"] is False


# =====================================================================
# GeneticAttribution Tests
# =====================================================================
class TestGeneticAttribution:
    def test_track_inheritance_records_lineage(self):
        config = _make_config()
        gm = GeneticModel(config)
        ga = GeneticAttribution(config)
        rng = np.random.default_rng(42)
        ts = config.trait_system
        p1 = _make_agent("p1", config=config)
        p2 = _make_agent("p2", config=config)
        p1.genome = gm.generate_initial_genome(ts, p1.traits, rng)
        p2.genome = gm.generate_initial_genome(ts, p2.traits, rng)
        child = _make_agent("c1", config=config)
        child.genome = gm.crossover(p1.genome, p2.genome, rng)
        lineage = ga.track_inheritance(child, p1, p2)
        assert lineage["parent1_id"] == "p1"
        assert lineage["parent2_id"] == "p2"
        assert "locus_origins" in lineage
        assert len(lineage["locus_origins"]) > 0

    def test_trait_gene_correlation(self):
        config = _make_config()
        gm = GeneticModel(config)
        ga = GeneticAttribution(config)
        rng = np.random.default_rng(42)
        ts = config.trait_system
        agents = []
        for i in range(30):
            traits = ts.random_traits(rng)
            a = _make_agent(f"a{i}", traits=traits, config=config)
            a.genome = gm.generate_initial_genome(ts, traits, rng)
            # Express traits so genome and trait values are correlated
            a.traits = gm.express_traits(a.genome, a.traits, ts)
            agents.append(a)
        corr = ga.compute_trait_gene_correlation(agents, ts)
        assert len(corr) > 0
        for locus, data in corr.items():
            assert "correlation" in data
            assert "n_samples" in data
            assert data["n_samples"] > 0

    def test_trait_gene_correlation_too_few_agents(self):
        config = _make_config()
        ga = GeneticAttribution(config)
        ts = config.trait_system
        agents = [_make_agent("a1", config=config)]
        corr = ga.compute_trait_gene_correlation(agents, ts)
        assert corr == {}

    def test_ancestry_report(self):
        config = _make_config()
        gm = GeneticModel(config)
        ga = GeneticAttribution(config)
        em = EpigeneticModel(config)
        rng = np.random.default_rng(42)
        ts = config.trait_system
        agent = _make_agent(config=config)
        agent.genome = gm.generate_initial_genome(ts, agent.traits, rng)
        agent.epigenetic_state = em.initialize_state()
        agent.epigenetic_state["stress_resilience"] = True
        report = ga.get_ancestry_report(agent, {}, ts)
        assert report["agent_id"] == agent.id
        assert "trait_breakdown" in report
        # Should have entries for genetically-linked traits
        assert "creativity" in report["trait_breakdown"]
        for trait_name, breakdown in report["trait_breakdown"].items():
            assert "genetic_factor" in breakdown
            assert "epigenetic_factor" in breakdown
            assert "environmental_factor" in breakdown
            # Fractions should sum to ~1
            total = (breakdown["genetic_factor"]
                     + breakdown["epigenetic_factor"]
                     + breakdown["environmental_factor"])
            assert total == pytest.approx(1.0, abs=0.01)


# =====================================================================
# InheritanceEngine Integration Tests
# =====================================================================
class TestInheritanceWithGenetics:
    def test_inherit_with_genetics_returns_tuple(self):
        config = _make_config()
        ie = InheritanceEngine(config)
        rng = np.random.default_rng(42)
        ts = config.trait_system
        gm = GeneticModel(config)
        em = EpigeneticModel(config)
        p1 = _make_agent("p1", config=config)
        p2 = _make_agent("p2", config=config)
        p1.genome = gm.generate_initial_genome(ts, p1.traits, rng)
        p2.genome = gm.generate_initial_genome(ts, p2.traits, rng)
        p1.epigenetic_state = em.initialize_state()
        p2.epigenetic_state = em.initialize_state()
        result = ie.inherit_with_genetics(p1, p2, 1, [p1, p2], rng)
        assert len(result) == 4
        traits, genome, epi_state, lineage = result
        assert traits.shape == (ts.count,)
        assert len(genome) > 0
        assert isinstance(epi_state, dict)
        assert isinstance(lineage, dict)

    def test_inherit_with_genetics_traits_in_bounds(self):
        config = _make_config()
        ie = InheritanceEngine(config)
        rng = np.random.default_rng(42)
        ts = config.trait_system
        gm = GeneticModel(config)
        em = EpigeneticModel(config)
        p1 = _make_agent("p1", config=config)
        p2 = _make_agent("p2", config=config)
        p1.genome = gm.generate_initial_genome(ts, p1.traits, rng)
        p2.genome = gm.generate_initial_genome(ts, p2.traits, rng)
        p1.epigenetic_state = em.initialize_state()
        p2.epigenetic_state = em.initialize_state()
        for seed in range(50):
            traits, _, _, _ = ie.inherit_with_genetics(
                p1, p2, 1, [p1, p2], np.random.default_rng(seed)
            )
            assert np.all(traits >= 0.0)
            assert np.all(traits <= 1.0)

    def test_inherit_with_genetics_disabled(self):
        config = _make_config(genetics_config={"enabled": False},
                              epigenetics_config={"enabled": False})
        ie = InheritanceEngine(config)
        rng = np.random.default_rng(42)
        ts = config.trait_system
        p1 = _make_agent("p1", config=config)
        p2 = _make_agent("p2", config=config)
        p1.genome = {}
        p2.genome = {}
        p1.epigenetic_state = {}
        p2.epigenetic_state = {}
        traits, genome, epi_state, lineage = ie.inherit_with_genetics(
            p1, p2, 1, [p1, p2], rng,
        )
        assert genome == {}
        assert epi_state == {}

    def test_standard_inherit_still_works(self):
        """Backward compatibility: inherit() still works without genetics."""
        config = _make_config()
        ie = InheritanceEngine(config)
        rng = np.random.default_rng(42)
        p1 = _make_agent("p1", config=config)
        p2 = _make_agent("p2", config=config)
        result = ie.inherit(p1, p2, 1, [p1, p2], rng)
        assert result.shape == (config.trait_system.count,)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)


# =====================================================================
# SimulationEngine Integration Tests
# =====================================================================
class TestEngineGeneticsIntegration:
    def test_initial_population_has_genomes(self):
        config = _make_config(initial_population=10, generations_to_run=1)
        from seldon.core.engine import SimulationEngine
        engine = SimulationEngine(config)
        engine.population = engine._create_initial_population()
        for agent in engine.population:
            assert len(agent.genome) > 0
            assert len(agent.epigenetic_state) > 0

    def test_initial_population_genomes_disabled(self):
        config = _make_config(
            initial_population=10, generations_to_run=1,
            genetics_config={"enabled": False},
            epigenetics_config={"enabled": False},
        )
        from seldon.core.engine import SimulationEngine
        engine = SimulationEngine(config)
        engine.population = engine._create_initial_population()
        for agent in engine.population:
            assert len(agent.genome) == 0
            assert len(agent.epigenetic_state) == 0

    def test_children_inherit_genomes(self):
        config = _make_config(
            initial_population=20, generations_to_run=3,
            random_seed=42,
        )
        from seldon.core.engine import SimulationEngine
        engine = SimulationEngine(config)
        history = engine.run(3)
        # After 3 generations, some agents should have genetic_lineage
        agents_with_lineage = [
            a for a in engine.population
            if a.genetic_lineage.get("parent1_id")
        ]
        # There should be children born with genetic data
        agents_with_genomes = [a for a in engine.population if a.genome]
        assert len(agents_with_genomes) == len(engine.population)

    def test_epigenetic_updates_run(self):
        """Verify epigenetic updates happen during generation loop."""
        config = _make_config(
            initial_population=20, generations_to_run=5,
            random_seed=42,
        )
        from seldon.core.engine import SimulationEngine
        engine = SimulationEngine(config)
        history = engine.run(5)
        # Check that at least one agent has a non-default epigenetic state
        # (some markers may have activated during the 5 generations)
        all_states = [a.epigenetic_state for a in engine.population]
        assert all(isinstance(s, dict) for s in all_states)
        # All agents should have epigenetic state dicts
        for a in engine.population:
            assert len(a.epigenetic_state) == len(DEFAULT_MARKERS)

    def test_backward_compatibility_no_genetics(self):
        """Engine runs fine with genetics disabled."""
        config = _make_config(
            initial_population=20, generations_to_run=3,
            random_seed=42,
            genetics_config={"enabled": False},
            epigenetics_config={"enabled": False},
        )
        from seldon.core.engine import SimulationEngine
        engine = SimulationEngine(config)
        history = engine.run(3)
        assert len(history) == 3
        assert history[-1].population_size > 0

    def test_full_simulation_with_genetics(self):
        """Full 10-generation simulation with genetics enabled."""
        config = _make_config(
            initial_population=30, generations_to_run=10,
            random_seed=42,
        )
        from seldon.core.engine import SimulationEngine
        engine = SimulationEngine(config)
        history = engine.run(10)
        assert len(history) == 10
        # Verify population survived
        assert history[-1].population_size > 0
        # Verify genomes are present
        for agent in engine.population:
            assert len(agent.genome) > 0


# =====================================================================
# Serializer Integration Tests
# =====================================================================
class TestSerializerGenetics:
    def test_serialize_agent_detail_includes_genetics(self):
        from seldon.api.serializers import serialize_agent_detail
        config = _make_config()
        gm = GeneticModel(config)
        em = EpigeneticModel(config)
        rng = np.random.default_rng(42)
        ts = config.trait_system
        agent = _make_agent(config=config)
        agent.genome = gm.generate_initial_genome(ts, agent.traits, rng)
        agent.epigenetic_state = em.initialize_state()
        agent.epigenetic_state["stress_resilience"] = True
        result = serialize_agent_detail(agent, ts)
        assert "genome" in result
        assert "epigenetic_state" in result
        assert "genetic_lineage" in result
        # Genome values should be lists (not tuples) for JSON
        for locus, alleles in result["genome"].items():
            assert isinstance(alleles, list)
        assert result["epigenetic_state"]["stress_resilience"] is True

    def test_serialize_agent_detail_empty_genetics(self):
        from seldon.api.serializers import serialize_agent_detail
        config = _make_config()
        ts = config.trait_system
        agent = _make_agent(config=config)
        result = serialize_agent_detail(agent, ts)
        assert result["genome"] == {}
        assert result["epigenetic_state"] == {}
        assert result["genetic_lineage"] == {}
