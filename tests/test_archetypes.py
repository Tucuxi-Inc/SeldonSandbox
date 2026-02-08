"""Tests for Archetypes."""

import numpy as np
import pytest

from seldon.core.config import ExperimentConfig
from seldon.experiment.archetypes import (
    ARCHETYPES,
    ArchetypeDefinition,
    create_agent_from_archetype,
    get_archetype,
    list_archetypes,
)


class TestArchetypeDefinitions:
    def test_eleven_archetypes_defined(self):
        assert len(ARCHETYPES) == 11

    def test_all_archetypes_have_required_fields(self):
        for name, arch in ARCHETYPES.items():
            assert isinstance(arch, ArchetypeDefinition)
            assert arch.name, f"{name} missing name"
            assert arch.description, f"{name} missing description"
            assert arch.key_traits, f"{name} missing key_traits"
            assert arch.use_case, f"{name} missing use_case"
            assert arch.trait_values, f"{name} missing trait_values"

    def test_trait_values_in_range(self):
        for name, arch in ARCHETYPES.items():
            for trait, value in arch.trait_values.items():
                assert 0.0 <= value <= 1.0, f"{name}.{trait} = {value} out of range"

    def test_all_archetypes_have_compact_traits(self):
        """All archetypes should define values for the compact trait set."""
        compact_names = {
            "openness", "conscientiousness", "extraversion", "agreeableness",
            "neuroticism", "creativity", "resilience", "ambition", "empathy",
            "dominance", "trust", "risk_taking", "adaptability", "self_control",
            "depth_drive",
        }
        for name, arch in ARCHETYPES.items():
            defined_traits = set(arch.trait_values.keys())
            assert defined_traits == compact_names, (
                f"{name} is missing traits: {compact_names - defined_traits}"
            )


class TestGetArchetype:
    def test_get_by_name(self):
        arch = get_archetype("einstein")
        assert arch.name == "Einstein"

    def test_get_case_insensitive(self):
        arch = get_archetype("Einstein")
        assert arch.name == "Einstein"

    def test_get_with_spaces(self):
        arch = get_archetype("fred rogers")
        assert arch.name == "Fred Rogers"

    def test_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown archetype"):
            get_archetype("unknown_person")


class TestListArchetypes:
    def test_returns_all_names(self):
        names = list_archetypes()
        assert len(names) == 11
        assert "einstein" in names
        assert "curie" in names
        assert "da_vinci" in names


class TestCreateAgentFromArchetype:
    def test_creates_agent_with_correct_traits(self):
        config = ExperimentConfig()
        ts = config.trait_system
        agent = create_agent_from_archetype("einstein", config, "agent_001")

        assert agent.id == "agent_001"
        assert agent.name == "Einstein"
        assert agent.traits[ts.DEPTH_DRIVE] == pytest.approx(0.95)
        assert agent.traits[ts.OPENNESS] == pytest.approx(0.9)

    def test_custom_name_and_age(self):
        config = ExperimentConfig()
        agent = create_agent_from_archetype(
            "curie", config, "agent_002",
            agent_name="Marie", age=30,
        )
        assert agent.name == "Marie"
        assert agent.age == 30

    def test_default_age_from_config(self):
        config = ExperimentConfig(outsider_injection_age=25)
        agent = create_agent_from_archetype("da_vinci", config, "agent_003")
        assert agent.age == 25

    def test_noise_adds_variation(self):
        config = ExperimentConfig()
        rng = np.random.default_rng(42)
        agent1 = create_agent_from_archetype(
            "einstein", config, "a1", noise_sigma=0.1, rng=rng,
        )
        agent2 = create_agent_from_archetype(
            "einstein", config, "a2", noise_sigma=0.1, rng=rng,
        )
        # With noise, two agents from same archetype should differ
        assert not np.array_equal(agent1.traits, agent2.traits)

    def test_no_noise_gives_exact_values(self):
        config = ExperimentConfig()
        ts = config.trait_system
        agent = create_agent_from_archetype("curie", config, "a1", noise_sigma=0.0)
        assert agent.traits[ts.DEPTH_DRIVE] == pytest.approx(0.95)

    def test_traits_bounded_with_noise(self):
        config = ExperimentConfig()
        rng = np.random.default_rng(42)
        for _ in range(20):
            agent = create_agent_from_archetype(
                "da_vinci", config, "a1", noise_sigma=0.5, rng=rng,
            )
            assert np.all(agent.traits >= 0.0)
            assert np.all(agent.traits <= 1.0)

    def test_works_with_full_preset(self):
        config = ExperimentConfig(trait_preset="full")
        ts = config.trait_system
        agent = create_agent_from_archetype("einstein", config, "a1")
        assert agent.traits.shape == (50,)
        # depth_drive should be set correctly
        assert agent.traits[ts.DEPTH_DRIVE] == pytest.approx(0.95)
        # Unspecified full-preset traits should be 0.5
        assert agent.traits[ts.trait_index("curiosity")] == pytest.approx(0.5)

    def test_traits_at_birth_matches(self):
        config = ExperimentConfig()
        agent = create_agent_from_archetype("fred_rogers", config, "a1")
        np.testing.assert_array_equal(agent.traits, agent.traits_at_birth)
