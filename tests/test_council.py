"""Tests for CognitiveCouncil."""

import numpy as np
import pytest

from seldon.core.agent import Agent
from seldon.core.config import ExperimentConfig
from seldon.core.council import (
    CognitiveCouncil,
    DEFAULT_COUNCIL_WEIGHTS,
    SUB_AGENT_NAMES,
)


def _make_agent(traits: np.ndarray, **kwargs) -> Agent:
    defaults = dict(
        id="agent_001", name="Test", age=25, generation=0,
        birth_order=1, traits=traits, traits_at_birth=traits.copy(),
    )
    defaults.update(kwargs)
    return Agent(**defaults)


class TestCouncilDisabled:
    def test_dominant_voice_returns_none_when_disabled(self):
        config = ExperimentConfig(cognitive_council_enabled=False)
        council = CognitiveCouncil(config)
        agent = _make_agent(config.trait_system.random_traits(np.random.default_rng(42)))
        assert council.get_dominant_voice(agent) is None

    def test_modulation_returns_none_when_disabled(self):
        config = ExperimentConfig(cognitive_council_enabled=False)
        council = CognitiveCouncil(config)
        agent = _make_agent(config.trait_system.random_traits(np.random.default_rng(42)))
        assert council.compute_council_modulation(agent) is None


class TestCouncilEnabled:
    def test_dominant_voice_returns_string(self):
        config = ExperimentConfig(cognitive_council_enabled=True)
        council = CognitiveCouncil(config)
        agent = _make_agent(config.trait_system.random_traits(np.random.default_rng(42)))
        voice = council.get_dominant_voice(agent)
        assert isinstance(voice, str)
        assert voice in SUB_AGENT_NAMES

    def test_all_sub_agents_have_activations(self):
        config = ExperimentConfig(cognitive_council_enabled=True)
        council = CognitiveCouncil(config)
        agent = _make_agent(config.trait_system.random_traits(np.random.default_rng(42)))
        activations = council.get_activations(agent)
        assert len(activations) == 8

    def test_modulation_shape(self):
        config = ExperimentConfig(cognitive_council_enabled=True)
        council = CognitiveCouncil(config)
        agent = _make_agent(config.trait_system.random_traits(np.random.default_rng(42)))
        mod = council.compute_council_modulation(agent)
        assert mod is not None
        assert mod.shape == (config.trait_system.count,)

    def test_modulation_bounded(self):
        config = ExperimentConfig(cognitive_council_enabled=True)
        council = CognitiveCouncil(config)
        rng = np.random.default_rng(42)
        for _ in range(20):
            agent = _make_agent(config.trait_system.random_traits(rng))
            mod = council.compute_council_modulation(agent)
            assert mod is not None
            assert np.all(mod >= 0.5)
            assert np.all(mod <= 1.5)


class TestDominantVoiceBehavior:
    def test_high_neuroticism_activates_amygdala(self):
        config = ExperimentConfig(cognitive_council_enabled=True)
        ts = config.trait_system
        council = CognitiveCouncil(config)

        traits = np.full(ts.count, 0.3)
        traits[ts.NEUROTICISM] = 0.99
        traits[ts.RESILIENCE] = 0.01
        traits[ts.TRUST] = 0.01
        agent = _make_agent(traits)

        activations = council.get_activations(agent)
        # Amygdala should be very active with high neuroticism + low resilience/trust
        assert activations["amygdala"] > 0.3

    def test_high_creativity_depth_activates_seer(self):
        config = ExperimentConfig(cognitive_council_enabled=True)
        ts = config.trait_system
        council = CognitiveCouncil(config)

        traits = np.full(ts.count, 0.3)
        traits[ts.CREATIVITY] = 0.95
        traits[ts.DEPTH_DRIVE] = 0.95
        traits[ts.OPENNESS] = 0.9
        agent = _make_agent(traits)

        activations = council.get_activations(agent)
        assert activations["seer"] > activations["house"]

    def test_high_empathy_agreeableness_activates_conscience(self):
        config = ExperimentConfig(cognitive_council_enabled=True)
        ts = config.trait_system
        council = CognitiveCouncil(config)

        traits = np.full(ts.count, 0.3)
        traits[ts.EMPATHY] = 0.95
        traits[ts.AGREEABLENESS] = 0.95
        traits[ts.TRUST] = 0.9
        traits[ts.SELF_CONTROL] = 0.8
        agent = _make_agent(traits)

        voice = council.get_dominant_voice(agent)
        assert voice == "conscience"

    def test_high_extraversion_dominance_activates_hypothalamus(self):
        config = ExperimentConfig(cognitive_council_enabled=True)
        ts = config.trait_system
        council = CognitiveCouncil(config)

        traits = np.full(ts.count, 0.2)
        traits[ts.EXTRAVERSION] = 0.95
        traits[ts.DOMINANCE] = 0.95
        traits[ts.AMBITION] = 0.9
        traits[ts.RISK_TAKING] = 0.85
        agent = _make_agent(traits)

        voice = council.get_dominant_voice(agent)
        assert voice == "hypothalamus"


class TestCustomWeights:
    def test_custom_council_weights(self):
        custom_weights = {
            "custom_voice": [("openness", 1.0)],
        }
        config = ExperimentConfig(
            cognitive_council_enabled=True,
            cognitive_council_weights=custom_weights,
        )
        council = CognitiveCouncil(config)
        ts = config.trait_system

        traits = ts.random_traits(np.random.default_rng(42))
        agent = _make_agent(traits)

        voice = council.get_dominant_voice(agent)
        assert voice == "custom_voice"

    def test_unknown_trait_in_weights_ignored(self):
        custom_weights = {
            "test_voice": [("openness", 0.5), ("nonexistent_trait", 0.5)],
        }
        config = ExperimentConfig(
            cognitive_council_enabled=True,
            cognitive_council_weights=custom_weights,
        )
        council = CognitiveCouncil(config)
        # Should not raise; nonexistent trait silently skipped
        assert "test_voice" in council._valid_agents


class TestFullPresetCouncil:
    def test_council_works_with_full_preset(self):
        config = ExperimentConfig(
            cognitive_council_enabled=True,
            trait_preset="full",
        )
        council = CognitiveCouncil(config)
        agent = _make_agent(config.trait_system.random_traits(np.random.default_rng(42)))
        voice = council.get_dominant_voice(agent)
        assert voice is not None
        assert voice in SUB_AGENT_NAMES


class TestUniformTraits:
    def test_modulation_centered_near_one_for_uniform_traits(self):
        config = ExperimentConfig(cognitive_council_enabled=True)
        council = CognitiveCouncil(config)
        traits = np.full(config.trait_system.count, 0.5)
        agent = _make_agent(traits)
        mod = council.compute_council_modulation(agent)
        assert mod is not None
        # With uniform traits, modulation should be centered near 1.0
        assert np.mean(mod) == pytest.approx(1.0, abs=0.1)
        # And bounded
        assert np.all(mod >= 0.5)
        assert np.all(mod <= 1.5)
