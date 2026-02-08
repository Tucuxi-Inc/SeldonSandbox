"""Tests for DecisionModel."""

import numpy as np
import pytest

from seldon.core.agent import Agent
from seldon.core.config import ExperimentConfig
from seldon.core.decision import DecisionContext, DecisionModel, DecisionResult


def _make_agent(traits):
    return Agent(
        id="test", name="Test", age=25, generation=0, birth_order=1,
        traits=traits, traits_at_birth=traits.copy(),
    )


class TestDecisionBasics:
    def test_single_action_returns_certainty(self):
        config = ExperimentConfig()
        dm = DecisionModel(config)
        agent = _make_agent(config.trait_system.random_traits())

        result = dm.decide(
            agent, DecisionContext.MIGRATION,
            situation={"x": 1.0}, actions=["stay"],
        )
        assert result.chosen_action == "stay"
        assert result.probabilities["stay"] == 1.0

    def test_probabilities_sum_to_one(self):
        config = ExperimentConfig()
        dm = DecisionModel(config)
        rng = np.random.default_rng(42)
        agent = _make_agent(config.trait_system.random_traits(rng))

        result = dm.decide(
            agent, DecisionContext.PAIRING,
            situation={"attractiveness": 0.7},
            actions=["pair", "skip"],
            rng=rng,
        )
        assert abs(sum(result.probabilities.values()) - 1.0) < 1e-6

    def test_empty_actions_raises(self):
        config = ExperimentConfig()
        dm = DecisionModel(config)
        agent = _make_agent(config.trait_system.random_traits())

        with pytest.raises(ValueError, match="(?i)at least one action"):
            dm.decide(agent, DecisionContext.MIGRATION, {}, [])

    def test_result_has_trait_contributions(self):
        config = ExperimentConfig()
        dm = DecisionModel(config)
        agent = _make_agent(config.trait_system.random_traits())

        result = dm.decide(
            agent, DecisionContext.CONTRIBUTION,
            situation={"effort": 0.5},
            actions=["low", "high"],
        )
        assert result.trait_contributions.shape == (config.trait_system.count,)


class TestTemperature:
    def test_zero_temperature_is_deterministic(self):
        config = ExperimentConfig(decision_temperature=0.0)
        dm = DecisionModel(config)
        ts = config.trait_system

        # Create agent with traits that favor action A via custom weights
        traits = np.zeros(ts.count)
        traits[ts.OPENNESS] = 1.0
        agent = _make_agent(traits)

        weights = {
            "a": np.zeros(ts.count),
            "b": np.zeros(ts.count),
        }
        weights["a"][ts.OPENNESS] = 10.0  # Strongly favors A

        result = dm.decide(
            agent, DecisionContext.SOCIAL,
            situation={"x": 1.0},
            actions=["a", "b"],
            action_weights=weights,
        )
        assert result.probabilities["a"] == 1.0

    def test_high_temperature_more_uniform(self):
        config = ExperimentConfig(decision_temperature=100.0)
        dm = DecisionModel(config)
        agent = _make_agent(config.trait_system.random_traits())

        result = dm.decide(
            agent, DecisionContext.MIGRATION,
            situation={"x": 1.0},
            actions=["stay", "go"],
        )
        # With very high temperature, probabilities should be close to uniform
        assert abs(result.probabilities["stay"] - 0.5) < 0.1


class TestExplainability:
    def test_explain_returns_trait_names(self):
        config = ExperimentConfig()
        dm = DecisionModel(config)
        agent = _make_agent(config.trait_system.random_traits(np.random.default_rng(42)))

        result = dm.decide(
            agent, DecisionContext.PAIRING,
            situation={"x": 1.0},
            actions=["pair", "skip"],
        )
        explanation = result.explain(config.trait_system.names())
        assert isinstance(explanation, dict)
        # At least some traits should have non-trivial influence
        assert len(explanation) > 0

    def test_to_dict(self):
        config = ExperimentConfig()
        dm = DecisionModel(config)
        agent = _make_agent(config.trait_system.random_traits())

        result = dm.decide(
            agent, DecisionContext.MIGRATION,
            situation={"x": 1.0},
            actions=["stay", "go"],
        )
        d = result.to_dict()
        assert "chosen_action" in d
        assert "probabilities" in d
        assert "context" in d
