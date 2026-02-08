"""Tests for InheritanceEngine."""

import numpy as np
import pytest

from seldon.core.agent import Agent
from seldon.core.config import ExperimentConfig
from seldon.core.inheritance import InheritanceEngine


def _make_agent(traits, **kwargs):
    defaults = dict(
        id="test", name="Test", age=25, generation=0, birth_order=1,
        traits=traits, traits_at_birth=traits.copy(),
    )
    defaults.update(kwargs)
    return Agent(**defaults)


class TestWorstInheritance:
    def test_positive_trait_gets_minimum(self):
        config = ExperimentConfig(inheritance_noise_sigma=0.0)
        ts = config.trait_system
        ie = InheritanceEngine(config)

        p1 = _make_agent(np.full(ts.count, 0.8), id="p1")
        p2 = _make_agent(np.full(ts.count, 0.3), id="p2")

        child = ie.inherit(p1, p2, birth_order=1, population=[])
        # Conscientiousness (desirability +1) should get min
        assert child[ts.CONSCIENTIOUSNESS] == pytest.approx(0.3, abs=0.01)

    def test_negative_trait_gets_maximum(self):
        config = ExperimentConfig(inheritance_noise_sigma=0.0)
        ts = config.trait_system
        ie = InheritanceEngine(config)

        p1 = _make_agent(np.full(ts.count, 0.3), id="p1")
        p2 = _make_agent(np.full(ts.count, 0.8), id="p2")

        child = ie.inherit(p1, p2, birth_order=1, population=[])
        # Neuroticism (desirability -1) should get max (worse)
        assert child[ts.NEUROTICISM] == pytest.approx(0.8, abs=0.01)


class TestBestInheritance:
    def test_positive_trait_gets_maximum(self):
        config = ExperimentConfig(inheritance_noise_sigma=0.0)
        ts = config.trait_system
        ie = InheritanceEngine(config)

        p1 = _make_agent(np.full(ts.count, 0.8), id="p1")
        p2 = _make_agent(np.full(ts.count, 0.3), id="p2")

        child = ie.inherit(p1, p2, birth_order=3, population=[])
        assert child[ts.CONSCIENTIOUSNESS] == pytest.approx(0.8, abs=0.01)

    def test_negative_trait_gets_minimum(self):
        config = ExperimentConfig(inheritance_noise_sigma=0.0)
        ts = config.trait_system
        ie = InheritanceEngine(config)

        p1 = _make_agent(np.full(ts.count, 0.3), id="p1")
        p2 = _make_agent(np.full(ts.count, 0.8), id="p2")

        child = ie.inherit(p1, p2, birth_order=3, population=[])
        assert child[ts.NEUROTICISM] == pytest.approx(0.3, abs=0.01)


class TestWeirdestInheritance:
    def test_farther_from_mean_selected(self):
        config = ExperimentConfig(inheritance_noise_sigma=0.0)
        ts = config.trait_system
        ie = InheritanceEngine(config)

        # Population mean ~ 0.5
        pop_agent = _make_agent(np.full(ts.count, 0.5), id="pop1")
        p1 = _make_agent(np.full(ts.count, 0.9), id="p1")  # Far from mean
        p2 = _make_agent(np.full(ts.count, 0.6), id="p2")  # Close to mean

        child = ie.inherit(p1, p2, birth_order=2, population=[pop_agent])
        # Should get p1's value (farther from 0.5)
        assert child[ts.OPENNESS] == pytest.approx(0.9, abs=0.01)

    def test_empty_population_averages(self):
        config = ExperimentConfig(inheritance_noise_sigma=0.0)
        ts = config.trait_system
        ie = InheritanceEngine(config)

        p1 = _make_agent(np.full(ts.count, 0.8), id="p1")
        p2 = _make_agent(np.full(ts.count, 0.4), id="p2")

        child = ie.inherit(p1, p2, birth_order=2, population=[])
        assert child[ts.OPENNESS] == pytest.approx(0.6, abs=0.01)


class TestConfigurableRules:
    def test_inverted_birth_order(self):
        config = ExperimentConfig(
            birth_order_rules={1: "best", 3: "worst"},
            inheritance_noise_sigma=0.0,
        )
        ts = config.trait_system
        ie = InheritanceEngine(config)

        p1 = _make_agent(np.full(ts.count, 0.8), id="p1")
        p2 = _make_agent(np.full(ts.count, 0.3), id="p2")

        child = ie.inherit(p1, p2, birth_order=1, population=[])
        # With inverted rules, first-born gets BEST
        assert child[ts.CONSCIENTIOUSNESS] == pytest.approx(0.8, abs=0.01)

    def test_fourth_child_uses_random_weighted(self):
        config = ExperimentConfig(inheritance_noise_sigma=0.0)
        ts = config.trait_system
        ie = InheritanceEngine(config)

        p1 = _make_agent(np.full(ts.count, 0.0), id="p1")
        p2 = _make_agent(np.full(ts.count, 1.0), id="p2")

        rng = np.random.default_rng(42)
        child = ie.inherit(p1, p2, birth_order=4, population=[], rng=rng)
        # Should be a mix — not all 0 or all 1
        assert 0.0 < child.mean() < 1.0


class TestInheritanceProperties:
    def test_child_traits_always_bounded(self):
        config = ExperimentConfig()
        ts = config.trait_system
        ie = InheritanceEngine(config)
        rng = np.random.default_rng(42)

        for _ in range(50):
            p1 = _make_agent(ts.random_traits(rng), id="p1")
            p2 = _make_agent(ts.random_traits(rng), id="p2")
            for bo in range(1, 6):
                child = ie.inherit(p1, p2, bo, [p1, p2], rng)
                assert np.all(child >= 0.0)
                assert np.all(child <= 1.0)

    def test_child_traits_correct_shape(self):
        config = ExperimentConfig(trait_preset="full")
        ts = config.trait_system
        ie = InheritanceEngine(config)

        p1 = _make_agent(ts.random_traits(), id="p1")
        p2 = _make_agent(ts.random_traits(), id="p2")
        child = ie.inherit(p1, p2, 1, [])
        assert child.shape == (50,)
