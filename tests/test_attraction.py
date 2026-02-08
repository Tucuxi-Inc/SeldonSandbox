"""Tests for AttractionModel."""

import numpy as np
import pytest

from seldon.core.agent import Agent
from seldon.core.attraction import AttractionModel
from seldon.core.config import ExperimentConfig


def _make_agent(traits, age=25, **kwargs):
    defaults = dict(
        id="test", name="Test", age=age, generation=0, birth_order=1,
        traits=traits, traits_at_birth=traits.copy(),
    )
    defaults.update(kwargs)
    return Agent(**defaults)


class TestAttractionBasics:
    def test_attraction_is_bounded(self):
        config = ExperimentConfig()
        am = AttractionModel(config)
        rng = np.random.default_rng(42)
        ts = config.trait_system

        for _ in range(100):
            a1 = _make_agent(ts.random_traits(rng), id="a1")
            a2 = _make_agent(ts.random_traits(rng), id="a2")
            score = am.calculate(a1, a2, rng)
            assert 0.0 <= score <= 1.0

    def test_identical_agents_have_high_similarity(self):
        config = ExperimentConfig()
        am = AttractionModel(config)
        ts = config.trait_system

        traits = ts.random_traits(np.random.default_rng(42))
        a1 = _make_agent(traits.copy(), id="a1")
        a2 = _make_agent(traits.copy(), id="a2")

        # Similarity component should be ~1.0 for identical traits
        sim = am._similarity(a1, a2)
        assert sim == pytest.approx(1.0, abs=0.01)


class TestAttractionComponents:
    def test_age_compatibility_same_age(self):
        config = ExperimentConfig()
        am = AttractionModel(config)
        ts = config.trait_system

        a1 = _make_agent(ts.random_traits(), age=25, id="a1")
        a2 = _make_agent(ts.random_traits(), age=25, id="a2")
        assert am._age_compatibility(a1, a2) == pytest.approx(1.0, abs=0.01)

    def test_age_compatibility_decreases_with_gap(self):
        config = ExperimentConfig()
        am = AttractionModel(config)
        ts = config.trait_system

        a1 = _make_agent(ts.random_traits(), age=25, id="a1")
        a2_close = _make_agent(ts.random_traits(), age=27, id="a2")
        a2_far = _make_agent(ts.random_traits(), age=45, id="a3")

        close_compat = am._age_compatibility(a1, a2_close)
        far_compat = am._age_compatibility(a1, a2_far)
        assert close_compat > far_compat

    def test_social_proximity_with_bond(self):
        config = ExperimentConfig()
        am = AttractionModel(config)
        ts = config.trait_system

        a1 = _make_agent(ts.random_traits(), id="a1")
        a2 = _make_agent(ts.random_traits(), id="a2")
        a1.social_bonds["a2"] = 0.8

        prox = am._social_proximity(a1, a2)
        assert prox > 0.5
