"""Tests for FertilityManager."""

import numpy as np
import pytest

from seldon.core.agent import Agent
from seldon.core.config import ExperimentConfig
from seldon.social.fertility import FertilityManager


def _make_agent(agent_id: str, age: int = 25, **kwargs) -> Agent:
    traits = np.full(15, 0.5)
    defaults = dict(
        id=agent_id, name=f"Agent-{agent_id}", age=age, generation=0,
        birth_order=1, traits=traits, traits_at_birth=traits.copy(),
    )
    defaults.update(kwargs)
    return Agent(**defaults)


class TestCanReproduce:
    def test_basic_pair_can_reproduce(self):
        config = ExperimentConfig()
        fm = FertilityManager(config)
        p1 = _make_agent("p1", age=25)
        p2 = _make_agent("p2", age=25)
        p1.partner_id = p2.id
        p2.partner_id = p1.id
        assert fm.can_reproduce(p1, p2, 5)

    def test_dead_parent_cannot_reproduce(self):
        config = ExperimentConfig()
        fm = FertilityManager(config)
        p1 = _make_agent("p1", age=25)
        p2 = _make_agent("p2", age=25, is_alive=False)
        assert not fm.can_reproduce(p1, p2, 5)

    def test_too_young_cannot_reproduce(self):
        config = ExperimentConfig()
        fm = FertilityManager(config)
        p1 = _make_agent("p1", age=10)
        p2 = _make_agent("p2", age=10)
        assert not fm.can_reproduce(p1, p2, 5)

    def test_too_old_cannot_reproduce(self):
        config = ExperimentConfig()
        fm = FertilityManager(config)
        p1 = _make_agent("p1", age=50)
        p2 = _make_agent("p2", age=50)
        assert not fm.can_reproduce(p1, p2, 5)

    def test_birth_spacing_enforced(self):
        config = ExperimentConfig(
            fertility_config={
                **ExperimentConfig().fertility_config,
                "min_birth_spacing_generations": 2,
            }
        )
        fm = FertilityManager(config)
        p1 = _make_agent("p1", age=25)
        p2 = _make_agent("p2", age=25)
        p1.last_birth_generation = 4  # Born child in gen 4

        # Gen 5 — only 1 gen gap, should fail
        assert not fm.can_reproduce(p1, p2, 5)
        # Gen 6 — 2 gen gap, should pass
        assert fm.can_reproduce(p1, p2, 6)

    def test_one_parent_in_window_sufficient(self):
        config = ExperimentConfig()
        fm = FertilityManager(config)
        p1 = _make_agent("p1", age=25)  # In window
        p2 = _make_agent("p2", age=50)  # Out of window
        assert fm.can_reproduce(p1, p2, 5)


class TestWillReproduce:
    def test_will_reproduce_probabilistic(self):
        config = ExperimentConfig()
        fm = FertilityManager(config)
        rng = np.random.default_rng(42)

        p1 = _make_agent("p1", age=25)
        p2 = _make_agent("p2", age=25)

        results = [fm.will_reproduce(p1, p2, 5, rng) for _ in range(100)]
        # Should get some True and some False
        assert any(results)
        assert not all(results)

    def test_societal_pressure_increases_reproduction(self):
        config_low = ExperimentConfig(
            fertility_config={
                **ExperimentConfig().fertility_config,
                "societal_fertility_pressure": 0.0,
            }
        )
        config_high = ExperimentConfig(
            fertility_config={
                **ExperimentConfig().fertility_config,
                "societal_fertility_pressure": 1.0,
            }
        )
        fm_low = FertilityManager(config_low)
        fm_high = FertilityManager(config_high)

        p1 = _make_agent("p1", age=25)
        p2 = _make_agent("p2", age=25)
        rng = np.random.default_rng(42)

        low_count = sum(fm_low.will_reproduce(p1, p2, 5, rng) for _ in range(300))
        high_count = sum(fm_high.will_reproduce(p1, p2, 5, rng) for _ in range(300))
        assert high_count > low_count

    def test_above_target_reduces_probability(self):
        config = ExperimentConfig(
            fertility_config={
                **ExperimentConfig().fertility_config,
                "target_children_mean": 2.0,
            }
        )
        fm = FertilityManager(config)
        rng = np.random.default_rng(42)

        p1 = _make_agent("p1", age=25)
        p2 = _make_agent("p2", age=25)

        # No children yet
        no_kids = sum(fm.will_reproduce(p1, p2, 5, rng) for _ in range(200))

        # Already have 3 children (above target of 2)
        p1.children_ids = ["c1", "c2", "c3"]
        p2.children_ids = ["c1", "c2", "c3"]
        many_kids = sum(fm.will_reproduce(p1, p2, 5, rng) for _ in range(200))

        assert no_kids > many_kids


class TestChildMortality:
    def test_child_mortality_occurs(self):
        config = ExperimentConfig(
            fertility_config={
                **ExperimentConfig().fertility_config,
                "child_mortality_rate": 0.5,
            }
        )
        fm = FertilityManager(config)
        rng = np.random.default_rng(42)

        deaths = sum(fm.check_child_mortality(rng) for _ in range(100))
        assert 30 < deaths < 70  # Should be close to 50%

    def test_zero_child_mortality(self):
        config = ExperimentConfig(
            fertility_config={
                **ExperimentConfig().fertility_config,
                "child_mortality_rate": 0.0,
            }
        )
        fm = FertilityManager(config)
        rng = np.random.default_rng(42)

        deaths = sum(fm.check_child_mortality(rng) for _ in range(100))
        assert deaths == 0


class TestMaternalMortality:
    def test_maternal_mortality_occurs(self):
        config = ExperimentConfig(
            fertility_config={
                **ExperimentConfig().fertility_config,
                "maternal_mortality_rate": 0.5,
            }
        )
        fm = FertilityManager(config)
        rng = np.random.default_rng(42)

        deaths = sum(fm.check_maternal_mortality(rng) for _ in range(100))
        assert 30 < deaths < 70

    def test_zero_maternal_mortality(self):
        config = ExperimentConfig(
            fertility_config={
                **ExperimentConfig().fertility_config,
                "maternal_mortality_rate": 0.0,
            }
        )
        fm = FertilityManager(config)
        rng = np.random.default_rng(42)

        deaths = sum(fm.check_maternal_mortality(rng) for _ in range(100))
        assert deaths == 0
