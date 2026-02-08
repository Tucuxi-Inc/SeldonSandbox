"""Tests for ProcessingClassifier."""

import numpy as np
import pytest

from seldon.core.agent import Agent
from seldon.core.config import ExperimentConfig
from seldon.core.processing import ProcessingClassifier, ProcessingRegion


def _make_agent(depth_drive, creativity=0.5, resilience=0.5, burnout=0.0):
    config = ExperimentConfig()
    ts = config.trait_system
    traits = np.full(ts.count, 0.5)
    traits[ts.DEPTH_DRIVE] = depth_drive
    traits[ts.CREATIVITY] = creativity
    traits[ts.RESILIENCE] = resilience
    return Agent(
        id="test", name="Test", age=25, generation=0, birth_order=1,
        traits=traits, traits_at_birth=traits.copy(), burnout_level=burnout,
    )


class TestRegionClassification:
    def test_low_depth_is_under_processing(self):
        config = ExperimentConfig()
        classifier = ProcessingClassifier(config)
        agent = _make_agent(depth_drive=0.1)
        assert classifier.classify(agent) == ProcessingRegion.UNDER_PROCESSING

    def test_moderate_depth_is_optimal(self):
        config = ExperimentConfig()
        classifier = ProcessingClassifier(config)
        agent = _make_agent(depth_drive=0.4)
        assert classifier.classify(agent) == ProcessingRegion.OPTIMAL

    def test_high_depth_is_deep(self):
        config = ExperimentConfig()
        classifier = ProcessingClassifier(config)
        agent = _make_agent(depth_drive=0.6)
        assert classifier.classify(agent) == ProcessingRegion.DEEP

    def test_extreme_depth_high_potential_is_sacrificial(self):
        config = ExperimentConfig()
        classifier = ProcessingClassifier(config)
        agent = _make_agent(depth_drive=0.9, creativity=0.9, resilience=0.9, burnout=0.0)
        assert classifier.classify(agent) == ProcessingRegion.SACRIFICIAL

    def test_extreme_depth_low_potential_is_pathological(self):
        config = ExperimentConfig()
        classifier = ProcessingClassifier(config)
        agent = _make_agent(depth_drive=0.9, creativity=0.1, resilience=0.1, burnout=0.8)
        assert classifier.classify(agent) == ProcessingRegion.PATHOLOGICAL


class TestConfigurableThresholds:
    def test_custom_thresholds(self):
        config = ExperimentConfig(
            region_thresholds={
                "under_to_optimal": 0.1,
                "optimal_to_deep": 0.2,
                "deep_to_extreme": 0.3,
                "productive_potential_threshold": 0.5,
            }
        )
        classifier = ProcessingClassifier(config)
        # depth_drive=0.25 is now "deep" (between 0.2 and 0.3)
        agent = _make_agent(depth_drive=0.25)
        assert classifier.classify(agent) == ProcessingRegion.DEEP


class TestR4R5Distinction:
    def test_burnout_pushes_to_pathological(self):
        config = ExperimentConfig()
        classifier = ProcessingClassifier(config)

        # Same agent with low vs high burnout
        productive = _make_agent(depth_drive=0.9, creativity=0.8, resilience=0.8, burnout=0.0)
        burned_out = _make_agent(depth_drive=0.9, creativity=0.8, resilience=0.8, burnout=0.9)

        r_prod = classifier.classify(productive)
        r_burn = classifier.classify(burned_out)

        # Productive should be sacrificial, burned out should be pathological
        assert r_prod == ProcessingRegion.SACRIFICIAL
        assert r_burn == ProcessingRegion.PATHOLOGICAL
