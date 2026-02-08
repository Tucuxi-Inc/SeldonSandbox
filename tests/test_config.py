"""Tests for ExperimentConfig."""

import pytest

from seldon.core.config import ExperimentConfig
from seldon.core.traits import TraitSystem


class TestConfigDefaults:
    def test_default_experiment_name(self):
        c = ExperimentConfig()
        assert c.experiment_name == "default"

    def test_default_trait_preset(self):
        c = ExperimentConfig()
        assert c.trait_preset == "compact"

    def test_default_birth_order_rules(self):
        c = ExperimentConfig()
        assert c.birth_order_rules == {1: "worst", 2: "weirdest", 3: "best"}

    def test_default_region_thresholds(self):
        c = ExperimentConfig()
        assert "under_to_optimal" in c.region_thresholds
        assert "productive_potential_threshold" in c.region_thresholds


class TestTraitSystemIntegration:
    def test_lazy_trait_system(self):
        c = ExperimentConfig()
        ts = c.trait_system
        assert isinstance(ts, TraitSystem)
        assert ts.count == 15

    def test_full_trait_preset(self):
        c = ExperimentConfig(trait_preset="full")
        assert c.trait_system.count == 50

    def test_trait_system_cached(self):
        c = ExperimentConfig()
        ts1 = c.trait_system
        ts2 = c.trait_system
        assert ts1 is ts2


class TestSerialization:
    def test_to_dict_roundtrip(self):
        c = ExperimentConfig(experiment_name="test", initial_population=50)
        d = c.to_dict()
        c2 = ExperimentConfig.from_dict(d)
        assert c2.experiment_name == "test"
        assert c2.initial_population == 50

    def test_to_json_roundtrip(self):
        c = ExperimentConfig(experiment_name="json_test")
        j = c.to_json()
        c2 = ExperimentConfig.from_json(j)
        assert c2.experiment_name == "json_test"

    def test_diff(self):
        c1 = ExperimentConfig(experiment_name="a", trait_drift_rate=0.02)
        c2 = ExperimentConfig(experiment_name="b", trait_drift_rate=0.1)
        diffs = c1.diff(c2)
        assert "experiment_name" in diffs
        assert "trait_drift_rate" in diffs
        assert "initial_population" not in diffs

    def test_to_dict_excludes_private(self):
        c = ExperimentConfig()
        _ = c.trait_system  # Force cache
        d = c.to_dict()
        assert "_trait_system" not in d
