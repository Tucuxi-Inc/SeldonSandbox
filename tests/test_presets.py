"""Tests for experiment presets."""

import pytest

from seldon.core.config import ExperimentConfig
from seldon.core.engine import SimulationEngine
from seldon.experiment.presets import (
    PRESETS,
    baseline,
    no_birth_order,
    inverted_birth_order,
    high_sacrificial,
    no_recovery,
    high_trait_drift,
    opposites_attract,
    archetype_society,
    high_lore_decay,
    stable_lore,
    get_preset,
    list_presets,
)


class TestPresetCount:
    def test_ten_presets_defined(self):
        assert len(PRESETS) == 10

    def test_list_presets(self):
        names = list_presets()
        assert len(names) == 10
        assert "baseline" in names
        assert "high_sacrificial" in names


class TestPresetReturnTypes:
    def test_all_presets_return_config(self):
        for name, factory in PRESETS.items():
            config = factory()
            assert isinstance(config, ExperimentConfig), f"{name} failed"
            assert config.experiment_name == name


class TestGetPreset:
    def test_get_known_preset(self):
        config = get_preset("baseline")
        assert config.experiment_name == "baseline"

    def test_get_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown preset"):
            get_preset("nonexistent")


class TestPresetValues:
    def test_baseline_has_defaults(self):
        config = baseline()
        assert config.initial_population == 100
        assert config.generations_to_run == 50

    def test_no_birth_order_uses_average(self):
        config = no_birth_order()
        assert config.birth_order_rules == {1: "average", 2: "average", 3: "average"}

    def test_inverted_birth_order(self):
        config = inverted_birth_order()
        assert config.birth_order_rules[1] == "best"
        assert config.birth_order_rules[3] == "worst"

    def test_high_sacrificial_lower_threshold(self):
        config = high_sacrificial()
        default = ExperimentConfig()
        assert config.region_thresholds["deep_to_extreme"] < default.region_thresholds["deep_to_extreme"]

    def test_high_trait_drift(self):
        config = high_trait_drift()
        assert config.trait_drift_rate == 0.1

    def test_opposites_attract_high_complementarity(self):
        config = opposites_attract()
        assert config.attraction_weights["complementarity"] == 0.5

    def test_high_lore_decay(self):
        config = high_lore_decay()
        assert config.lore_decay_rate == 0.2
        assert config.lore_enabled is True

    def test_stable_lore(self):
        config = stable_lore()
        assert config.lore_decay_rate == 0.01
        assert config.lore_transmission_rate == 0.9


class TestPresetsRunnable:
    """Each preset should produce a valid simulation that runs."""

    @pytest.mark.parametrize("preset_name", list(PRESETS.keys()))
    def test_preset_runs(self, preset_name):
        config = get_preset(preset_name)
        config = ExperimentConfig(
            **{
                **config.to_dict(),
                "initial_population": 20,
                "generations_to_run": 3,
                "random_seed": 42,
            }
        )
        engine = SimulationEngine(config)
        history = engine.run()
        assert len(history) == 3
        assert history[0].population_size > 0
