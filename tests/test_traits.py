"""Tests for TraitSystem."""

import numpy as np
import pytest

from seldon.core.traits import TraitSystem


class TestTraitPresets:
    def test_compact_has_15_traits(self):
        ts = TraitSystem("compact")
        assert ts.count == 15

    def test_full_has_50_traits(self):
        ts = TraitSystem("full")
        assert ts.count == 50

    def test_unknown_preset_raises(self):
        with pytest.raises(ValueError, match="Unknown preset"):
            TraitSystem("nonexistent")

    def test_custom_preset_requires_traits(self):
        with pytest.raises(ValueError, match="custom_traits required"):
            TraitSystem("custom")

    def test_custom_preset(self):
        custom = [
            {"name": "trait_a", "desirability": 1, "stability": 0.5, "description": "A"},
            {"name": "trait_b", "desirability": -1, "stability": 0.3, "description": "B"},
        ]
        ts = TraitSystem("custom", custom_traits=custom)
        assert ts.count == 2
        assert ts.TRAIT_A == 0
        assert ts.TRAIT_B == 1


class TestTraitIndexing:
    def test_dynamic_index_constants(self):
        ts = TraitSystem("compact")
        assert hasattr(ts, "OPENNESS")
        assert hasattr(ts, "DEPTH_DRIVE")
        assert ts.OPENNESS == 0
        assert ts.CONSCIENTIOUSNESS == 1

    def test_trait_name_by_index(self):
        ts = TraitSystem("compact")
        assert ts.trait_name(0) == "openness"
        assert ts.trait_name(14) == "depth_drive"

    def test_trait_index_by_name(self):
        ts = TraitSystem("compact")
        assert ts.trait_index("openness") == 0
        assert ts.trait_index("depth_drive") == 14

    def test_invalid_index_raises(self):
        ts = TraitSystem("compact")
        with pytest.raises(IndexError):
            ts.trait_name(99)

    def test_invalid_name_raises(self):
        ts = TraitSystem("compact")
        with pytest.raises(KeyError):
            ts.trait_index("nonexistent")

    def test_depth_drive_in_both_presets(self):
        compact = TraitSystem("compact")
        full = TraitSystem("full")
        assert compact.trait_index("depth_drive") == compact.count - 1
        assert full.trait_index("depth_drive") == full.count - 1


class TestTraitVectors:
    def test_desirability_shape(self):
        ts = TraitSystem("compact")
        assert ts.desirability.shape == (15,)

    def test_stability_shape(self):
        ts = TraitSystem("compact")
        assert ts.stability.shape == (15,)

    def test_desirability_values(self):
        ts = TraitSystem("compact")
        # Neuroticism is negative
        assert ts.desirability[ts.NEUROTICISM] == -1
        # Openness is positive
        assert ts.desirability[ts.OPENNESS] == 1
        # Extraversion is neutral
        assert ts.desirability[ts.EXTRAVERSION] == 0

    def test_random_traits_shape(self):
        ts = TraitSystem("compact")
        traits = ts.random_traits()
        assert traits.shape == (15,)

    def test_random_traits_bounded(self):
        ts = TraitSystem("compact")
        rng = np.random.default_rng(42)
        for _ in range(100):
            traits = ts.random_traits(rng)
            assert np.all(traits >= 0.0) and np.all(traits <= 1.0)

    def test_random_population_shape(self):
        ts = TraitSystem("full")
        pop = ts.random_population_traits(20)
        assert pop.shape == (20, 50)

    def test_names_list(self):
        ts = TraitSystem("compact")
        names = ts.names()
        assert len(names) == 15
        assert names[0] == "openness"
        assert names[-1] == "depth_drive"
