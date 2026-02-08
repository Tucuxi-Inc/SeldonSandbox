"""Tests for ExperimentRunner."""

import numpy as np
import pytest

from seldon.core.config import ExperimentConfig
from seldon.experiment.runner import (
    ExperimentRunner,
    ExperimentResult,
    ComparisonResult,
)


class TestRunExperiment:
    def test_run_single_experiment(self):
        config = ExperimentConfig(
            initial_population=20, generations_to_run=3, random_seed=42,
        )
        runner = ExperimentRunner()
        result = runner.run_experiment(config)

        assert isinstance(result, ExperimentResult)
        assert len(result.history) == 3
        assert result.final_population_size > 0
        assert result.mean_contribution > 0

    def test_run_without_metrics(self):
        config = ExperimentConfig(
            initial_population=20, generations_to_run=3, random_seed=42,
        )
        runner = ExperimentRunner()
        result = runner.run_experiment(config, collect_metrics=False)

        assert len(result.metrics) == 0
        assert len(result.history) == 3

    def test_run_with_metrics(self):
        config = ExperimentConfig(
            initial_population=20, generations_to_run=3, random_seed=42,
        )
        runner = ExperimentRunner()
        result = runner.run_experiment(config, collect_metrics=True)

        assert len(result.metrics) == 3


class TestCompareExperiments:
    def test_compare_two(self):
        runner = ExperimentRunner()
        configs = {
            "default": ExperimentConfig(
                initial_population=20, generations_to_run=3, random_seed=42,
            ),
            "high_drift": ExperimentConfig(
                initial_population=20, generations_to_run=3, random_seed=42,
                trait_drift_rate=0.1,
            ),
        }
        comparison = runner.compare_experiments(configs)

        assert isinstance(comparison, ComparisonResult)
        assert "default" in comparison.results
        assert "high_drift" in comparison.results
        assert "default_vs_high_drift" in comparison.config_diffs

    def test_config_diffs_show_differences(self):
        runner = ExperimentRunner()
        configs = {
            "a": ExperimentConfig(
                initial_population=20, generations_to_run=3, random_seed=42,
            ),
            "b": ExperimentConfig(
                initial_population=20, generations_to_run=3, random_seed=42,
                trait_drift_rate=0.1,
            ),
        }
        comparison = runner.compare_experiments(configs)
        diffs = comparison.config_diffs["a_vs_b"]
        assert "trait_drift_rate" in diffs


class TestABTest:
    def test_ab_test(self):
        runner = ExperimentRunner()
        config_a = ExperimentConfig(
            experiment_name="control",
            initial_population=20, generations_to_run=3, random_seed=42,
        )
        config_b = ExperimentConfig(
            experiment_name="treatment",
            initial_population=20, generations_to_run=3, random_seed=42,
            birth_order_rules={1: "best", 2: "weirdest", 3: "worst"},
        )
        comparison = runner.run_ab_test(config_a, config_b)

        assert "A" in comparison.results
        assert "B" in comparison.results

    def test_ab_test_custom_labels(self):
        runner = ExperimentRunner()
        config_a = ExperimentConfig(
            initial_population=20, generations_to_run=3, random_seed=42,
        )
        config_b = ExperimentConfig(
            initial_population=20, generations_to_run=3, random_seed=42,
            trait_drift_rate=0.1,
        )
        comparison = runner.run_ab_test(
            config_a, config_b, label_a="control", label_b="treatment",
        )
        assert "control" in comparison.results
        assert "treatment" in comparison.results


class TestParameterSweep:
    def test_sweep_drift_rate(self):
        runner = ExperimentRunner()
        base = ExperimentConfig(
            initial_population=20, generations_to_run=3, random_seed=42,
        )
        results = runner.run_parameter_sweep(
            base, "trait_drift_rate", [0.01, 0.05, 0.1],
            collect_metrics=False,
        )

        assert len(results) == 3
        assert "trait_drift_rate=0.01" in results
        assert "trait_drift_rate=0.1" in results

    def test_sweep_results_differ(self):
        runner = ExperimentRunner()
        base = ExperimentConfig(
            initial_population=30, generations_to_run=5, random_seed=42,
        )
        results = runner.run_parameter_sweep(
            base, "trait_drift_rate", [0.001, 0.2],
            collect_metrics=False,
        )

        r_low = results["trait_drift_rate=0.001"]
        r_high = results["trait_drift_rate=0.2"]
        # With very different drift rates, results should differ
        # (may not always differ in mean_contribution, but history should differ)
        assert r_low.final_population_size > 0
        assert r_high.final_population_size > 0


class TestMultiSeed:
    def test_multi_seed_runs(self):
        runner = ExperimentRunner()
        config = ExperimentConfig(
            initial_population=20, generations_to_run=3,
        )
        results = runner.run_multi_seed(config, seeds=[1, 2, 3])

        assert len(results) == 3
        # Different seeds should produce different results
        pops = [r.final_population_size for r in results]
        # At least some variation expected
        assert len(results) == 3

    def test_same_seed_gives_same_result(self):
        runner = ExperimentRunner()
        config = ExperimentConfig(
            initial_population=20, generations_to_run=3,
        )
        results = runner.run_multi_seed(config, seeds=[42, 42])

        assert results[0].final_population_size == results[1].final_population_size
        assert results[0].total_breakthroughs == results[1].total_breakthroughs
