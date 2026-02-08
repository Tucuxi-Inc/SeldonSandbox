"""
Experiment Runner — A/B testing, parameter sweeps, and batch execution.

Provides tools for running comparative experiments, sweeping parameters,
and collecting results across multiple simulation runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from seldon.core.config import ExperimentConfig
from seldon.core.engine import SimulationEngine, GenerationSnapshot
from seldon.metrics.collector import MetricsCollector


@dataclass
class ExperimentResult:
    """Result of a single experiment run."""
    config: ExperimentConfig
    history: list[GenerationSnapshot]
    metrics: list[Any]  # GenerationMetrics from collector
    final_population_size: int
    total_breakthroughs: int
    mean_contribution: float
    mean_suffering: float


@dataclass
class ComparisonResult:
    """Result of comparing two or more experiments."""
    results: dict[str, ExperimentResult]
    config_diffs: dict[str, Any]


class ExperimentRunner:
    """
    Run, compare, and sweep simulation experiments.
    """

    def run_experiment(
        self,
        config: ExperimentConfig,
        collect_metrics: bool = True,
    ) -> ExperimentResult:
        """Run a single experiment and return results."""
        engine = SimulationEngine(config)
        history = engine.run()

        metrics_list = []
        if collect_metrics:
            collector = MetricsCollector(config)
            for snap in history:
                m = collector.collect(engine.population, snap)
                metrics_list.append(m)

        # Aggregate summary stats
        total_bt = sum(s.breakthroughs for s in history)
        all_contrib = [s.mean_contribution for s in history if s.population_size > 0]
        all_suffer = [s.mean_suffering for s in history if s.population_size > 0]

        return ExperimentResult(
            config=config,
            history=history,
            metrics=metrics_list,
            final_population_size=history[-1].population_size if history else 0,
            total_breakthroughs=total_bt,
            mean_contribution=float(np.mean(all_contrib)) if all_contrib else 0.0,
            mean_suffering=float(np.mean(all_suffer)) if all_suffer else 0.0,
        )

    def compare_experiments(
        self,
        configs: dict[str, ExperimentConfig],
        collect_metrics: bool = True,
    ) -> ComparisonResult:
        """Run multiple experiments and compare results."""
        results: dict[str, ExperimentResult] = {}
        for name, config in configs.items():
            results[name] = self.run_experiment(config, collect_metrics)

        # Compute config diffs between all pairs
        config_names = list(configs.keys())
        diffs: dict[str, Any] = {}
        if len(config_names) >= 2:
            base = configs[config_names[0]]
            for name in config_names[1:]:
                diffs[f"{config_names[0]}_vs_{name}"] = base.diff(configs[name])

        return ComparisonResult(results=results, config_diffs=diffs)

    def run_ab_test(
        self,
        config_a: ExperimentConfig,
        config_b: ExperimentConfig,
        label_a: str = "A",
        label_b: str = "B",
        collect_metrics: bool = True,
    ) -> ComparisonResult:
        """Run an A/B test between two configurations."""
        return self.compare_experiments(
            {label_a: config_a, label_b: config_b},
            collect_metrics=collect_metrics,
        )

    def run_parameter_sweep(
        self,
        base_config: ExperimentConfig,
        param_name: str,
        values: list[Any],
        collect_metrics: bool = True,
    ) -> dict[str, ExperimentResult]:
        """
        Sweep a single parameter across multiple values.

        Args:
            base_config: Base configuration to modify
            param_name: Name of the parameter to sweep (attribute on ExperimentConfig)
            values: List of values to test
            collect_metrics: Whether to collect detailed metrics

        Returns:
            Dict mapping value label -> ExperimentResult
        """
        results: dict[str, ExperimentResult] = {}

        for val in values:
            # Create config copy with modified parameter
            config_dict = base_config.to_dict()
            config_dict[param_name] = val
            config_dict["experiment_name"] = f"sweep_{param_name}={val}"
            config = ExperimentConfig.from_dict(config_dict)

            label = f"{param_name}={val}"
            results[label] = self.run_experiment(config, collect_metrics)

        return results

    def run_multi_seed(
        self,
        config: ExperimentConfig,
        seeds: list[int],
        collect_metrics: bool = False,
    ) -> list[ExperimentResult]:
        """
        Run the same configuration with multiple random seeds.

        Useful for measuring variance in outcomes.
        """
        results: list[ExperimentResult] = []
        for seed in seeds:
            config_dict = config.to_dict()
            config_dict["random_seed"] = seed
            config_dict["experiment_name"] = f"{config.experiment_name}_seed{seed}"
            seed_config = ExperimentConfig.from_dict(config_dict)
            results.append(self.run_experiment(seed_config, collect_metrics))
        return results
