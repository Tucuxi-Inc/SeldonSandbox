"""
Phase 2 Demo — exercises outsider injection, lore, archetypes, and the experiment runner.

Demonstrates:
1. Archetype-based society with Einstein and Curie injected
2. Lore evolution across generations
3. A/B testing with the ExperimentRunner
4. Outsider ripple tracking
"""

from seldon.core.config import ExperimentConfig
from seldon.core.engine import SimulationEngine
from seldon.experiment.archetypes import list_archetypes, get_archetype
from seldon.experiment.outsider import OutsiderInterface, RippleTracker
from seldon.experiment.presets import get_preset, list_presets
from seldon.experiment.runner import ExperimentRunner
from seldon.metrics.collector import MetricsCollector


def demo_archetype_injection():
    """Inject archetypes into a baseline population and observe effects."""
    print("=" * 60)
    print("DEMO 1: Archetype Injection + Ripple Tracking")
    print("=" * 60)

    config = ExperimentConfig(
        experiment_name="archetype_demo",
        initial_population=50,
        generations_to_run=15,
        random_seed=42,
        lore_enabled=True,
        scheduled_injections=[
            {"generation": 3, "archetype": "einstein", "count": 2, "noise": 0.05},
            {"generation": 3, "archetype": "curie", "count": 2, "noise": 0.05},
        ],
    )

    engine = SimulationEngine(config)
    history = engine.run()

    print(f"\nAvailable archetypes: {list_archetypes()}")
    print(f"\nSimulation ran {len(history)} generations")
    print(f"{'Gen':>3} {'Pop':>4} {'Births':>6} {'Deaths':>6} {'BT':>3} {'Outsiders':>9} {'AvgSuf':>7} {'AvgCon':>7}")
    print("-" * 60)
    for s in history:
        print(
            f"{s.generation:3d} {s.population_size:4d} {s.births:6d} {s.deaths:6d} "
            f"{s.breakthroughs:3d} {s.events.get('outsiders_injected', 0):9d} "
            f"{s.mean_suffering:7.3f} {s.mean_contribution:7.3f}"
        )

    # Ripple tracker report
    report = engine.ripple_tracker.get_diffusion_report()
    print(f"\nRipple Tracker: {report['injections']} injections")
    for snap in report["snapshots"]:
        if snap["outsider_count"] > 0 or snap["descendant_count"] > 0:
            print(
                f"  Gen {snap['generation']}: "
                f"{snap['outsider_count']} outsiders, "
                f"{snap['descendant_count']} descendants, "
                f"{snap['outsider_fraction']:.1%} of pop"
            )

    # Lore stats
    total_memories = sum(
        len(a.personal_memories) + len(a.inherited_lore)
        for a in engine.population
    )
    print(f"\nTotal memories in living population: {total_memories}")
    print(f"Societal memories: {len(engine.lore_engine.societal_memories)}")


def demo_ab_test():
    """Compare default vs inverted birth order rules."""
    print("\n" + "=" * 60)
    print("DEMO 2: A/B Test — Default vs Inverted Birth Order")
    print("=" * 60)

    runner = ExperimentRunner()
    comparison = runner.run_ab_test(
        config_a=ExperimentConfig(
            experiment_name="default_order",
            initial_population=50,
            generations_to_run=20,
            random_seed=42,
        ),
        config_b=ExperimentConfig(
            experiment_name="inverted_order",
            initial_population=50,
            generations_to_run=20,
            random_seed=42,
            birth_order_rules={1: "best", 2: "weirdest", 3: "worst"},
        ),
        label_a="Default",
        label_b="Inverted",
        collect_metrics=False,
    )

    print(f"\n{'Metric':<25} {'Default':>10} {'Inverted':>10}")
    print("-" * 47)
    for label in ["Default", "Inverted"]:
        r = comparison.results[label]
        print(f"{'Final population':<25} {r.final_population_size:>10d}" if label == "Default" else "", end="")
    print()

    for metric_name in ["final_population_size", "total_breakthroughs", "mean_contribution", "mean_suffering"]:
        vals = [getattr(comparison.results[l], metric_name) for l in ["Default", "Inverted"]]
        if isinstance(vals[0], float):
            print(f"{metric_name:<25} {vals[0]:>10.4f} {vals[1]:>10.4f}")
        else:
            print(f"{metric_name:<25} {vals[0]:>10d} {vals[1]:>10d}")

    # Config differences
    print("\nConfig differences:")
    for key, diffs in comparison.config_diffs.items():
        print(f"  {key}:")
        for param, (v1, v2) in diffs.items():
            print(f"    {param}: {v1} → {v2}")


def demo_parameter_sweep():
    """Sweep trait drift rate to find optimal value."""
    print("\n" + "=" * 60)
    print("DEMO 3: Parameter Sweep — Trait Drift Rate")
    print("=" * 60)

    runner = ExperimentRunner()
    base = ExperimentConfig(
        initial_population=50,
        generations_to_run=15,
        random_seed=42,
    )

    results = runner.run_parameter_sweep(
        base, "trait_drift_rate",
        [0.001, 0.01, 0.05, 0.1, 0.2],
        collect_metrics=False,
    )

    print(f"\n{'Drift Rate':<20} {'FinalPop':>8} {'Breakthroughs':>13} {'MeanContrib':>12} {'MeanSuffer':>11}")
    print("-" * 66)
    for label, result in results.items():
        print(
            f"{label:<20} {result.final_population_size:>8d} "
            f"{result.total_breakthroughs:>13d} "
            f"{result.mean_contribution:>12.4f} "
            f"{result.mean_suffering:>11.4f}"
        )


def demo_presets():
    """List available presets and run one."""
    print("\n" + "=" * 60)
    print("DEMO 4: Presets")
    print("=" * 60)

    print(f"\nAvailable presets: {list_presets()}")

    config = get_preset("high_sacrificial")
    config = ExperimentConfig(
        **{**config.to_dict(), "initial_population": 30, "generations_to_run": 10, "random_seed": 42}
    )
    engine = SimulationEngine(config)
    history = engine.run()

    print(f"\nPreset 'high_sacrificial' — {len(history)} generations:")
    last = history[-1]
    print(f"  Final population: {last.population_size}")
    print(f"  Region distribution: {last.region_counts}")
    print(f"  Total breakthroughs: {sum(s.breakthroughs for s in history)}")


if __name__ == "__main__":
    demo_archetype_injection()
    demo_ab_test()
    demo_parameter_sweep()
    demo_presets()
    print("\n" + "=" * 60)
    print("Phase 2 demo complete!")
    print("=" * 60)
