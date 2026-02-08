#!/usr/bin/env python3
"""Run a baseline Seldon Sandbox simulation and print results."""

from seldon.core.config import ExperimentConfig
from seldon.core.engine import SimulationEngine


def main():
    config = ExperimentConfig(
        experiment_name="baseline",
        initial_population=100,
        generations_to_run=25,
        random_seed=42,
    )

    print(f"=== Seldon Sandbox: {config.experiment_name} ===")
    print(f"Trait system: {config.trait_system}")
    print(f"Population: {config.initial_population}")
    print(f"Generations: {config.generations_to_run}")
    print(f"Birth order rules: {config.birth_order_rules}")
    print()

    engine = SimulationEngine(config)
    history = engine.run()

    print(f"{'Gen':>4} {'Pop':>5} {'Births':>6} {'Deaths':>6} {'BT':>3} "
          f"{'Pairs':>5} {'AvgAge':>6} {'Suffer':>7} {'Contrib':>8} "
          f"{'R1':>4} {'R2':>4} {'R3':>4} {'R4':>4} {'R5':>4}")
    print("-" * 90)

    for snap in history:
        rc = snap.region_counts
        print(
            f"{snap.generation:4d} {snap.population_size:5d} "
            f"{snap.births:6d} {snap.deaths:6d} {snap.breakthroughs:3d} "
            f"{snap.pairs_formed:5d} {snap.mean_age:6.1f} "
            f"{snap.mean_suffering:7.3f} {snap.mean_contribution:8.3f} "
            f"{rc.get('under_processing', 0):4d} "
            f"{rc.get('optimal', 0):4d} "
            f"{rc.get('deep', 0):4d} "
            f"{rc.get('sacrificial', 0):4d} "
            f"{rc.get('pathological', 0):4d}"
        )

    final = history[-1]
    print()
    print(f"=== Final State (Generation {final.generation}) ===")
    print(f"Population: {final.population_size}")
    print(f"Total breakthroughs: {sum(s.breakthroughs for s in history)}")
    print(f"Mean contribution: {final.mean_contribution:.3f}")
    print(f"Mean suffering: {final.mean_suffering:.3f}")
    print(f"Mean age: {final.mean_age:.1f}")

    # Region distribution
    total = final.population_size
    if total > 0:
        print(f"\nRegion distribution:")
        for region, count in sorted(final.region_counts.items()):
            pct = count / total * 100
            print(f"  {region:20s}: {count:4d} ({pct:5.1f}%)")

    # Birth order distribution
    print(f"\nBirth order distribution:")
    for bo, count in sorted(final.birth_order_counts.items()):
        print(f"  Order {bo}: {count}")


if __name__ == "__main__":
    main()
