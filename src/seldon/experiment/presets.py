"""
Experiment presets — pre-configured experiment templates.

Each preset returns an ExperimentConfig with specific parameter settings
designed to test different hypotheses about societal dynamics.
"""

from __future__ import annotations

from seldon.core.config import ExperimentConfig


def baseline() -> ExperimentConfig:
    """Standard baseline configuration with default parameters."""
    return ExperimentConfig(
        experiment_name="baseline",
        initial_population=100,
        generations_to_run=50,
    )


def no_birth_order() -> ExperimentConfig:
    """All children inherit via averaging — no birth order effects."""
    return ExperimentConfig(
        experiment_name="no_birth_order",
        initial_population=100,
        generations_to_run=50,
        birth_order_rules={1: "average", 2: "average", 3: "average"},
    )


def inverted_birth_order() -> ExperimentConfig:
    """Inverted rules: 1st=best, 2nd=weirdest, 3rd=worst."""
    return ExperimentConfig(
        experiment_name="inverted_birth_order",
        initial_population=100,
        generations_to_run=50,
        birth_order_rules={1: "best", 2: "weirdest", 3: "worst"},
    )


def high_sacrificial() -> ExperimentConfig:
    """Lower R4 threshold to produce more sacrificial processing agents."""
    return ExperimentConfig(
        experiment_name="high_sacrificial",
        initial_population=100,
        generations_to_run=50,
        region_thresholds={
            "under_to_optimal": 0.3,
            "optimal_to_deep": 0.4,
            "deep_to_extreme": 0.6,  # Lower threshold → more extreme
            "productive_potential_threshold": 0.4,  # More end up in R4 vs R5
        },
    )


def no_recovery() -> ExperimentConfig:
    """Agents in R3/R4/R5 cannot recover — suffering only accumulates."""
    return ExperimentConfig(
        experiment_name="no_recovery",
        initial_population=100,
        generations_to_run=50,
        region_effects={
            "under_processing": {},
            "optimal": {},
            "deep": {"depth_drive": 0.02, "neuroticism": 0.01},
            "sacrificial": {"depth_drive": 0.03, "neuroticism": 0.03, "resilience": -0.02},
            "pathological": {"neuroticism": 0.05, "resilience": -0.03, "self_control": -0.02},
        },
    )


def high_trait_drift() -> ExperimentConfig:
    """High trait drift rate — personality changes rapidly over time."""
    return ExperimentConfig(
        experiment_name="high_trait_drift",
        initial_population=100,
        generations_to_run=50,
        trait_drift_rate=0.1,
    )


def opposites_attract() -> ExperimentConfig:
    """High complementarity weight — opposites attract for pairing."""
    return ExperimentConfig(
        experiment_name="opposites_attract",
        initial_population=100,
        generations_to_run=50,
        attraction_weights={
            "similarity": 0.1,
            "complementarity": 0.5,
            "universal_attractiveness": 0.1,
            "social_proximity": 0.1,
            "age_compatibility": 0.1,
            "random_chemistry": 0.1,
        },
    )


def archetype_society() -> ExperimentConfig:
    """
    Society seeded with archetype outsiders at generation 0.

    Use with OutsiderInterface.inject_archetype() during setup.
    Small initial pop + archetype injections.
    """
    return ExperimentConfig(
        experiment_name="archetype_society",
        initial_population=20,
        generations_to_run=50,
    )


def high_lore_decay() -> ExperimentConfig:
    """Fast lore decay — memories become myths quickly."""
    return ExperimentConfig(
        experiment_name="high_lore_decay",
        initial_population=100,
        generations_to_run=50,
        lore_enabled=True,
        lore_decay_rate=0.2,
        lore_myth_threshold=0.5,
        lore_mutation_rate=0.1,
    )


def stable_lore() -> ExperimentConfig:
    """Very slow lore decay — memories persist across many generations."""
    return ExperimentConfig(
        experiment_name="stable_lore",
        initial_population=100,
        generations_to_run=50,
        lore_enabled=True,
        lore_decay_rate=0.01,
        lore_myth_threshold=0.1,
        lore_mutation_rate=0.005,
        lore_transmission_rate=0.9,
    )


# Registry of all presets
PRESETS: dict[str, callable] = {
    "baseline": baseline,
    "no_birth_order": no_birth_order,
    "inverted_birth_order": inverted_birth_order,
    "high_sacrificial": high_sacrificial,
    "no_recovery": no_recovery,
    "high_trait_drift": high_trait_drift,
    "opposites_attract": opposites_attract,
    "archetype_society": archetype_society,
    "high_lore_decay": high_lore_decay,
    "stable_lore": stable_lore,
}


def get_preset(name: str) -> ExperimentConfig:
    """Get a preset config by name."""
    if name not in PRESETS:
        raise KeyError(f"Unknown preset: '{name}'. Available: {list(PRESETS.keys())}")
    return PRESETS[name]()


def list_presets() -> list[str]:
    """Return list of available preset names."""
    return list(PRESETS.keys())
