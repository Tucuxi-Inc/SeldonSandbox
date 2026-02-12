"""
Master configuration for the Seldon Sandbox.

ALL tunable parameters live here. Nothing in the simulation is hardcoded.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from seldon.core.traits import TraitSystem


@dataclass
class ExperimentConfig:
    """
    Master configuration — ALL parameters as tunable sliders.

    Every threshold, weight, rate, and rule is configurable.
    Use ``to_dict()`` / ``from_dict()`` for serialization and comparison.
    """

    # === Experiment identity ===
    experiment_name: str = "default"
    random_seed: int | None = None

    # === Trait system ===
    trait_preset: str = "compact"  # 'compact' (15), 'full' (50), 'custom'
    custom_traits: list[dict[str, Any]] | None = None

    # === Population ===
    initial_population: int = 100
    generations_to_run: int = 50

    # === Birth order rules ===
    # Keys are birth order positions; values are rule names.
    # Default 4+ falls back to 'random_weighted' in InheritanceEngine.
    birth_order_rules: dict[int, str] = field(default_factory=lambda: {
        1: "worst",
        2: "weirdest",
        3: "best",
    })
    inheritance_noise_sigma: float = 0.05

    # === RSH Processing regions ===
    region_thresholds: dict[str, float] = field(default_factory=lambda: {
        "under_to_optimal": 0.3,
        "optimal_to_deep": 0.5,
        "deep_to_extreme": 0.8,
        "productive_potential_threshold": 0.5,
    })
    productive_weights: dict[str, float] = field(default_factory=lambda: {
        "creativity": 0.4,
        "resilience": 0.3,
        "burnout_penalty": 0.3,
    })

    # === Region effects on traits (drift modifiers when in a region) ===
    # Maps region name -> dict of trait_name -> drift modifier
    region_effects: dict[str, dict[str, float]] = field(default_factory=lambda: {
        "under_processing": {},
        "optimal": {},
        "deep": {"depth_drive": 0.01, "neuroticism": 0.005},
        "sacrificial": {"depth_drive": 0.02, "neuroticism": 0.02, "resilience": -0.01},
        "pathological": {"neuroticism": 0.03, "resilience": -0.02, "self_control": -0.01},
    })

    # === Trait drift ===
    trait_drift_rate: float = 0.02
    trait_drift_age_factor: float = 0.01  # Drift magnitude decreases as 1/(1 + age * factor)

    # === Attraction model ===
    attraction_weights: dict[str, float] = field(default_factory=lambda: {
        "similarity": 0.3,
        "complementarity": 0.2,
        "universal_attractiveness": 0.1,
        "social_proximity": 0.15,
        "age_compatibility": 0.1,
        "random_chemistry": 0.15,
    })

    # === Decision model ===
    decision_temperature: float = 1.0  # Higher = more random; lower = more deterministic

    # === Cognitive council ===
    cognitive_council_enabled: bool = False  # Off by default for faster sims
    cognitive_council_weights: dict[str, list[tuple[str, float]]] | None = None  # None = use defaults

    # === Relationships ===
    relationship_config: dict[str, Any] = field(default_factory=lambda: {
        "pairing_min_age": 16,
        "pairing_permanent": False,
        "dissolution_enabled": True,
        "dissolution_compatibility_threshold": 0.3,
        "dissolution_base_rate": 0.05,
        "reparing_after_death": True,
        "reparing_after_dissolution": True,
        "reparing_cooldown_generations": 1,
        "infidelity_enabled": False,
        "infidelity_base_rate": 0.15,
        "single_by_choice_rate": 0.10,
        "lgbtq_rate": 0.035,
        "assisted_reproduction_rate": 0.075,
    })

    # === Fertility ===
    fertility_config: dict[str, Any] = field(default_factory=lambda: {
        "female_fertility_start": 16,
        "female_fertility_end": 40,
        "male_fertility_start": 16,
        "min_birth_spacing_generations": 1,
        "max_children_per_generation": 1,
        "maternal_mortality_rate": 0.015,
        "child_mortality_rate": 0.30,
        "societal_fertility_pressure": 0.5,
        "target_children_mean": 3.0,
    })

    # === Mortality ===
    base_mortality_rate: float = 0.02
    age_mortality_factor: float = 0.001
    burnout_mortality_factor: float = 0.1

    # === Contribution ===
    contribution_config: dict[str, float] = field(default_factory=lambda: {
        "base_contribution": 0.5,
        "region_multipliers": {
            "under_processing": 0.3,
            "optimal": 1.0,
            "deep": 1.5,
            "sacrificial": 2.0,
            "pathological": 0.0,
        },
        "creativity_weight": 0.3,
        "resilience_weight": 0.2,
        "conscientiousness_weight": 0.2,
        "breakthrough_threshold": 0.95,
        "breakthrough_base_probability": 0.02,
    })

    # === Lore / Memory ===
    lore_enabled: bool = True
    lore_decay_rate: float = 0.05
    lore_myth_threshold: float = 0.3
    lore_mutation_rate: float = 0.02
    lore_transmission_rate: float = 0.7

    # === Outsider injection ===
    outsider_injection_age: int = 20
    scheduled_injections: list[dict[str, Any]] = field(default_factory=list)

    # === Social hierarchy (Phase 7) ===
    hierarchy_config: dict[str, Any] = field(default_factory=lambda: {
        "status_contribution_weight": 0.3,
        "status_age_weight": 0.15,
        "status_family_weight": 0.15,
        "status_region_weight": 0.2,
        "status_social_weight": 0.2,
        "role_thresholds": {
            "leader_percentile": 0.95,
            "leader_extraversion_min": 0.65,
            "innovator_creativity_min": 0.7,
            "mediator_agreeableness_min": 0.7,
        },
        "influence_decay_rate": 0.1,
    })
    mentorship_config: dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "max_mentees": 3,
        "mentor_min_age": 25,
        "mentee_max_age": 25,
        "mentorship_influence_rate": 0.01,
        "dissolution_compatibility_threshold": 0.2,
        "skill_transfer_rate": 0.05,
    })

    # === Genetics (Phase 8) ===
    genetics_config: dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "mutation_rate": 0.001,
        "crossover_rate": 0.5,
        "dominance_modifier": 0.1,
        "gene_trait_influence": 0.3,
    })
    epigenetics_config: dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "transgenerational_rate": 0.3,
        "activation_threshold_generations": 3,
        "max_active_markers": 5,
    })

    # === Marriage, Clans, Institutions (Phase D) ===
    marriage_config: dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "formalization_delay_generations": 1,
        "divorce_base_rate": 0.02,
        "property_sharing_rate": 0.5,
        "political_marriage_enabled": True,
        "political_marriage_alliance_bonus": 0.1,
    })
    clan_config: dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "min_living_members": 3,
        "founder_min_status": 0.6,
        "honor_status_weight": 0.1,
        "rival_threshold": 0.3,
        "max_clans": 10,
        "max_depth": 3,
    })
    institutions_config: dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "council_min_elders": 3,
        "council_elder_min_age": 40,
        "council_min_community_size": 10,
        "guild_min_members": 5,
        "guild_min_skill": 0.3,
        "institution_prestige_weight": 0.1,
        "election_frequency_generations": 5,
    })

    # === Community (Phase 9) ===
    community_config: dict[str, Any] = field(default_factory=lambda: {
        "cohesion_trait_weight": 0.4,
        "cohesion_bond_weight": 0.3,
        "cohesion_culture_weight": 0.2,
        "cohesion_conflict_weight": 0.1,
        "faction_detection_threshold": 0.3,
        "min_community_size": 5,
    })
    diplomacy_config: dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "standing_learning_rate": 0.05,
        "alliance_threshold": 0.7,
        "rivalry_threshold": -0.5,
        "cultural_exchange_rate": 0.1,
        "trait_compatibility_weight": 0.3,
        "leader_compatibility_weight": 0.2,
        "resource_competition_weight": 0.25,
        "cultural_similarity_weight": 0.25,
    })

    # === Economics (Phase 10 + Phase C) ===
    economics_config: dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "base_production_rate": 1.0,
        "trade_distance_cost": 0.1,
        "price_elasticity": 0.5,
        "poverty_threshold": 0.2,
        "poverty_mortality_multiplier": 1.5,
        "wealth_inheritance_rate": 0.7,
        "occupation_change_rate": 0.1,
        "specialization_bonus": 0.3,
        # Phase C: per-tick economy
        "consumption_per_tick": {"food": 0.03, "goods": 0.01, "knowledge": 0.005},
        "skill_gain_per_tick": 0.01,
        "skill_decay_rate": 0.002,
        "skill_max": 1.0,
        "min_settlement_size": 3,
        "trade_frequency_ticks": 3,
        "trade_volume_base": 1.0,
        "trader_facilitation_income": 0.05,
        "governance_enabled": True,
        "base_taxation_rate": 0.10,
        "infrastructure_growth_rate": 0.01,
        "infrastructure_capacity_bonus": 2,
        "poverty_relief_priority": 0.5,
        "terrain_production_modifiers": {
            "farmer": {
                "valley": 1.3, "river_valley": 1.4, "coastal": 1.0,
                "grassland": 1.1, "forest": 0.7, "foothills": 0.6,
                "mountain": 0.3, "desert": 0.2, "tundra": 0.2,
            },
            "artisan": {
                "valley": 1.0, "river_valley": 1.0, "coastal": 1.2,
                "grassland": 0.8, "forest": 1.1, "foothills": 1.2,
                "mountain": 1.3, "desert": 0.5, "tundra": 0.4,
            },
            "scholar": {
                "valley": 1.1, "river_valley": 1.1, "coastal": 1.0,
                "grassland": 0.9, "forest": 1.2, "foothills": 0.9,
                "mountain": 0.8, "desert": 0.6, "tundra": 0.5,
            },
            "soldier": {
                "valley": 1.0, "river_valley": 0.9, "coastal": 1.1,
                "grassland": 1.2, "forest": 1.1, "foothills": 1.3,
                "mountain": 1.0, "desert": 0.8, "tundra": 0.7,
            },
            "trader": {
                "valley": 1.1, "river_valley": 1.2, "coastal": 1.4,
                "grassland": 1.0, "forest": 0.7, "foothills": 0.8,
                "mountain": 0.5, "desert": 0.6, "tundra": 0.3,
            },
        },
        "season_production_modifiers": {
            "spring": {"food": 1.1, "goods": 1.0, "knowledge": 1.0},
            "summer": {"food": 1.3, "goods": 1.0, "knowledge": 0.9},
            "autumn": {"food": 1.2, "goods": 1.1, "knowledge": 1.0},
            "winter": {"food": 0.6, "goods": 1.1, "knowledge": 1.2},
        },
    })

    # === Environment (Phase 11) ===
    environment_config: dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "seasons_enabled": True,
        "season_length_generations": 5,
        "base_event_probability": 0.1,
        "drought_probability": 0.05,
        "flood_probability": 0.03,
        "plague_probability": 0.02,
        "bountiful_probability": 0.08,
        "discovery_probability": 0.04,
        "climate_drift_rate": 0.001,
        "disease_transmission_rate": 0.15,
        "disease_base_mortality": 0.1,
        "quarantine_effectiveness": 0.5,
    })

    # === Tick-based engine (Phase A) ===
    tick_config: dict[str, Any] = field(default_factory=lambda: {
        "enabled": False,
        "ticks_per_year": 12,
        "life_phase_boundaries": {
            "infant": [0, 2],
            "child": [3, 12],
            "adolescent": [13, 17],
            "young_adult": [18, 30],
            "mature": [31, 55],
            "elder": [56, 999],
        },
        "season_effects_enabled": True,
        "season_modifiers": {
            "spring": {"fertility_mult": 1.2, "mortality_mult": 0.9,
                       "contribution_mult": 1.0, "food_mult": 1.1, "mood_offset": 0.05},
            "summer": {"fertility_mult": 1.0, "mortality_mult": 0.95,
                       "contribution_mult": 1.1, "food_mult": 1.3, "mood_offset": 0.03},
            "autumn": {"fertility_mult": 0.8, "mortality_mult": 1.0,
                       "contribution_mult": 1.05, "food_mult": 1.2, "mood_offset": -0.02},
            "winter": {"fertility_mult": 0.6, "mortality_mult": 1.3,
                       "contribution_mult": 0.8, "food_mult": 0.6, "mood_offset": -0.05},
        },
    })

    # === Hex grid geography (Phase A) ===
    hex_grid_config: dict[str, Any] = field(default_factory=lambda: {
        "enabled": False,
        "width": 20,
        "height": 10,
        "map_generator": "california_slice",
        "starting_hex": [5, 5],
        "movement_cost_base": 1.0,
        "vision_range": 2,
        "stay_bias": 0.3,
    })

    # === Needs system (Phase A) ===
    needs_config: dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "base_decay_rates": {
            "hunger": 0.08,
            "thirst": 0.12,
            "shelter": 0.04,
            "safety": 0.05,
            "warmth": 0.06,
            "rest": 0.10,
        },
        "warning_threshold": 0.4,
        "critical_threshold": 0.2,
        "lethal_threshold": 0.05,
        "warning_suffering_per_tick": 0.01,
        "critical_suffering_per_tick": 0.03,
        "critical_health_damage_per_tick": 0.03,
        "lethal_suffering_per_tick": 0.06,
        "lethal_health_damage_per_tick": 0.08,
        "health_recovery_rate": 0.02,
        "health_mortality_threshold": 0.5,
        "needs_mortality_multiplier": 0.3,
        "burnout_decay_amplifier": 0.2,
    })

    # === Extensions ===
    extensions_enabled: list[str] = field(default_factory=list)
    extensions: dict[str, dict[str, Any]] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Derived / cached
    # ------------------------------------------------------------------
    _trait_system: TraitSystem | None = field(default=None, repr=False, compare=False)

    @property
    def trait_system(self) -> TraitSystem:
        """Lazily build and cache the TraitSystem instance."""
        if self._trait_system is None:
            self._trait_system = TraitSystem(
                preset=self.trait_preset,
                custom_traits=self.custom_traits,
            )
        return self._trait_system

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict (excludes cached objects)."""
        d: dict[str, Any] = {}
        for k, v in self.__dict__.items():
            if k.startswith("_"):
                continue
            d[k] = v
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ExperimentConfig:
        """Deserialize from a dict."""
        return cls(**{k: v for k, v in d.items() if not k.startswith("_")})

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_json(cls, s: str) -> ExperimentConfig:
        return cls.from_dict(json.loads(s))

    def enable_extension(
        self, name: str,
        config_overrides: dict[str, Any] | None = None,
    ) -> None:
        """Enable an extension and optionally set its configuration."""
        if name not in self.extensions_enabled:
            self.extensions_enabled.append(name)
        if name not in self.extensions:
            self.extensions[name] = {}
        if config_overrides:
            self.extensions[name].update(config_overrides)

    def configure_extension(self, name: str, **kwargs: Any) -> None:
        """Update configuration parameters for a named extension."""
        if name not in self.extensions:
            self.extensions[name] = {}
        self.extensions[name].update(kwargs)

    def diff(self, other: ExperimentConfig) -> dict[str, tuple[Any, Any]]:
        """Return parameters that differ between two configs."""
        diffs: dict[str, tuple[Any, Any]] = {}
        for k in self.to_dict():
            v1 = getattr(self, k)
            v2 = getattr(other, k)
            if v1 != v2:
                diffs[k] = (v1, v2)
        return diffs
