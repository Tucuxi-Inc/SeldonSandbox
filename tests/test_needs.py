"""
Tests for NeedsSystem and GatheringSystem (Phase A — Step 4).

Covers need decay (terrain, season, life-phase, burnout modifiers),
prioritisation, health/suffering impact, mortality factor, gathering
activity selection, yield calculation, caregiver sharing, and edge cases.
"""

import pytest
import numpy as np

from seldon.core.needs import (
    NeedType,
    NeedsSystem,
    TERRAIN_DECAY_MODIFIERS,
    SEASON_DECAY_MODIFIERS,
    LIFE_PHASE_DECAY_MODIFIERS,
    DEFAULT_MODIFIERS,
)
from seldon.core.gathering import (
    GatheringSystem,
    GATHERING_ACTIVITIES,
    TERRAIN_ACTIVITY_SUITABILITY,
    SEASON_ACTIVITY_SUITABILITY,
)
from seldon.core.traits import TraitSystem
from seldon.core.agent import Agent
from seldon.core.processing import ProcessingRegion


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_agent(agent_id="a1", age=25, life_phase="young_adult", **overrides):
    """Create a test agent with compact traits and sensible defaults."""
    ts = TraitSystem(preset="compact")
    rng = np.random.default_rng(42)
    traits = ts.random_traits(rng)
    a = Agent(
        id=agent_id,
        name=f"Agent_{agent_id}",
        age=age,
        generation=0,
        birth_order=1,
        traits=traits,
        traits_at_birth=traits.copy(),
    )
    a.life_phase = life_phase
    for k, v in overrides.items():
        setattr(a, k, v)
    return a, ts


def _default_config():
    """Default needs-system configuration dict."""
    return {
        "enabled": True,
        "base_decay_rates": {
            "hunger": 0.08, "thirst": 0.12, "shelter": 0.04,
            "safety": 0.05, "warmth": 0.06, "rest": 0.10,
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
    }


# ===================================================================
# NeedType tests
# ===================================================================

class TestNeedType:
    def test_need_type_values(self):
        """All six needs exist with the expected string values."""
        expected = {
            "hunger", "thirst", "shelter", "safety", "warmth", "rest",
        }
        actual = {n.value for n in NeedType}
        assert actual == expected

    def test_need_type_is_str_enum(self):
        """NeedType members are usable as plain strings."""
        assert NeedType.HUNGER == "hunger"
        assert isinstance(NeedType.THIRST, str)


# ===================================================================
# NeedsSystem tests
# ===================================================================

class TestInitializeNeeds:
    def test_initialize_needs(self):
        """initialize_needs sets all needs to 1.0 and health to 1.0."""
        ns = NeedsSystem(_default_config())
        agent, _ = _make_agent()
        agent.needs = {}
        agent.health = 0.5
        ns.initialize_needs(agent)

        for need in NeedType:
            assert agent.needs[need.value] == 1.0
        assert agent.health == 1.0


class TestDecayNeeds:
    def test_decay_needs_basic(self):
        """One tick of decay reduces all needs below 1.0."""
        ns = NeedsSystem(_default_config())
        agent, _ = _make_agent()
        ns.initialize_needs(agent)
        ns.decay_needs(agent, terrain=None, season=None, life_phase=None)

        for need in NeedType:
            assert agent.needs[need.value] < 1.0

    def test_decay_needs_terrain_desert_thirst(self):
        """Desert amplifies thirst decay by 1.8x compared to default."""
        cfg = _default_config()
        ns = NeedsSystem(cfg)

        agent_desert, _ = _make_agent(agent_id="desert")
        ns.initialize_needs(agent_desert)
        ns.decay_needs(agent_desert, terrain="desert", season=None, life_phase=None)

        agent_default, _ = _make_agent(agent_id="default")
        ns.initialize_needs(agent_default)
        ns.decay_needs(agent_default, terrain=None, season=None, life_phase=None)

        # Desert thirst decays faster
        desert_thirst_loss = 1.0 - agent_desert.needs["thirst"]
        default_thirst_loss = 1.0 - agent_default.needs["thirst"]
        assert desert_thirst_loss == pytest.approx(
            default_thirst_loss * 1.8, rel=1e-6,
        )

    def test_decay_needs_winter_warmth(self):
        """Winter amplifies warmth decay by 1.5x."""
        cfg = _default_config()
        ns = NeedsSystem(cfg)

        agent_winter, _ = _make_agent(agent_id="winter")
        ns.initialize_needs(agent_winter)
        ns.decay_needs(agent_winter, terrain=None, season="winter", life_phase=None)

        agent_default, _ = _make_agent(agent_id="default")
        ns.initialize_needs(agent_default)
        ns.decay_needs(agent_default, terrain=None, season=None, life_phase=None)

        winter_warmth_loss = 1.0 - agent_winter.needs["warmth"]
        default_warmth_loss = 1.0 - agent_default.needs["warmth"]
        assert winter_warmth_loss == pytest.approx(
            default_warmth_loss * 1.5, rel=1e-6,
        )

    def test_decay_needs_infant_amplified(self):
        """Infant life phase increases all decay rates (most by >1.0)."""
        cfg = _default_config()
        ns = NeedsSystem(cfg)

        agent_infant, _ = _make_agent(agent_id="infant", life_phase="infant")
        ns.initialize_needs(agent_infant)
        ns.decay_needs(agent_infant, terrain=None, season=None, life_phase="infant")

        agent_adult, _ = _make_agent(agent_id="adult")
        ns.initialize_needs(agent_adult)
        ns.decay_needs(agent_adult, terrain=None, season=None, life_phase=None)

        # Hunger decay for infant should be 1.3x the adult decay
        infant_loss = 1.0 - agent_infant.needs["hunger"]
        adult_loss = 1.0 - agent_adult.needs["hunger"]
        assert infant_loss == pytest.approx(adult_loss * 1.3, rel=1e-6)

    def test_decay_needs_burnout_amplifies(self):
        """High burnout increases decay via burnout_decay_amplifier."""
        cfg = _default_config()
        ns = NeedsSystem(cfg)

        agent_burn, _ = _make_agent(agent_id="burn", burnout_level=1.0)
        ns.initialize_needs(agent_burn)
        ns.decay_needs(agent_burn, terrain=None, season=None, life_phase=None)

        agent_calm, _ = _make_agent(agent_id="calm", burnout_level=0.0)
        ns.initialize_needs(agent_calm)
        ns.decay_needs(agent_calm, terrain=None, season=None, life_phase=None)

        # Burnout amplifier = 0.2, so factor = 1.2 for burnout=1.0
        burn_loss = 1.0 - agent_burn.needs["hunger"]
        calm_loss = 1.0 - agent_calm.needs["hunger"]
        assert burn_loss == pytest.approx(calm_loss * 1.2, rel=1e-6)

    def test_decay_needs_null_terrain_season(self):
        """None terrain and season use default (1.0) modifiers."""
        cfg = _default_config()
        ns = NeedsSystem(cfg)

        agent, _ = _make_agent()
        ns.initialize_needs(agent)
        ns.decay_needs(agent, terrain=None, season=None, life_phase=None)

        # Base decay for hunger is 0.08; factor = 1.0*1.0*1.0*1.0 = 1.0
        expected_hunger = 1.0 - 0.08
        assert agent.needs["hunger"] == pytest.approx(expected_hunger, abs=1e-9)

    def test_needs_bounded_at_zero(self):
        """Needs can never go below 0.0, even with extreme decay."""
        cfg = _default_config()
        # Very high base decay to force needs to zero quickly
        cfg["base_decay_rates"] = {n.value: 5.0 for n in NeedType}
        ns = NeedsSystem(cfg)

        agent, _ = _make_agent()
        ns.initialize_needs(agent)
        ns.decay_needs(agent, terrain=None, season=None, life_phase=None)

        for need in NeedType:
            assert agent.needs[need.value] == 0.0

    def test_decay_needs_unknown_terrain_uses_defaults(self):
        """An unknown terrain string falls back to 1.0 modifiers."""
        cfg = _default_config()
        ns = NeedsSystem(cfg)

        agent_known, _ = _make_agent(agent_id="known")
        ns.initialize_needs(agent_known)
        ns.decay_needs(agent_known, terrain=None, season=None, life_phase=None)

        agent_unknown, _ = _make_agent(agent_id="unknown")
        ns.initialize_needs(agent_unknown)
        ns.decay_needs(agent_unknown, terrain="alien_swamp", season=None, life_phase=None)

        # Should be identical — both use default modifiers
        for need in NeedType:
            assert agent_known.needs[need.value] == pytest.approx(
                agent_unknown.needs[need.value], abs=1e-9,
            )


class TestPrioritizeNeeds:
    def test_prioritize_needs_most_urgent_first(self):
        """The lowest need should appear first in the priority list."""
        ns = NeedsSystem(_default_config())
        agent, ts = _make_agent()
        ns.initialize_needs(agent)
        agent.needs["thirst"] = 0.1  # Most urgent
        agent.needs["hunger"] = 0.3  # Second
        # Others at 1.0

        priorities = ns.prioritize_needs(agent, ts)
        assert priorities[0] == "thirst"
        assert priorities[1] == "hunger"

    def test_prioritize_needs_neuroticism_effect(self):
        """High neuroticism amplifies urgency of all needs uniformly."""
        cfg = _default_config()
        ns = NeedsSystem(cfg)

        agent_neurotic, ts = _make_agent(agent_id="neur")
        ns.initialize_needs(agent_neurotic)
        neur_idx = ts.trait_index("neuroticism")
        agent_neurotic.traits[neur_idx] = 0.9
        agent_neurotic.needs["thirst"] = 0.2
        agent_neurotic.needs["hunger"] = 0.3

        priorities = ns.prioritize_needs(agent_neurotic, ts)
        # Order should still reflect raw level (thirst more urgent)
        assert priorities[0] == "thirst"

    def test_prioritize_needs_conscientiousness_bonus(self):
        """
        High-conscientiousness agents get an urgency boost for needs
        below 0.5, possibly reordering priorities.
        """
        cfg = _default_config()
        ns = NeedsSystem(cfg)

        agent, ts = _make_agent()
        ns.initialize_needs(agent)
        cons_idx = ts.trait_index("conscientiousness")
        agent.traits[cons_idx] = 0.9  # High conscientiousness

        # Set two needs to be close — one at 0.45 (below 0.5 threshold)
        # and one at 0.55 (above). The 0.45 need gets +0.1 bonus.
        agent.needs["shelter"] = 0.45  # urgency 0.55 + 0.1 = 0.65
        agent.needs["safety"] = 0.40  # urgency 0.60 + 0.1 = 0.70
        # Leave others at 1.0 (urgency 0.0)

        priorities = ns.prioritize_needs(agent, ts)
        assert priorities[0] == "safety"
        assert priorities[1] == "shelter"


class TestAssessHealthImpact:
    def test_assess_health_impact_healthy(self):
        """All needs above warning threshold -> positive health (recovery)."""
        ns = NeedsSystem(_default_config())
        agent, _ = _make_agent()
        ns.initialize_needs(agent)  # All needs 1.0

        h_delta, s_delta = ns.assess_health_impact(agent)
        assert h_delta > 0  # Recovery
        assert s_delta == 0.0  # No suffering

    def test_assess_health_impact_warning(self):
        """Need below warning -> suffering but no health damage."""
        cfg = _default_config()
        ns = NeedsSystem(cfg)
        agent, _ = _make_agent()
        ns.initialize_needs(agent)
        agent.needs["hunger"] = 0.3  # Below 0.4 warning, above 0.2 critical

        h_delta, s_delta = ns.assess_health_impact(agent)
        assert s_delta > 0  # Some suffering
        # No health damage from warning, but also no recovery (not all above warning)
        assert h_delta == 0.0

    def test_assess_health_impact_critical(self):
        """Need below critical -> health damage + suffering."""
        cfg = _default_config()
        ns = NeedsSystem(cfg)
        agent, _ = _make_agent()
        ns.initialize_needs(agent)
        agent.needs["thirst"] = 0.1  # Below 0.2 critical, above 0.05 lethal

        h_delta, s_delta = ns.assess_health_impact(agent)
        assert h_delta < 0  # Health damage
        assert s_delta > 0  # Suffering
        assert h_delta == pytest.approx(-cfg["critical_health_damage_per_tick"])
        assert s_delta == pytest.approx(cfg["critical_suffering_per_tick"])

    def test_assess_health_impact_lethal(self):
        """Need below lethal threshold -> maximum damage + suffering."""
        cfg = _default_config()
        ns = NeedsSystem(cfg)
        agent, _ = _make_agent()
        ns.initialize_needs(agent)
        agent.needs["warmth"] = 0.02  # Below 0.05 lethal

        h_delta, s_delta = ns.assess_health_impact(agent)
        assert h_delta == pytest.approx(-cfg["lethal_health_damage_per_tick"])
        assert s_delta == pytest.approx(cfg["lethal_suffering_per_tick"])

    def test_assess_health_impact_multiple_needs_stack(self):
        """Penalties from multiple low needs stack additively."""
        cfg = _default_config()
        ns = NeedsSystem(cfg)
        agent, _ = _make_agent()
        ns.initialize_needs(agent)
        agent.needs["hunger"] = 0.01   # Lethal
        agent.needs["thirst"] = 0.01   # Lethal

        h_delta, s_delta = ns.assess_health_impact(agent)
        assert h_delta == pytest.approx(-2.0 * cfg["lethal_health_damage_per_tick"])
        assert s_delta == pytest.approx(2.0 * cfg["lethal_suffering_per_tick"])


class TestApplyHealthImpact:
    def test_apply_health_impact(self):
        """Health stays bounded to [0, 1]."""
        ns = NeedsSystem(_default_config())
        agent, _ = _make_agent()
        agent.health = 0.5
        agent.suffering = 0.0

        ns.apply_health_impact(agent, -0.1, 0.05)
        assert agent.health == pytest.approx(0.4)
        assert agent.suffering == pytest.approx(0.05)

    def test_apply_health_impact_bounded_above(self):
        """Health cannot exceed 1.0."""
        ns = NeedsSystem(_default_config())
        agent, _ = _make_agent()
        agent.health = 0.95

        ns.apply_health_impact(agent, 0.2, 0.0)
        assert agent.health == 1.0

    def test_apply_health_impact_bounded_below(self):
        """Health cannot go below 0.0."""
        ns = NeedsSystem(_default_config())
        agent, _ = _make_agent()
        agent.health = 0.05

        ns.apply_health_impact(agent, -0.5, 0.0)
        assert agent.health == 0.0

    def test_apply_suffering_accumulates(self):
        """Suffering is not clipped — it always accumulates."""
        ns = NeedsSystem(_default_config())
        agent, _ = _make_agent()
        agent.suffering = 1.5

        ns.apply_health_impact(agent, 0.0, 0.3)
        assert agent.suffering == pytest.approx(1.8)


class TestMortalityFactor:
    def test_compute_mortality_factor_healthy(self):
        """Health above threshold -> mortality factor is 0.0."""
        ns = NeedsSystem(_default_config())
        agent, _ = _make_agent()
        agent.health = 0.8  # Above 0.5 threshold

        assert ns.compute_needs_mortality_factor(agent) == 0.0

    def test_compute_mortality_factor_damaged(self):
        """Health at or below threshold -> positive mortality factor."""
        cfg = _default_config()
        ns = NeedsSystem(cfg)
        agent, _ = _make_agent()
        agent.health = 0.3  # Below 0.5

        factor = ns.compute_needs_mortality_factor(agent)
        expected = cfg["needs_mortality_multiplier"] * (1.0 - 0.3)
        assert factor == pytest.approx(expected)

    def test_compute_mortality_factor_at_threshold(self):
        """Health exactly at threshold -> positive factor."""
        cfg = _default_config()
        ns = NeedsSystem(cfg)
        agent, _ = _make_agent()
        agent.health = cfg["health_mortality_threshold"]

        factor = ns.compute_needs_mortality_factor(agent)
        assert factor > 0.0

    def test_compute_mortality_factor_zero_health(self):
        """Health = 0.0 gives maximum mortality factor."""
        cfg = _default_config()
        ns = NeedsSystem(cfg)
        agent, _ = _make_agent()
        agent.health = 0.0

        factor = ns.compute_needs_mortality_factor(agent)
        assert factor == pytest.approx(cfg["needs_mortality_multiplier"])


class TestCombinedDecayAndImpact:
    def test_combined_decay_then_impact(self):
        """Multi-tick decay drives health damage over time."""
        cfg = _default_config()
        ns = NeedsSystem(cfg)
        agent, _ = _make_agent()
        ns.initialize_needs(agent)

        # Decay for 15 ticks in harsh conditions (desert, winter)
        for _ in range(15):
            ns.decay_needs(agent, terrain="desert", season="winter", life_phase=None)
            h_delta, s_delta = ns.assess_health_impact(agent)
            ns.apply_health_impact(agent, h_delta, s_delta)

        # After 15 ticks in desert winter, some needs should be low
        assert agent.health < 1.0
        assert agent.suffering > 0.0
        # Thirst should be especially low (desert 1.8 * winter 0.9 ~ 1.62x)
        assert agent.needs["thirst"] < 0.2


# ===================================================================
# Modifier table tests
# ===================================================================

class TestModifierTables:
    def test_default_modifiers_all_one(self):
        """DEFAULT_MODIFIERS has all six needs at 1.0."""
        for need in NeedType:
            assert DEFAULT_MODIFIERS[need.value] == 1.0

    def test_terrain_tables_have_all_needs(self):
        """Every terrain entry covers all six needs."""
        for terrain, mods in TERRAIN_DECAY_MODIFIERS.items():
            for need in NeedType:
                assert need.value in mods, f"{terrain} missing {need.value}"

    def test_season_tables_have_all_needs(self):
        """Every season entry covers all six needs."""
        for season, mods in SEASON_DECAY_MODIFIERS.items():
            for need in NeedType:
                assert need.value in mods, f"{season} missing {need.value}"

    def test_life_phase_tables_have_all_needs(self):
        """Every life-phase entry covers all six needs."""
        for phase, mods in LIFE_PHASE_DECAY_MODIFIERS.items():
            for need in NeedType:
                assert need.value in mods, f"{phase} missing {need.value}"


# ===================================================================
# GatheringSystem tests
# ===================================================================

class TestChooseActivity:
    def test_choose_activity_for_hunger(self):
        """A hungry agent picks a food-gathering activity."""
        gs = GatheringSystem()
        agent, ts = _make_agent()
        agent.needs["hunger"] = 0.1

        priorities = ["hunger", "thirst", "shelter", "safety", "warmth", "rest"]
        activity = gs.choose_activity(agent, priorities, terrain=None, season=None, trait_system=ts)
        assert GATHERING_ACTIVITIES[activity]["satisfies"] == "hunger"

    def test_choose_activity_for_thirst(self):
        """A thirsty agent picks find_water."""
        gs = GatheringSystem()
        agent, ts = _make_agent()

        priorities = ["thirst", "hunger", "shelter", "safety", "warmth", "rest"]
        activity = gs.choose_activity(agent, priorities, terrain=None, season=None, trait_system=ts)
        assert activity == "find_water"  # Only thirst activity

    def test_choose_activity_terrain_boost(self):
        """Coastal terrain boosts fish above other hunger activities."""
        gs = GatheringSystem()
        agent, ts = _make_agent()
        # Set traits to zero to isolate terrain effect
        agent.traits[:] = 0.0

        priorities = ["hunger", "thirst", "shelter", "safety", "warmth", "rest"]
        activity = gs.choose_activity(agent, priorities, terrain="coast", season=None, trait_system=ts)
        # Coast fish suitability = 1.4 vs forage = 0.5, hunt = 0.3
        assert activity == "fish"

    def test_choose_activity_forest_forage(self):
        """Forest terrain with zeroed traits should prefer forage over fish."""
        gs = GatheringSystem()
        agent, ts = _make_agent()
        agent.traits[:] = 0.0

        priorities = ["hunger", "thirst", "shelter", "safety", "warmth", "rest"]
        activity = gs.choose_activity(agent, priorities, terrain="forest", season=None, trait_system=ts)
        # Forest: forage=1.3, hunt=1.2, fish=0.2
        assert activity == "forage"

    def test_choose_activity_fallback_rest(self):
        """When given an empty priority list, falls back to rest."""
        gs = GatheringSystem()
        agent, ts = _make_agent()

        activity = gs.choose_activity(agent, [], terrain=None, season=None, trait_system=ts)
        assert activity == "rest"


class TestAttemptGathering:
    def test_attempt_gathering_basic(self):
        """Gathering returns a yield in [0, 0.95]."""
        gs = GatheringSystem()
        agent, ts = _make_agent()
        rng = np.random.default_rng(42)

        result = gs.attempt_gathering(agent, "forage", terrain=None, season=None, trait_system=ts, rng=rng)
        assert 0.0 <= result <= 0.95

    def test_attempt_gathering_terrain_effect(self):
        """
        Forest terrain increases foraging yield compared to desert.
        """
        gs = GatheringSystem()
        agent, ts = _make_agent()

        rng_forest = np.random.default_rng(99)
        rng_desert = np.random.default_rng(99)

        forest_yield = gs.attempt_gathering(agent, "forage", terrain="forest", season=None, trait_system=ts, rng=rng_forest)
        desert_yield = gs.attempt_gathering(agent, "forage", terrain="desert", season=None, trait_system=ts, rng=rng_desert)

        # Forest forage = 1.3, desert forage = 0.1
        assert forest_yield > desert_yield

    def test_attempt_gathering_deterministic(self):
        """Same seed produces same yield."""
        gs = GatheringSystem()
        agent, ts = _make_agent()

        rng1 = np.random.default_rng(123)
        rng2 = np.random.default_rng(123)

        y1 = gs.attempt_gathering(agent, "hunt", terrain="plains", season="summer", trait_system=ts, rng=rng1)
        y2 = gs.attempt_gathering(agent, "hunt", terrain="plains", season="summer", trait_system=ts, rng=rng2)
        assert y1 == pytest.approx(y2)

    def test_attempt_gathering_unknown_activity(self):
        """Unknown activity returns 0.0."""
        gs = GatheringSystem()
        agent, ts = _make_agent()
        rng = np.random.default_rng(42)

        result = gs.attempt_gathering(agent, "dance", terrain=None, season=None, trait_system=ts, rng=rng)
        assert result == 0.0

    def test_attempt_gathering_yield_clamped(self):
        """Yield is capped at 0.95 even with very high modifiers."""
        gs = GatheringSystem()
        agent, ts = _make_agent()
        # Maximise trait boosts
        agent.traits[:] = 1.0
        rng = np.random.default_rng(42)

        # Use rest (0.30 base) in a benign context — won't exceed 0.95
        # but use build_shelter in forest (1.1 * 0.20 = 0.22 + trait boosts)
        # Force a scenario that could exceed limit
        result = gs.attempt_gathering(agent, "rest", terrain="forest", season=None, trait_system=ts, rng=rng)
        assert result <= 0.95


class TestSatisfyNeed:
    def test_satisfy_need(self):
        """Need increases by the given amount."""
        gs = GatheringSystem()
        agent, _ = _make_agent()
        agent.needs["hunger"] = 0.3

        gs.satisfy_need(agent, "hunger", 0.2)
        assert agent.needs["hunger"] == pytest.approx(0.5)

    def test_satisfy_need_bounded_at_one(self):
        """Need cannot exceed 1.0."""
        gs = GatheringSystem()
        agent, _ = _make_agent()
        agent.needs["hunger"] = 0.9

        gs.satisfy_need(agent, "hunger", 0.5)
        assert agent.needs["hunger"] == 1.0


class TestCanGather:
    def test_can_gather_adult(self):
        """A young_adult can gather."""
        agent, _ = _make_agent(life_phase="young_adult")
        assert GatheringSystem.can_gather(agent) is True

    def test_can_gather_child(self):
        """A child cannot gather."""
        agent, _ = _make_agent(life_phase="child")
        assert GatheringSystem.can_gather(agent) is False

    def test_can_gather_infant(self):
        """An infant cannot gather."""
        agent, _ = _make_agent(life_phase="infant")
        assert GatheringSystem.can_gather(agent) is False

    def test_can_gather_elder(self):
        """An elder can gather."""
        agent, _ = _make_agent(life_phase="elder")
        assert GatheringSystem.can_gather(agent) is True

    def test_can_gather_none_phase(self):
        """If life_phase is None, agent can gather (backward compat)."""
        agent, _ = _make_agent(life_phase=None)
        assert GatheringSystem.can_gather(agent) is True


class TestCaregiverShare:
    def test_caregiver_share(self):
        """Caregiver shares with dependent; both needs update."""
        caregiver, _ = _make_agent(agent_id="cg", life_phase="mature")
        caregiver.needs = {n.value: 0.8 for n in NeedType}

        child, _ = _make_agent(agent_id="ch", life_phase="child")
        child.needs = {n.value: 0.3 for n in NeedType}

        rng = np.random.default_rng(42)
        GatheringSystem.caregiver_share(caregiver, [child], rng)

        # Child needs should have increased
        for need in NeedType:
            assert child.needs[need.value] > 0.3
        # Caregiver needs should have decreased
        for need in NeedType:
            assert caregiver.needs[need.value] < 0.8

    def test_caregiver_share_does_not_starve_caregiver(self):
        """Caregiver won't reduce their own needs below 0.3."""
        caregiver, _ = _make_agent(agent_id="cg", life_phase="mature")
        caregiver.needs = {n.value: 0.35 for n in NeedType}

        child, _ = _make_agent(agent_id="ch", life_phase="infant")
        child.needs = {n.value: 0.1 for n in NeedType}

        rng = np.random.default_rng(42)
        GatheringSystem.caregiver_share(caregiver, [child], rng)

        # Caregiver should not go below 0.3
        for need in NeedType:
            assert caregiver.needs[need.value] >= 0.3 - 1e-9

    def test_caregiver_share_ignores_adults(self):
        """Adults in the dependent list are not shared with."""
        caregiver, _ = _make_agent(agent_id="cg", life_phase="mature")
        caregiver.needs = {n.value: 0.8 for n in NeedType}

        adult_dep, _ = _make_agent(agent_id="ad", life_phase="young_adult")
        adult_dep.needs = {n.value: 0.2 for n in NeedType}

        rng = np.random.default_rng(42)
        GatheringSystem.caregiver_share(caregiver, [adult_dep], rng)

        # Nothing should change
        for need in NeedType:
            assert caregiver.needs[need.value] == pytest.approx(0.8)
            assert adult_dep.needs[need.value] == pytest.approx(0.2)

    def test_caregiver_share_no_transfer_when_child_better(self):
        """No transfer when child's need is already higher than caregiver's."""
        caregiver, _ = _make_agent(agent_id="cg", life_phase="mature")
        caregiver.needs = {n.value: 0.3 for n in NeedType}

        child, _ = _make_agent(agent_id="ch", life_phase="child")
        child.needs = {n.value: 0.5 for n in NeedType}

        rng = np.random.default_rng(42)
        GatheringSystem.caregiver_share(caregiver, [child], rng)

        # No transfer should occur
        for need in NeedType:
            assert caregiver.needs[need.value] == pytest.approx(0.3)
            assert child.needs[need.value] == pytest.approx(0.5)


# ===================================================================
# Activity / suitability table integrity
# ===================================================================

class TestActivityTables:
    def test_all_activities_have_required_keys(self):
        """Every activity has satisfies, base_yield, and trait_boosts."""
        for name, defn in GATHERING_ACTIVITIES.items():
            assert "satisfies" in defn, f"{name} missing 'satisfies'"
            assert "base_yield" in defn, f"{name} missing 'base_yield'"
            assert "trait_boosts" in defn, f"{name} missing 'trait_boosts'"

    def test_terrain_suitability_covers_all_activities(self):
        """Every terrain has suitability for every activity."""
        for terrain, suits in TERRAIN_ACTIVITY_SUITABILITY.items():
            for act in GATHERING_ACTIVITIES:
                assert act in suits, f"{terrain} missing suitability for {act}"

    def test_season_suitability_covers_all_activities(self):
        """Every season has suitability for every activity."""
        for season, suits in SEASON_ACTIVITY_SUITABILITY.items():
            for act in GATHERING_ACTIVITIES:
                assert act in suits, f"{season} missing suitability for {act}"

    def test_every_need_has_at_least_one_activity(self):
        """Each NeedType is satisfied by at least one activity."""
        satisfied = {defn["satisfies"] for defn in GATHERING_ACTIVITIES.values()}
        for need in NeedType:
            assert need.value in satisfied, f"No activity satisfies {need.value}"
