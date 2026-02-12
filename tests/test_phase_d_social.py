"""
Tests for Phase D: Rich Social Systems — Marriage, Clans, Institutions.

Tests MarriageManager, ClanManager, InstitutionManager, SocialDynamicsExtension
integration with Phase D features, and the social API endpoints for Phase D data.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from seldon.core.agent import Agent
from seldon.core.config import ExperimentConfig
from seldon.core.processing import ProcessingRegion
from seldon.extensions.social_dynamics import SocialDynamicsExtension
from seldon.social.clans import ClanManager
from seldon.social.institutions import InstitutionManager
from seldon.social.marriage import MarriageManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(**kwargs) -> ExperimentConfig:
    defaults = dict(random_seed=42, initial_population=20, generations_to_run=5)
    defaults.update(kwargs)
    return ExperimentConfig(**defaults)


def _make_agent(config: ExperimentConfig | None = None, **kwargs) -> Agent:
    if config is None:
        config = _make_config()
    ts = config.trait_system
    rng = np.random.default_rng(kwargs.pop("seed", 42))
    traits = kwargs.pop("traits", None)
    if traits is None:
        traits = ts.random_traits(rng)
    defaults = dict(
        id=f"agent_{rng.integers(10000):04d}",
        name="Test Agent",
        age=25,
        generation=0,
        birth_order=1,
        traits=traits,
        traits_at_birth=traits.copy(),
    )
    defaults.update(kwargs)
    return Agent(**defaults)


def _make_population(config: ExperimentConfig, n: int = 20) -> list[Agent]:
    rng = np.random.default_rng(config.random_seed)
    ts = config.trait_system
    population = []
    for i in range(n):
        traits = ts.random_traits(rng)
        age = int(rng.integers(5, 60))
        agent = Agent(
            id=f"pop_{i:03d}",
            name=f"Agent {i}",
            age=age,
            generation=0,
            birth_order=int(rng.integers(1, 5)),
            traits=traits,
            traits_at_birth=traits.copy(),
            processing_region=rng.choice([
                ProcessingRegion.OPTIMAL,
                ProcessingRegion.DEEP,
                ProcessingRegion.UNDER_PROCESSING,
            ]),
            contribution_history=[float(rng.uniform(0.0, 2.0)) for _ in range(5)],
        )
        population.append(agent)
    return population


# ===========================================================================
# MarriageManager Tests
# ===========================================================================

class TestMarriageManager:
    """Tests for marriage formalization, divorce, and property sharing."""

    def test_courtship_tracking(self):
        """Newly paired agents get courtship_start_gen set."""
        config = _make_config()
        mm = MarriageManager(config)
        rng = np.random.default_rng(42)

        pop = _make_population(config, 4)
        pop[0].partner_id = pop[1].id
        pop[1].partner_id = pop[0].id

        mm.process_marriages(pop, generation=0, rng=rng, config=config)

        assert pop[0].extension_data.get("courtship_start_gen") is not None or \
               pop[0].extension_data.get("marriage") is not None

    def test_formalization_after_delay(self):
        """Marriage is formalized after courtship delay."""
        config = _make_config()
        config.marriage_config["formalization_delay_generations"] = 2
        mm = MarriageManager(config)
        rng = np.random.default_rng(42)

        pop = _make_population(config, 4)
        pop[0].partner_id = pop[1].id
        pop[1].partner_id = pop[0].id

        # Gen 0: courtship starts
        mm.process_marriages(pop, generation=0, rng=rng, config=config)
        assert pop[0].extension_data.get("marriage") is None

        # Gen 1: still courting
        mm.process_marriages(pop, generation=1, rng=rng, config=config)
        assert pop[0].extension_data.get("marriage") is None

        # Gen 2: formalized
        mm.process_marriages(pop, generation=2, rng=rng, config=config)
        assert pop[0].extension_data.get("marriage") is not None
        assert pop[0].relationship_status == "married"
        assert pop[1].relationship_status == "married"

    def test_immediate_when_delay_zero(self):
        """With delay=0, marriage happens immediately."""
        config = _make_config()
        config.marriage_config["formalization_delay_generations"] = 0
        mm = MarriageManager(config)
        rng = np.random.default_rng(42)

        pop = _make_population(config, 4)
        pop[0].partner_id = pop[1].id
        pop[1].partner_id = pop[0].id

        mm.process_marriages(pop, generation=0, rng=rng, config=config)
        assert pop[0].extension_data.get("marriage") is not None
        assert pop[0].relationship_status == "married"

    def test_divorce_conditions(self):
        """Divorce can happen based on personality."""
        config = _make_config()
        config.marriage_config["divorce_base_rate"] = 1.0  # Very high to guarantee
        mm = MarriageManager(config)
        rng = np.random.default_rng(42)

        pop = _make_population(config, 4)
        ts = config.trait_system

        # Set low agreeableness/conscientiousness to maximize divorce prob
        try:
            agree_idx = ts.trait_index("agreeableness")
            cons_idx = ts.trait_index("conscientiousness")
            pop[0].traits[agree_idx] = 0.0
            pop[0].traits[cons_idx] = 0.0
            pop[1].traits[agree_idx] = 0.0
            pop[1].traits[cons_idx] = 0.0
        except KeyError:
            pass

        # Create married pair
        contract = {
            "partner1_id": pop[0].id,
            "partner2_id": pop[1].id,
            "generation_formed": 0,
            "shared_wealth": 10.0,
            "is_political": False,
        }
        pop[0].extension_data["marriage"] = contract
        pop[1].extension_data["marriage"] = contract
        pop[0].partner_id = pop[1].id
        pop[1].partner_id = pop[0].id

        result = mm.process_marriages(pop, generation=5, rng=rng, config=config)
        assert result["divorces"] >= 1

    def test_property_split_on_divorce(self):
        """Shared wealth is split evenly on divorce."""
        config = _make_config()
        config.marriage_config["divorce_base_rate"] = 1.0
        mm = MarriageManager(config)
        rng = np.random.default_rng(42)

        pop = _make_population(config, 4)
        ts = config.trait_system

        # Force low stability
        try:
            agree_idx = ts.trait_index("agreeableness")
            cons_idx = ts.trait_index("conscientiousness")
            pop[0].traits[agree_idx] = 0.0
            pop[0].traits[cons_idx] = 0.0
            pop[1].traits[agree_idx] = 0.0
            pop[1].traits[cons_idx] = 0.0
        except KeyError:
            pass

        pop[0].wealth = 0.0
        pop[1].wealth = 0.0
        contract = {
            "partner1_id": pop[0].id,
            "partner2_id": pop[1].id,
            "generation_formed": 0,
            "shared_wealth": 20.0,
            "is_political": False,
        }
        pop[0].extension_data["marriage"] = contract
        pop[1].extension_data["marriage"] = contract
        pop[0].partner_id = pop[1].id
        pop[1].partner_id = pop[0].id

        mm.process_marriages(pop, generation=5, rng=rng, config=config)

        # If divorced, wealth should be split
        if pop[0].extension_data.get("marriage") is None:
            assert pop[0].wealth == pytest.approx(10.0)
            assert pop[1].wealth == pytest.approx(10.0)

    def test_widowed_clears_contract(self):
        """Dead partner causes marriage contract to be cleared."""
        config = _make_config()
        mm = MarriageManager(config)
        rng = np.random.default_rng(42)

        pop = _make_population(config, 4)
        contract = {
            "partner1_id": pop[0].id,
            "partner2_id": pop[1].id,
            "generation_formed": 0,
            "shared_wealth": 5.0,
            "is_political": False,
        }
        pop[0].extension_data["marriage"] = contract
        pop[1].is_alive = False

        mm.process_marriages(pop, generation=3, rng=rng, config=config)
        assert pop[0].extension_data.get("marriage") is None

    def test_political_marriage_detection(self):
        """Cross-clan marriages are detected as political."""
        config = _make_config()
        config.marriage_config["formalization_delay_generations"] = 0
        mm = MarriageManager(config)
        rng = np.random.default_rng(42)

        pop = _make_population(config, 4)
        pop[0].partner_id = pop[1].id
        pop[1].partner_id = pop[0].id
        pop[0].extension_data["clan_id"] = "clan_A"
        pop[1].extension_data["clan_id"] = "clan_B"

        mm.process_marriages(pop, generation=0, rng=rng, config=config)

        marriage = pop[0].extension_data.get("marriage")
        assert marriage is not None
        assert marriage["is_political"] is True

    def test_disabled_no_effect(self):
        """When disabled, returns disabled flag and does nothing."""
        config = _make_config()
        config.marriage_config["enabled"] = False
        mm = MarriageManager(config)
        rng = np.random.default_rng(42)

        pop = _make_population(config, 4)
        pop[0].partner_id = pop[1].id
        pop[1].partner_id = pop[0].id

        result = mm.process_marriages(pop, generation=0, rng=rng, config=config)
        assert result.get("marriages_enabled") is False
        assert pop[0].extension_data.get("marriage") is None


# ===========================================================================
# ClanManager Tests
# ===========================================================================

class TestClanManager:
    """Tests for clan formation, membership, honor, and rivalries."""

    def _setup_clan_population(self, config):
        """Create a population with a clear founder and descendants."""
        pop = _make_population(config, 15)

        # Make pop[0] a high-status founder
        pop[0].social_status = 0.8
        pop[0].age = 50

        # Create lineage: pop[0] → children pop[1..4]
        for i in range(1, 5):
            pop[i].parent1_id = pop[0].id
            pop[i].social_status = 0.5

        return pop

    def test_detect_from_lineage(self):
        """Clans form from high-status agents with descendants."""
        config = _make_config()
        cm = ClanManager(config)
        rng = np.random.default_rng(42)
        pop = self._setup_clan_population(config)

        metrics = cm.update_clans(pop, generation=5, rng=rng, config=config)
        assert metrics["clan_count"] >= 1

    def test_min_members_threshold(self):
        """Clan requires minimum living members."""
        config = _make_config()
        config.clan_config["min_living_members"] = 10  # Very high
        cm = ClanManager(config)
        rng = np.random.default_rng(42)

        pop = _make_population(config, 5)
        pop[0].social_status = 0.9
        # Only 1 descendant — not enough
        pop[1].parent1_id = pop[0].id

        metrics = cm.update_clans(pop, generation=5, rng=rng, config=config)
        assert metrics["clan_count"] == 0

    def test_founder_min_status(self):
        """Only agents above founder_min_status can found clans."""
        config = _make_config()
        config.clan_config["founder_min_status"] = 0.99
        cm = ClanManager(config)
        rng = np.random.default_rng(42)
        pop = self._setup_clan_population(config)
        pop[0].social_status = 0.5  # Below threshold

        metrics = cm.update_clans(pop, generation=5, rng=rng, config=config)
        assert metrics["clan_count"] == 0

    def test_child_inherits_clan(self):
        """Newborns are assigned to their parent's clan."""
        config = _make_config()
        cm = ClanManager(config)
        rng = np.random.default_rng(42)
        pop = self._setup_clan_population(config)

        cm.update_clans(pop, generation=5, rng=rng, config=config)

        # Now create a child from a clan member
        parent1 = pop[1]  # Should be in a clan
        parent2 = pop[5]
        child = _make_agent(config, id="child_new", seed=99)

        cm.assign_to_child(child, (parent1, parent2), config)

        if parent1.extension_data.get("clan_id") is not None:
            assert child.extension_data.get("clan_id") is not None

    def test_honor_computation(self):
        """Clan honor is the mean status of living members."""
        config = _make_config()
        cm = ClanManager(config)
        rng = np.random.default_rng(42)
        pop = self._setup_clan_population(config)

        cm.update_clans(pop, generation=5, rng=rng, config=config)

        for clan in cm.clans.values():
            assert 0.0 <= clan.honor <= 1.0

    def test_honor_bonus_in_extension_data(self):
        """Members get clan_honor_bonus in extension_data."""
        config = _make_config()
        cm = ClanManager(config)
        rng = np.random.default_rng(42)
        pop = self._setup_clan_population(config)

        cm.update_clans(pop, generation=5, rng=rng, config=config)

        # Find a clan member
        for agent in pop:
            if agent.extension_data.get("clan_id") is not None:
                assert "clan_honor_bonus" in agent.extension_data
                assert agent.extension_data["clan_honor_bonus"] >= 0.0
                break

    def test_rivalry_detection(self):
        """Clans in the same community with honor gap become rivals."""
        config = _make_config()
        config.clan_config["rival_threshold"] = 0.1
        config.clan_config["min_living_members"] = 2
        cm = ClanManager(config)
        rng = np.random.default_rng(42)

        pop = _make_population(config, 20)

        # Two founders with different statuses in same community
        pop[0].social_status = 0.9
        pop[0].age = 50
        pop[1].parent1_id = pop[0].id
        pop[2].parent1_id = pop[0].id
        pop[0].community_id = "comm_1"
        pop[1].community_id = "comm_1"
        pop[2].community_id = "comm_1"
        pop[1].social_status = 0.7
        pop[2].social_status = 0.7

        pop[5].social_status = 0.8
        pop[5].age = 50
        pop[6].parent1_id = pop[5].id
        pop[7].parent1_id = pop[5].id
        pop[5].community_id = "comm_1"
        pop[6].community_id = "comm_1"
        pop[7].community_id = "comm_1"
        pop[6].social_status = 0.3
        pop[7].social_status = 0.3

        cm.update_clans(pop, generation=5, rng=rng, config=config)

        if len(cm.clans) >= 2:
            has_rival = any(len(c.rival_clan_ids) > 0 for c in cm.clans.values())
            assert has_rival

    def test_multiple_clans(self):
        """Multiple clans can form from different founders."""
        config = _make_config()
        config.clan_config["min_living_members"] = 2
        cm = ClanManager(config)
        rng = np.random.default_rng(42)

        pop = _make_population(config, 20)

        # Founder 1
        pop[0].social_status = 0.9
        pop[0].age = 50
        pop[1].parent1_id = pop[0].id
        pop[2].parent1_id = pop[0].id

        # Founder 2
        pop[5].social_status = 0.8
        pop[5].age = 50
        pop[6].parent1_id = pop[5].id
        pop[7].parent1_id = pop[5].id

        metrics = cm.update_clans(pop, generation=5, rng=rng, config=config)
        assert metrics["clan_count"] >= 2

    def test_dissolution_below_min(self):
        """Clans below minimum membership are dissolved."""
        config = _make_config()
        config.clan_config["min_living_members"] = 3
        cm = ClanManager(config)
        rng = np.random.default_rng(42)

        pop = self._setup_clan_population(config)
        cm.update_clans(pop, generation=5, rng=rng, config=config)
        initial_count = len(cm.clans)

        # Kill most clan members
        for i in range(1, 5):
            pop[i].is_alive = False

        cm.update_clans(pop, generation=6, rng=rng, config=config)
        # Should have fewer clans or zero
        assert len(cm.clans) <= initial_count

    def test_disabled_no_effect(self):
        """When disabled, returns disabled flag."""
        config = _make_config()
        config.clan_config["enabled"] = False
        cm = ClanManager(config)
        rng = np.random.default_rng(42)

        pop = self._setup_clan_population(config)
        metrics = cm.update_clans(pop, generation=5, rng=rng, config=config)
        assert metrics.get("clans_enabled") is False


# ===========================================================================
# InstitutionManager Tests
# ===========================================================================

class TestInstitutionManager:
    """Tests for council and guild formation, elections, and prestige."""

    def test_council_formation(self):
        """Councils form in communities with enough elders."""
        config = _make_config()
        config.institutions_config["council_min_elders"] = 3
        config.institutions_config["council_elder_min_age"] = 40
        config.institutions_config["council_min_community_size"] = 5
        im = InstitutionManager(config)
        rng = np.random.default_rng(42)

        pop = _make_population(config, 20)
        # Put everyone in the same community with enough elders
        for i, a in enumerate(pop):
            a.community_id = "comm_1"
            a.influence_score = 0.5
        for i in range(5):
            pop[i].age = 50  # elders

        metrics = im.update_institutions(pop, generation=5, rng=rng, config=config)
        assert metrics["council_count"] >= 1

    def test_elder_age_requirement(self):
        """No council if elders too young."""
        config = _make_config()
        config.institutions_config["council_elder_min_age"] = 80  # Very old
        im = InstitutionManager(config)
        rng = np.random.default_rng(42)

        pop = _make_population(config, 20)
        for a in pop:
            a.community_id = "comm_1"
            a.age = 40

        metrics = im.update_institutions(pop, generation=5, rng=rng, config=config)
        assert metrics["council_count"] == 0

    def test_guild_from_occupation(self):
        """Guilds form from occupation groups."""
        config = _make_config()
        config.institutions_config["guild_min_members"] = 3
        config.institutions_config["guild_min_skill"] = 0.3
        im = InstitutionManager(config)
        rng = np.random.default_rng(42)

        pop = _make_population(config, 10)
        for i in range(5):
            pop[i].occupation = "farmer"
            pop[i].skills = {"farmer": 0.5}
            pop[i].influence_score = 0.4

        metrics = im.update_institutions(pop, generation=5, rng=rng, config=config)
        assert metrics["guild_count"] >= 1

    def test_guild_skill_threshold(self):
        """No guild if members don't have sufficient skill."""
        config = _make_config()
        config.institutions_config["guild_min_skill"] = 0.9  # Very high
        im = InstitutionManager(config)
        rng = np.random.default_rng(42)

        pop = _make_population(config, 10)
        for i in range(5):
            pop[i].occupation = "farmer"
            pop[i].skills = {"farmer": 0.2}  # Below threshold

        metrics = im.update_institutions(pop, generation=5, rng=rng, config=config)
        assert metrics["guild_count"] == 0

    def test_leader_election(self):
        """Leader elected by highest influence."""
        config = _make_config()
        config.institutions_config["election_frequency_generations"] = 1
        im = InstitutionManager(config)
        rng = np.random.default_rng(42)

        pop = _make_population(config, 20)
        for a in pop:
            a.community_id = "comm_1"
            a.influence_score = 0.3
        for i in range(5):
            pop[i].age = 50
            pop[i].influence_score = 0.3

        # Make pop[0] clearly the highest influence
        pop[0].influence_score = 0.99

        im.update_institutions(pop, generation=5, rng=rng, config=config)

        for inst in im.institutions.values():
            if inst.type == "council":
                assert inst.leader_id == pop[0].id

    def test_prestige_computation(self):
        """Prestige is mean influence of members."""
        config = _make_config()
        im = InstitutionManager(config)
        rng = np.random.default_rng(42)

        pop = _make_population(config, 20)
        for a in pop:
            a.community_id = "comm_1"
            a.influence_score = 0.5
        for i in range(5):
            pop[i].age = 50

        im.update_institutions(pop, generation=5, rng=rng, config=config)

        for inst in im.institutions.values():
            assert inst.prestige >= 0.0

    def test_prestige_bonus_in_extension_data(self):
        """Members get institution_prestige_bonus in extension_data."""
        config = _make_config()
        im = InstitutionManager(config)
        rng = np.random.default_rng(42)

        pop = _make_population(config, 20)
        for a in pop:
            a.community_id = "comm_1"
            a.influence_score = 0.5
        for i in range(5):
            pop[i].age = 50

        im.update_institutions(pop, generation=5, rng=rng, config=config)

        found_bonus = False
        for agent in pop:
            if agent.extension_data.get("institution_prestige_bonus") is not None:
                assert agent.extension_data["institution_prestige_bonus"] >= 0.0
                found_bonus = True
                break

        if im.institutions:
            assert found_bonus

    def test_multiple_institutions(self):
        """Multiple institutions can coexist."""
        config = _make_config()
        config.institutions_config["guild_min_members"] = 3
        config.institutions_config["guild_min_skill"] = 0.3
        im = InstitutionManager(config)
        rng = np.random.default_rng(42)

        pop = _make_population(config, 20)
        # Council
        for a in pop:
            a.community_id = "comm_1"
            a.influence_score = 0.5
        for i in range(5):
            pop[i].age = 50

        # Guild
        for i in range(10, 15):
            pop[i].occupation = "artisan"
            pop[i].skills = {"artisan": 0.5}

        metrics = im.update_institutions(pop, generation=5, rng=rng, config=config)
        total = metrics["council_count"] + metrics["guild_count"]
        assert total >= 2

    def test_dissolution(self):
        """Institutions dissolve when membership drops to zero."""
        config = _make_config()
        im = InstitutionManager(config)
        rng = np.random.default_rng(42)

        pop = _make_population(config, 20)
        for a in pop:
            a.community_id = "comm_1"
            a.influence_score = 0.5
        for i in range(5):
            pop[i].age = 50

        im.update_institutions(pop, generation=5, rng=rng, config=config)
        initial_count = len(im.institutions)

        # Remove the community
        for a in pop:
            a.community_id = None
            a.age = 10  # No longer elders

        im.update_institutions(pop, generation=6, rng=rng, config=config)
        assert len(im.institutions) < initial_count

    def test_disabled_no_effect(self):
        """When disabled, returns disabled flag."""
        config = _make_config()
        config.institutions_config["enabled"] = False
        im = InstitutionManager(config)
        rng = np.random.default_rng(42)

        pop = _make_population(config, 20)
        metrics = im.update_institutions(pop, generation=5, rng=rng, config=config)
        assert metrics.get("institutions_enabled") is False


# ===========================================================================
# SocialDynamicsExtension Integration Tests (Phase D)
# ===========================================================================

class TestSocialDynamicsPhaseD:
    """Integration tests for Phase D features wired into SocialDynamicsExtension."""

    def test_extension_with_marriage(self):
        """Extension processes marriages in on_generation_end."""
        config = _make_config()
        ext = SocialDynamicsExtension()
        pop = _make_population(config, 10)

        # Create a pair
        pop[0].partner_id = pop[1].id
        pop[1].partner_id = pop[0].id
        config.marriage_config["formalization_delay_generations"] = 0

        ext.on_simulation_start(pop, config)
        ext.on_generation_end(0, pop, config)

        metrics = ext.get_metrics(pop)
        assert "new_marriages" in metrics

    def test_extension_with_clans(self):
        """Extension updates clans in on_generation_start."""
        config = _make_config()
        ext = SocialDynamicsExtension()
        pop = _make_population(config, 15)

        # Create a founder with descendants
        pop[0].social_status = 0.9
        pop[0].age = 50
        for i in range(1, 5):
            pop[i].parent1_id = pop[0].id

        ext.on_simulation_start(pop, config)
        ext.on_generation_start(1, pop, config)

        metrics = ext.get_metrics(pop)
        assert "clan_count" in metrics

    def test_extension_with_institutions(self):
        """Extension updates institutions in on_generation_start."""
        config = _make_config()
        ext = SocialDynamicsExtension()
        pop = _make_population(config, 20)

        for a in pop:
            a.community_id = "comm_1"
            a.influence_score = 0.5
        for i in range(5):
            pop[i].age = 50

        ext.on_simulation_start(pop, config)
        ext.on_generation_start(1, pop, config)

        metrics = ext.get_metrics(pop)
        assert "council_count" in metrics

    def test_all_three_together(self):
        """All Phase D features run together without errors."""
        config = _make_config()
        config.marriage_config["formalization_delay_generations"] = 0
        ext = SocialDynamicsExtension()
        pop = _make_population(config, 25)

        # Founder with descendants (for clans)
        pop[0].social_status = 0.9
        pop[0].age = 50
        for i in range(1, 5):
            pop[i].parent1_id = pop[0].id

        # Community (for institutions)
        for a in pop:
            a.community_id = "comm_1"
            a.influence_score = 0.5

        # Pair (for marriage)
        pop[5].partner_id = pop[6].id
        pop[6].partner_id = pop[5].id

        ext.on_simulation_start(pop, config)

        for gen in range(3):
            ext.on_generation_start(gen, pop, config)
            ext.on_generation_end(gen, pop, config)

        metrics = ext.get_metrics(pop)
        assert "clan_count" in metrics
        assert "council_count" in metrics
        assert "new_marriages" in metrics

    def test_backward_compat_no_phase_d_features(self):
        """Phase D disabled configs produce zero behavioral change."""
        config = _make_config()
        config.marriage_config["enabled"] = False
        config.clan_config["enabled"] = False
        config.institutions_config["enabled"] = False
        ext = SocialDynamicsExtension()
        pop = _make_population(config, 15)

        ext.on_simulation_start(pop, config)
        ext.on_generation_start(0, pop, config)
        ext.on_generation_end(0, pop, config)

        metrics = ext.get_metrics(pop)
        # Marriage/clan/institution metrics should show disabled
        assert metrics.get("marriages_enabled") is False
        assert metrics.get("clans_enabled") is False
        assert metrics.get("institutions_enabled") is False

    def test_metrics_include_phase_d(self):
        """get_metrics includes Phase D data alongside existing data."""
        config = _make_config()
        ext = SocialDynamicsExtension()
        pop = _make_population(config, 15)

        ext.on_simulation_start(pop, config)
        ext.on_generation_start(0, pop, config)
        ext.on_generation_end(0, pop, config)

        metrics = ext.get_metrics(pop)
        # Should have both existing and Phase D metrics
        assert "role_counts" in metrics  # existing
        assert "active_mentorships" in metrics  # existing
        # Phase D
        assert "clan_count" in metrics or "clans_enabled" in metrics
        assert "new_marriages" in metrics or "marriages_enabled" in metrics

    def test_modify_decision_clan_stay_bonus(self):
        """Clan members get stay bonus in decisions."""
        config = _make_config()
        ext = SocialDynamicsExtension()

        agent = _make_agent(config)
        agent.social_role = "unassigned"
        agent.extension_data["clan_id"] = "clan_X"

        utilities = {"stay": 0.5, "migrate": 0.3}
        result = ext.modify_decision(agent, "migration", utilities, config)
        assert result["stay"] > 0.5

    def test_modify_decision_council_stay_bonus(self):
        """Council members get stay bonus in decisions."""
        config = _make_config()
        ext = SocialDynamicsExtension()

        agent = _make_agent(config)
        agent.social_role = "unassigned"
        agent.extension_data["institution_ids"] = ["council_comm_1"]

        utilities = {"stay": 0.5, "migrate": 0.3}
        result = ext.modify_decision(agent, "migration", utilities, config)
        assert result["stay"] > 0.5

    def test_modify_attraction_same_institution(self):
        """Agents in the same institution get attraction bonus."""
        config = _make_config()
        ext = SocialDynamicsExtension()

        a1 = _make_agent(config, id="a1", seed=1)
        a2 = _make_agent(config, id="a2", seed=2)
        a1.social_role = "unassigned"
        a2.social_role = "unassigned"
        a1.extension_data["institution_ids"] = ["guild_farmer"]
        a2.extension_data["institution_ids"] = ["guild_farmer"]

        score = ext.modify_attraction(a1, a2, 0.5, config)
        assert score > 0.5

    def test_on_agent_created_assigns_clan(self):
        """Child is assigned to parent's clan via on_agent_created."""
        config = _make_config()
        ext = SocialDynamicsExtension()
        pop = _make_population(config, 15)

        pop[0].social_status = 0.9
        pop[0].age = 50
        for i in range(1, 5):
            pop[i].parent1_id = pop[0].id

        ext.on_simulation_start(pop, config)
        ext.on_generation_start(1, pop, config)

        # Create a child from a clan member
        parent1 = pop[1]
        parent2 = pop[5]
        child = _make_agent(config, id="new_child", seed=99)

        ext.on_agent_created(child, (parent1, parent2), config)

        if parent1.extension_data.get("clan_id") is not None:
            assert child.extension_data.get("clan_id") is not None


# ===========================================================================
# API Router Tests (Phase D endpoints)
# ===========================================================================

class TestSocialAPIPhaseDRouter:
    """Tests for the Phase D social API endpoints."""

    @pytest.fixture
    def client(self):
        try:
            from fastapi.testclient import TestClient
            from seldon.api.app import create_app
        except ImportError:
            pytest.skip("fastapi/httpx not installed")

        app = create_app()
        return TestClient(app)

    @pytest.fixture
    def session_with_social(self, client):
        """Create a session with social_dynamics enabled."""
        resp = client.post("/api/simulation/sessions", json={
            "config": {
                "initial_population": 25,
                "generations_to_run": 10,
                "random_seed": 42,
                "extensions_enabled": [
                    "geography", "resources", "technology",
                    "culture", "conflict", "social_dynamics",
                ],
            },
        })
        assert resp.status_code == 200
        session_id = resp.json()["id"]

        # Run enough generations for marriages/clans/institutions to emerge
        resp = client.post(
            f"/api/simulation/sessions/{session_id}/step",
            json={"n": 5},
        )
        assert resp.status_code == 200
        return session_id

    def test_marriages_endpoint(self, client, session_with_social):
        resp = client.get(f"/api/social/{session_with_social}/marriages")
        assert resp.status_code == 200
        data = resp.json()
        assert data["enabled"] is True
        assert "married_count" in data
        assert "avg_duration" in data
        assert "total_divorces" in data

    def test_clans_endpoint(self, client, session_with_social):
        resp = client.get(f"/api/social/{session_with_social}/clans")
        assert resp.status_code == 200
        data = resp.json()
        assert data["enabled"] is True
        assert "clans" in data
        assert isinstance(data["clans"], list)

    def test_institutions_endpoint(self, client, session_with_social):
        resp = client.get(f"/api/social/{session_with_social}/institutions")
        assert resp.status_code == 200
        data = resp.json()
        assert data["enabled"] is True
        assert "institutions" in data
        assert isinstance(data["institutions"], list)

    def test_disabled_returns_false(self, client):
        """Endpoints return enabled=False without social_dynamics."""
        resp = client.post("/api/simulation/sessions", json={
            "config": {
                "initial_population": 10,
                "generations_to_run": 5,
                "random_seed": 42,
            },
        })
        sid = resp.json()["id"]
        client.post(f"/api/simulation/sessions/{sid}/step", json={"n": 1})

        for endpoint in ["marriages", "clans", "institutions"]:
            resp = client.get(f"/api/social/{sid}/{endpoint}")
            assert resp.status_code == 200
            assert resp.json()["enabled"] is False

    def test_404_session(self, client):
        """Missing session returns 404."""
        for endpoint in ["marriages", "clans", "institutions"]:
            resp = client.get(f"/api/social/nonexistent/{endpoint}")
            assert resp.status_code == 404
