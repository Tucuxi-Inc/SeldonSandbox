"""Tests for LoreEngine and Memory."""

import numpy as np
import pytest

from seldon.core.config import ExperimentConfig
from seldon.social.lore import LoreEngine, Memory, MemoryType


class TestMemoryCreation:
    def test_create_memory(self):
        config = ExperimentConfig()
        lore = LoreEngine(config)
        mem = lore.create_memory(
            content="Test event",
            emotional_valence=0.5,
            trait_modifiers={"creativity": 0.01},
            generation=3,
            source_agent_id="agent_001",
        )
        assert mem.fidelity == 1.0
        assert mem.memory_type == MemoryType.PERSONAL
        assert mem.emotional_valence == 0.5
        assert mem.content == "Test event"
        assert mem.created_generation == 3

    def test_create_breakthrough_memory(self):
        config = ExperimentConfig()
        lore = LoreEngine(config)
        mem = lore.create_breakthrough_memory("agent_001", 5, 1.2)
        assert mem.memory_type == MemoryType.PERSONAL
        assert mem.emotional_valence > 0
        assert "creativity" in mem.trait_modifiers

    def test_create_suffering_memory(self):
        config = ExperimentConfig()
        lore = LoreEngine(config)
        mem = lore.create_suffering_memory("agent_001", 5, 0.8)
        assert mem.emotional_valence < 0
        assert "resilience" in mem.trait_modifiers

    def test_emotional_valence_clamped(self):
        config = ExperimentConfig()
        lore = LoreEngine(config)
        mem = lore.create_memory("test", 5.0, {}, 0)
        assert mem.emotional_valence == 1.0
        mem2 = lore.create_memory("test", -5.0, {}, 0)
        assert mem2.emotional_valence == -1.0


class TestMemorySerialization:
    def test_to_dict_and_back(self):
        mem = Memory(
            id="abc123",
            content="A great discovery",
            memory_type=MemoryType.FAMILY,
            fidelity=0.75,
            emotional_valence=0.6,
            trait_modifiers={"creativity": 0.01, "resilience": 0.005},
            created_generation=2,
            source_agent_id="agent_001",
            transmission_count=1,
            mutation_count=0,
        )
        d = mem.to_dict()
        restored = Memory.from_dict(d)

        assert restored.id == mem.id
        assert restored.content == mem.content
        assert restored.memory_type == MemoryType.FAMILY
        assert restored.fidelity == mem.fidelity
        assert restored.emotional_valence == mem.emotional_valence
        assert restored.trait_modifiers == mem.trait_modifiers
        assert restored.transmission_count == 1


class TestFidelityDecay:
    def test_decay_reduces_fidelity(self):
        config = ExperimentConfig(lore_decay_rate=0.1)
        lore = LoreEngine(config)
        mem = lore.create_memory("test", 0.5, {}, 0)
        assert mem.fidelity == 1.0
        lore.decay_memory(mem)
        assert mem.fidelity == pytest.approx(0.9)

    def test_multiple_decays(self):
        config = ExperimentConfig(lore_decay_rate=0.1)
        lore = LoreEngine(config)
        mem = lore.create_memory("test", 0.5, {}, 0)
        for _ in range(10):
            lore.decay_memory(mem)
        assert mem.fidelity < 0.4

    def test_fidelity_never_negative(self):
        config = ExperimentConfig(lore_decay_rate=0.5)
        lore = LoreEngine(config)
        mem = lore.create_memory("test", 0.5, {}, 0)
        for _ in range(100):
            lore.decay_memory(mem)
        assert mem.fidelity >= 0.0


class TestTypeProgression:
    def test_personal_stays_personal(self):
        config = ExperimentConfig()
        lore = LoreEngine(config)
        mem = lore.create_memory("test", 0.5, {}, 0)
        assert mem.memory_type == MemoryType.PERSONAL

    def test_becomes_family_after_transmission(self):
        config = ExperimentConfig()
        lore = LoreEngine(config)
        mem = lore.create_memory("test", 0.5, {}, 0)
        rng = np.random.default_rng(42)
        child_mem = lore.transmit_to_child(mem, rng)
        assert child_mem.memory_type == MemoryType.FAMILY

    def test_becomes_societal_after_many_transmissions(self):
        config = ExperimentConfig(lore_mutation_rate=0.0)
        lore = LoreEngine(config)
        mem = lore.create_memory("test", 0.5, {}, 0)
        rng = np.random.default_rng(42)
        current = mem
        for _ in range(3):
            current = lore.transmit_to_child(current, rng)
        assert current.memory_type == MemoryType.SOCIETAL

    def test_becomes_myth_at_low_fidelity(self):
        config = ExperimentConfig(lore_myth_threshold=0.3, lore_decay_rate=0.3)
        lore = LoreEngine(config)
        mem = lore.create_memory("test", 0.5, {}, 0)
        # Decay until fidelity drops below myth threshold
        while mem.fidelity >= 0.3:
            lore.decay_memory(mem)
        assert mem.memory_type == MemoryType.MYTH


class TestTransmission:
    def test_transmission_probability(self):
        config = ExperimentConfig(lore_transmission_rate=1.0)
        lore = LoreEngine(config)
        mem = lore.create_memory("test", 0.8, {}, 0)
        rng = np.random.default_rng(42)
        # High emotional valence + rate 1.0 should almost always transmit
        transmissions = sum(lore.should_transmit(mem, rng) for _ in range(100))
        assert transmissions > 80

    def test_low_valence_less_likely_to_transmit(self):
        config = ExperimentConfig(lore_transmission_rate=0.5)
        lore = LoreEngine(config)
        mem_high = lore.create_memory("test", 0.9, {}, 0)
        mem_low = lore.create_memory("test", 0.1, {}, 0)
        rng = np.random.default_rng(42)

        high_count = sum(lore.should_transmit(mem_high, rng) for _ in range(200))
        low_count = sum(lore.should_transmit(mem_low, rng) for _ in range(200))
        assert high_count > low_count

    def test_transmission_disabled_when_lore_off(self):
        config = ExperimentConfig(lore_enabled=False)
        lore = LoreEngine(config)
        mem = lore.create_memory("test", 0.9, {}, 0)
        rng = np.random.default_rng(42)
        assert not lore.should_transmit(mem, rng)

    def test_child_memory_has_reduced_fidelity(self):
        config = ExperimentConfig(lore_decay_rate=0.1, lore_mutation_rate=0.0)
        lore = LoreEngine(config)
        mem = lore.create_memory("test", 0.5, {}, 0)
        rng = np.random.default_rng(42)
        child_mem = lore.transmit_to_child(mem, rng)
        assert child_mem.fidelity < mem.fidelity
        assert child_mem.fidelity == pytest.approx(0.9)  # 1.0 * (1 - 0.1)

    def test_transmission_increments_count(self):
        config = ExperimentConfig(lore_mutation_rate=0.0)
        lore = LoreEngine(config)
        mem = lore.create_memory("test", 0.5, {}, 0)
        rng = np.random.default_rng(42)
        child_mem = lore.transmit_to_child(mem, rng)
        assert child_mem.transmission_count == 1


class TestMutation:
    def test_mutation_modifies_memory(self):
        config = ExperimentConfig(lore_mutation_rate=1.0)  # Always mutate
        lore = LoreEngine(config)
        mem = lore.create_memory("test", 0.5, {"creativity": 0.01}, 0)
        rng = np.random.default_rng(42)
        child_mem = lore.transmit_to_child(mem, rng)
        assert child_mem.mutation_count == 1

    def test_no_mutation_when_rate_zero(self):
        config = ExperimentConfig(lore_mutation_rate=0.0)
        lore = LoreEngine(config)
        mem = lore.create_memory("test", 0.5, {"creativity": 0.01}, 0)
        rng = np.random.default_rng(42)
        child_mem = lore.transmit_to_child(mem, rng)
        assert child_mem.mutation_count == 0
        assert child_mem.trait_modifiers == mem.trait_modifiers


class TestSocietalLore:
    def test_evolve_societal_lore(self):
        config = ExperimentConfig()
        lore = LoreEngine(config)
        rng = np.random.default_rng(42)

        # Create a memory shared by many agents
        mem = lore.create_memory("Shared discovery", 0.8, {"creativity": 0.01}, 0)
        mem_dict = mem.to_dict()

        # Simulate many agents having the same memory
        population_memories = [[mem_dict] for _ in range(20)]
        result = lore.evolve_societal_lore(population_memories, 1, rng)
        assert len(result) > 0
        assert result[0].memory_type == MemoryType.SOCIETAL

    def test_societal_lore_decays(self):
        config = ExperimentConfig(lore_decay_rate=0.1)
        lore = LoreEngine(config)
        rng = np.random.default_rng(42)

        mem = Memory(
            id="test", content="Ancient event", memory_type=MemoryType.SOCIETAL,
            fidelity=0.5, emotional_valence=0.3, trait_modifiers={},
            created_generation=0, transmission_count=5,
        )
        lore.societal_memories = [mem]
        lore.evolve_societal_lore([], 5, rng)
        assert mem.fidelity < 0.5


class TestTraitModifiers:
    def test_aggregate_modifiers_weighted_by_fidelity(self):
        config = ExperimentConfig()
        lore = LoreEngine(config)
        memories = [
            {"fidelity": 1.0, "trait_modifiers": {"creativity": 0.01}},
            {"fidelity": 0.5, "trait_modifiers": {"creativity": 0.01}},
        ]
        result = lore.get_trait_modifiers_from_lore(memories)
        assert result["creativity"] == pytest.approx(0.015)

    def test_empty_memories_return_empty(self):
        config = ExperimentConfig()
        lore = LoreEngine(config)
        result = lore.get_trait_modifiers_from_lore([])
        assert result == {}
