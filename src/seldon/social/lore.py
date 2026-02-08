"""
Lore Engine — generational memory with fidelity decay.

Memories are created from significant events, transmitted parent→child,
and evolve over time. As fidelity decays, personal memories become family
stories, then societal lore, then myths.

Memory types:
    PERSONAL  — Agent's own experience (fidelity 1.0)
    FAMILY    — Inherited from parents/grandparents
    SOCIETAL  — Shared cultural memory (low fidelity, widely known)
    MYTH      — Heavily distorted memory (fidelity < myth_threshold)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from uuid import uuid4

import numpy as np

from seldon.core.config import ExperimentConfig


class MemoryType(Enum):
    PERSONAL = "personal"
    FAMILY = "family"
    SOCIETAL = "societal"
    MYTH = "myth"


@dataclass
class Memory:
    """A single memory with fidelity tracking and emotional valence."""

    id: str
    content: str
    memory_type: MemoryType
    fidelity: float  # 1.0 = perfect, decays over time
    emotional_valence: float  # -1 to 1 (negative=traumatic, positive=joyful)
    trait_modifiers: dict[str, float]  # trait_name -> modifier (how memory affects traits)
    created_generation: int
    source_agent_id: str | None = None

    # Tracking
    transmission_count: int = 0
    mutation_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for agent memory lists."""
        return {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type.value,
            "fidelity": self.fidelity,
            "emotional_valence": self.emotional_valence,
            "trait_modifiers": dict(self.trait_modifiers),
            "created_generation": self.created_generation,
            "source_agent_id": self.source_agent_id,
            "transmission_count": self.transmission_count,
            "mutation_count": self.mutation_count,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Memory:
        """Deserialize from dict."""
        return cls(
            id=d["id"],
            content=d["content"],
            memory_type=MemoryType(d["memory_type"]),
            fidelity=d["fidelity"],
            emotional_valence=d["emotional_valence"],
            trait_modifiers=dict(d["trait_modifiers"]),
            created_generation=d["created_generation"],
            source_agent_id=d.get("source_agent_id"),
            transmission_count=d.get("transmission_count", 0),
            mutation_count=d.get("mutation_count", 0),
        )


# ---------------------------------------------------------------------------
# Lore Engine
# ---------------------------------------------------------------------------
class LoreEngine:
    """
    Manages memory creation, transmission, and evolution.

    Memories decay in fidelity each generation. As they decay, they transition:
        PERSONAL -> FAMILY -> SOCIETAL -> MYTH

    Memories with strong emotional valence are more likely to be transmitted.
    Mutations introduce distortion during transmission.
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.ts = config.trait_system
        self.enabled = config.lore_enabled
        self.decay_rate = config.lore_decay_rate
        self.myth_threshold = config.lore_myth_threshold
        self.mutation_rate = config.lore_mutation_rate
        self.transmission_rate = config.lore_transmission_rate

        # Societal lore shared by the population
        self.societal_memories: list[Memory] = []

    def create_memory(
        self,
        content: str,
        emotional_valence: float,
        trait_modifiers: dict[str, float],
        generation: int,
        source_agent_id: str | None = None,
    ) -> Memory:
        """Create a new personal memory."""
        return Memory(
            id=str(uuid4())[:8],
            content=content,
            memory_type=MemoryType.PERSONAL,
            fidelity=1.0,
            emotional_valence=np.clip(emotional_valence, -1.0, 1.0),
            trait_modifiers=dict(trait_modifiers),
            created_generation=generation,
            source_agent_id=source_agent_id,
        )

    def create_breakthrough_memory(
        self, agent_id: str, generation: int, contribution: float,
    ) -> Memory:
        """Create a memory for a breakthrough event."""
        return self.create_memory(
            content=f"Breakthrough achievement (contribution={contribution:.2f})",
            emotional_valence=0.8,
            trait_modifiers={"creativity": 0.01, "resilience": 0.005, "depth_drive": 0.005},
            generation=generation,
            source_agent_id=agent_id,
        )

    def create_suffering_memory(
        self, agent_id: str, generation: int, suffering: float,
    ) -> Memory:
        """Create a memory for significant suffering."""
        return self.create_memory(
            content=f"Period of deep suffering (level={suffering:.2f})",
            emotional_valence=-0.7,
            trait_modifiers={"resilience": 0.01, "neuroticism": 0.005},
            generation=generation,
            source_agent_id=agent_id,
        )

    def decay_memory(self, memory: Memory) -> Memory:
        """Apply fidelity decay to a memory and update its type."""
        memory.fidelity *= (1.0 - self.decay_rate)
        memory.fidelity = max(0.0, memory.fidelity)
        memory.memory_type = self._classify_type(memory)
        return memory

    def _classify_type(self, memory: Memory) -> MemoryType:
        """Determine memory type based on fidelity and transmission count."""
        if memory.fidelity < self.myth_threshold:
            return MemoryType.MYTH
        if memory.transmission_count >= 3:
            return MemoryType.SOCIETAL
        if memory.transmission_count >= 1:
            return MemoryType.FAMILY
        return MemoryType.PERSONAL

    def should_transmit(self, memory: Memory, rng: np.random.Generator) -> bool:
        """Determine if a memory should be transmitted to a child."""
        if not self.enabled:
            return False
        # Emotional memories are more likely to be transmitted
        prob = self.transmission_rate * (0.5 + 0.5 * abs(memory.emotional_valence))
        return bool(rng.random() < prob)

    def transmit_to_child(
        self, memory: Memory, rng: np.random.Generator,
    ) -> Memory:
        """
        Create a transmitted copy of a memory for a child.

        The copy has reduced fidelity and may be mutated.
        """
        new_memory = Memory(
            id=str(uuid4())[:8],
            content=memory.content,
            memory_type=MemoryType.FAMILY,
            fidelity=memory.fidelity * (1.0 - self.decay_rate),
            emotional_valence=memory.emotional_valence,
            trait_modifiers=dict(memory.trait_modifiers),
            created_generation=memory.created_generation,
            source_agent_id=memory.source_agent_id,
            transmission_count=memory.transmission_count + 1,
            mutation_count=memory.mutation_count,
        )

        # Mutation: distort trait modifiers
        if rng.random() < self.mutation_rate:
            new_memory = self._mutate(new_memory, rng)

        new_memory.memory_type = self._classify_type(new_memory)
        return new_memory

    def _mutate(self, memory: Memory, rng: np.random.Generator) -> Memory:
        """Apply random distortion to a memory's trait modifiers."""
        memory.mutation_count += 1
        if memory.trait_modifiers:
            # Pick a random trait modifier and distort it
            key = rng.choice(list(memory.trait_modifiers.keys()))
            distortion = rng.normal(0, 0.005)
            memory.trait_modifiers[key] += distortion

        # Slightly shift emotional valence
        memory.emotional_valence += rng.normal(0, 0.05)
        memory.emotional_valence = float(np.clip(memory.emotional_valence, -1.0, 1.0))

        return memory

    def evolve_societal_lore(
        self, population_memories: list[list[dict[str, Any]]],
        generation: int,
        rng: np.random.Generator,
    ) -> list[Memory]:
        """
        Process societal-level lore evolution.

        Widely shared memories become societal lore. Conflicting memories
        may merge or one may dominate. All societal lore decays.
        """
        if not self.enabled:
            return self.societal_memories

        # Decay existing societal memories
        for mem in self.societal_memories:
            self.decay_memory(mem)

        # Remove myths that are nearly forgotten
        self.societal_memories = [
            m for m in self.societal_memories
            if m.fidelity > 0.05
        ]

        # Scan population for widely-held memories (by content matching)
        content_counts: dict[str, int] = {}
        content_to_memory: dict[str, dict[str, Any]] = {}
        for agent_mems in population_memories:
            for mem_dict in agent_mems:
                content = mem_dict.get("content", "")
                content_counts[content] = content_counts.get(content, 0) + 1
                content_to_memory[content] = mem_dict

        # Promote widely-held memories to societal lore
        pop_size = max(len(population_memories), 1)
        threshold = max(3, pop_size // 10)  # At least 10% of population
        existing_contents = {m.content for m in self.societal_memories}

        for content, count in content_counts.items():
            if count >= threshold and content not in existing_contents:
                mem_dict = content_to_memory[content]
                mem = Memory.from_dict(mem_dict)
                mem.memory_type = MemoryType.SOCIETAL
                self.societal_memories.append(mem)

        return self.societal_memories

    def get_trait_modifiers_from_lore(
        self, memories: list[dict[str, Any]],
    ) -> dict[str, float]:
        """
        Aggregate trait modifiers from a list of memories.

        Modifiers are weighted by fidelity — faded memories have less influence.
        """
        aggregated: dict[str, float] = {}
        for mem_dict in memories:
            fidelity = mem_dict.get("fidelity", 0.0)
            modifiers = mem_dict.get("trait_modifiers", {})
            for trait_name, modifier in modifiers.items():
                weighted = modifier * fidelity
                aggregated[trait_name] = aggregated.get(trait_name, 0.0) + weighted
        return aggregated
