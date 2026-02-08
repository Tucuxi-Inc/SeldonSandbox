"""
Outsider Interface — inject foreign agents and track their impact.

Supports manual injection, archetype-based injection, and scheduled
injections from config. The RippleTracker monitors how outsider traits
propagate through the population over generations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from seldon.core.agent import Agent
from seldon.core.config import ExperimentConfig
from seldon.core.processing import ProcessingClassifier
from seldon.experiment.archetypes import create_agent_from_archetype


@dataclass
class InjectionRecord:
    """Record of a single outsider injection."""
    agent_id: str
    generation: int
    origin: str
    traits_at_injection: np.ndarray


@dataclass
class DiffusionSnapshot:
    """Snapshot of outsider trait diffusion at a generation."""
    generation: int
    outsider_count: int
    descendant_count: int
    total_population: int
    trait_distance_from_mean: float  # How different outsider traits are from pop mean


class OutsiderInterface:
    """
    Inject foreign agents into the simulation.

    Supports:
    - Manual injection with custom traits
    - Archetype-based injection with noise
    - Scheduled injections from config
    """

    def __init__(self, config: ExperimentConfig, classifier: ProcessingClassifier):
        self.config = config
        self.ts = config.trait_system
        self.classifier = classifier
        self.injections: list[InjectionRecord] = []
        self._injection_counter = 0

    def inject_outsider(
        self,
        traits: np.ndarray,
        generation: int,
        origin: str = "manual",
        name: str | None = None,
        agent_id: str | None = None,
        age: int | None = None,
    ) -> Agent:
        """
        Inject a custom-trait outsider into the simulation.

        Returns the created agent (caller adds to population).
        """
        self._injection_counter += 1
        aid = agent_id or f"outsider_{self._injection_counter:04d}"
        aname = name or f"Outsider-{self._injection_counter}"
        aage = age if age is not None else self.config.outsider_injection_age

        agent = Agent(
            id=aid,
            name=aname,
            age=aage,
            generation=generation,
            birth_order=1,
            traits=np.clip(traits, 0.0, 1.0),
            traits_at_birth=np.clip(traits.copy(), 0.0, 1.0),
            is_outsider=True,
            outsider_origin=origin,
            injection_generation=generation,
        )
        agent.processing_region = self.classifier.classify(agent)

        self.injections.append(InjectionRecord(
            agent_id=aid,
            generation=generation,
            origin=origin,
            traits_at_injection=agent.traits.copy(),
        ))

        return agent

    def inject_archetype(
        self,
        archetype_name: str,
        generation: int,
        noise_sigma: float = 0.05,
        rng: np.random.Generator | None = None,
        agent_id: str | None = None,
    ) -> Agent:
        """
        Inject an outsider based on an archetype definition.

        Returns the created agent.
        """
        self._injection_counter += 1
        aid = agent_id or f"outsider_{self._injection_counter:04d}"
        rng = rng or np.random.default_rng()

        agent = create_agent_from_archetype(
            archetype_name, self.config, aid,
            generation=generation,
            noise_sigma=noise_sigma,
            rng=rng,
        )
        agent.is_outsider = True
        agent.outsider_origin = f"archetype:{archetype_name}"
        agent.injection_generation = generation
        agent.processing_region = self.classifier.classify(agent)

        self.injections.append(InjectionRecord(
            agent_id=aid,
            generation=generation,
            origin=f"archetype:{archetype_name}",
            traits_at_injection=agent.traits.copy(),
        ))

        return agent

    def process_scheduled_injections(
        self,
        generation: int,
        rng: np.random.Generator,
    ) -> list[Agent]:
        """
        Check config.scheduled_injections and create agents for the current generation.

        Scheduled injections format:
        [
            {"generation": 5, "archetype": "einstein", "count": 1, "noise": 0.05},
            {"generation": 10, "traits": {"openness": 0.9, ...}, "count": 2},
        ]
        """
        new_agents: list[Agent] = []

        for spec in self.config.scheduled_injections:
            if spec.get("generation") != generation:
                continue

            count = spec.get("count", 1)
            for _ in range(count):
                if "archetype" in spec:
                    agent = self.inject_archetype(
                        spec["archetype"],
                        generation,
                        noise_sigma=spec.get("noise", 0.05),
                        rng=rng,
                    )
                elif "traits" in spec:
                    trait_dict = spec["traits"]
                    traits = np.full(self.ts.count, 0.5)
                    for trait_name, value in trait_dict.items():
                        try:
                            idx = self.ts.trait_index(trait_name)
                            traits[idx] = value
                        except KeyError:
                            pass
                    agent = self.inject_outsider(
                        traits, generation,
                        origin=spec.get("origin", "scheduled"),
                    )
                else:
                    continue

                new_agents.append(agent)

        return new_agents


class RippleTracker:
    """
    Track how outsider traits diffuse through the population.

    Records per-generation snapshots of outsider and descendant counts,
    and measures trait distance from population mean.
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.ts = config.trait_system
        self.snapshots: list[DiffusionSnapshot] = []
        self.injection_records: list[InjectionRecord] = []

    def track_injection(self, record: InjectionRecord) -> None:
        """Register an injection for tracking."""
        self.injection_records.append(record)

    def track_generation(self, population: list[Agent], generation: int) -> DiffusionSnapshot:
        """Take a diffusion snapshot for this generation."""
        if not population:
            snap = DiffusionSnapshot(
                generation=generation,
                outsider_count=0,
                descendant_count=0,
                total_population=0,
                trait_distance_from_mean=0.0,
            )
            self.snapshots.append(snap)
            return snap

        outsiders = [a for a in population if a.is_outsider]
        descendants = [
            a for a in population
            if not a.is_outsider and a.is_descendant_of_outsider
        ]

        # Calculate trait distance
        trait_distance = 0.0
        if outsiders or descendants:
            pop_traits = np.array([a.traits for a in population])
            pop_mean = pop_traits.mean(axis=0)

            outsider_group = outsiders + descendants
            if outsider_group:
                outsider_traits = np.array([a.traits for a in outsider_group])
                outsider_mean = outsider_traits.mean(axis=0)
                trait_distance = float(np.linalg.norm(outsider_mean - pop_mean))

        snap = DiffusionSnapshot(
            generation=generation,
            outsider_count=len(outsiders),
            descendant_count=len(descendants),
            total_population=len(population),
            trait_distance_from_mean=trait_distance,
        )
        self.snapshots.append(snap)
        return snap

    def get_diffusion_report(self) -> dict[str, Any]:
        """Get a summary report of outsider diffusion over time."""
        if not self.snapshots:
            return {"injections": 0, "snapshots": []}

        return {
            "injections": len(self.injection_records),
            "snapshots": [
                {
                    "generation": s.generation,
                    "outsider_count": s.outsider_count,
                    "descendant_count": s.descendant_count,
                    "total_population": s.total_population,
                    "outsider_fraction": (
                        (s.outsider_count + s.descendant_count) / s.total_population
                        if s.total_population > 0 else 0
                    ),
                    "trait_distance": s.trait_distance_from_mean,
                }
                for s in self.snapshots
            ],
        }
