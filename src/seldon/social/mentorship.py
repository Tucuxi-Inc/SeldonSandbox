"""
Mentorship Manager — mentor-mentee matching, knowledge transfer, lineage tracking.

Experienced high-performing agents mentor younger agents, transferring
skills and nudging trait development. Matching uses the decision model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from seldon.core.agent import Agent
    from seldon.core.config import ExperimentConfig


class MentorshipManager:
    """Manages mentor-mentee relationships and knowledge transfer."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.ts = config.trait_system
        self.mc = config.mentorship_config

    def match_mentors(
        self, population: list[Agent], rng: np.random.Generator,
    ) -> list[tuple[Agent, Agent]]:
        """
        Match eligible mentors with mentees.

        Mentors: age > mentor_min_age, mean contribution > 0.5, < max_mentees.
        Mentees: age 5-mentee_max_age, no current mentor.

        Returns list of (mentor, mentee) tuples.
        """
        if not self.mc.get("enabled", True):
            return []

        mentor_min_age = self.mc["mentor_min_age"]
        mentee_max_age = self.mc["mentee_max_age"]
        max_mentees = self.mc["max_mentees"]

        eligible_mentors = [
            a for a in population
            if (a.is_alive
                and int(a.age) >= mentor_min_age
                and len(a.mentee_ids) < max_mentees
                and a.contribution_history
                and float(np.mean(a.contribution_history)) > 0.5)
        ]

        eligible_mentees = [
            a for a in population
            if (a.is_alive
                and 5 <= int(a.age) <= mentee_max_age
                and a.mentor_id is None)
        ]

        if not eligible_mentors or not eligible_mentees:
            return []

        matches: list[tuple[Agent, Agent]] = []
        rng.shuffle(eligible_mentees)

        for mentee in eligible_mentees:
            if not eligible_mentors:
                break

            # Score each potential mentor
            scores = []
            for mentor in eligible_mentors:
                score = self._compatibility_score(mentor, mentee)
                scores.append(score)

            scores_arr = np.array(scores)
            scores_arr = np.maximum(scores_arr, 0.01)
            probs = scores_arr / scores_arr.sum()

            chosen_idx = rng.choice(len(eligible_mentors), p=probs)
            mentor = eligible_mentors[chosen_idx]

            # Link them
            mentee.mentor_id = mentor.id
            mentor.mentee_ids.append(mentee.id)
            matches.append((mentor, mentee))

            # Update social bonds
            mentee.social_bonds[mentor.id] = max(
                mentee.social_bonds.get(mentor.id, 0.0), 0.3,
            )
            mentor.social_bonds[mentee.id] = max(
                mentor.social_bonds.get(mentee.id, 0.0), 0.2,
            )

            # Remove mentor if at capacity
            if len(mentor.mentee_ids) >= max_mentees:
                eligible_mentors.remove(mentor)

        return matches

    def apply_mentorship_effects(
        self, population: list[Agent], rng: np.random.Generator,
    ) -> int:
        """
        Apply mentorship effects for all active mentorships.

        - Mentee traits nudge toward mentor's
        - Skill transfer
        - Memory creation

        Returns number of active mentorships processed.
        """
        if not self.mc.get("enabled", True):
            return 0

        agent_map = {a.id: a for a in population}
        influence_rate = self.mc["mentorship_influence_rate"]
        skill_rate = self.mc["skill_transfer_rate"]
        count = 0

        for agent in population:
            if agent.mentor_id is None or not agent.is_alive:
                continue

            mentor = agent_map.get(agent.mentor_id)
            if mentor is None or not mentor.is_alive:
                continue

            # Trait nudge: move mentee's traits slightly toward mentor's
            direction = mentor.traits - agent.traits
            agent.traits = np.clip(
                agent.traits + direction * influence_rate, 0.0, 1.0,
            )

            # Skill transfer: mentor's top skills partially copied
            if mentor.skills:
                for skill_name, skill_val in mentor.skills.items():
                    current = agent.skills.get(skill_name, 0.0)
                    agent.skills[skill_name] = min(
                        1.0, current + (skill_val - current) * skill_rate,
                    )

            # Strengthen bond
            agent.social_bonds[mentor.id] = min(
                1.0, agent.social_bonds.get(mentor.id, 0.3) + 0.02,
            )
            mentor.social_bonds[agent.id] = min(
                1.0, mentor.social_bonds.get(agent.id, 0.2) + 0.01,
            )

            count += 1

        return count

    def dissolve_mentorships(
        self, population: list[Agent], rng: np.random.Generator,
    ) -> list[tuple[str, str]]:
        """
        Dissolve stale mentorships.

        Dissolves when:
        - Mentee ages out (> mentee_max_age + 5)
        - Mentor died
        - Compatibility drops below threshold

        Returns list of (mentor_id, mentee_id) dissolved.
        """
        dissolved: list[tuple[str, str]] = []
        agent_map = {a.id: a for a in population}
        max_age = self.mc["mentee_max_age"] + 5
        threshold = self.mc["dissolution_compatibility_threshold"]

        for agent in population:
            if agent.mentor_id is None:
                continue

            should_dissolve = False
            mentor = agent_map.get(agent.mentor_id)

            # Mentor died or missing
            if mentor is None or not mentor.is_alive:
                should_dissolve = True
            # Mentee aged out
            elif int(agent.age) > max_age:
                should_dissolve = True
            # Low compatibility
            elif self._compatibility_score(mentor, agent) < threshold:
                if rng.random() < 0.3:
                    should_dissolve = True

            if should_dissolve:
                mentor_id = agent.mentor_id
                agent.mentor_id = None
                if mentor and mentor_id in [a.id for a in population]:
                    if agent.id in mentor.mentee_ids:
                        mentor.mentee_ids.remove(agent.id)
                dissolved.append((mentor_id, agent.id))

        return dissolved

    def get_mentorship_chains(
        self, population: list[Agent],
    ) -> list[dict[str, Any]]:
        """
        Build mentorship chain trees for visualization.

        Returns list of tree roots (mentors with no mentor of their own).
        """
        agent_map = {a.id: a for a in population}
        # Find root mentors (mentors who don't have a mentor themselves)
        all_mentor_ids = {a.mentor_id for a in population if a.mentor_id}

        chains: list[dict[str, Any]] = []
        visited: set[str] = set()

        for mentor_id in all_mentor_ids:
            mentor = agent_map.get(mentor_id)
            if mentor is None or mentor.id in visited:
                continue
            # Check if this mentor has a mentor (not a root)
            if mentor.mentor_id is not None:
                continue

            chain = self._build_chain(mentor, agent_map, visited)
            if chain:
                chains.append(chain)

        return chains

    def _build_chain(
        self, agent: Agent, agent_map: dict[str, Agent],
        visited: set[str],
    ) -> dict[str, Any]:
        """Recursively build a mentorship chain from a root mentor."""
        if agent.id in visited:
            return {}
        visited.add(agent.id)

        mentees = []
        for mentee_id in agent.mentee_ids:
            mentee = agent_map.get(mentee_id)
            if mentee and mentee.is_alive:
                mentees.append(self._build_chain(mentee, agent_map, visited))

        return {
            "id": agent.id,
            "name": agent.name,
            "role": agent.social_role,
            "status": round(agent.social_status, 3),
            "mentees": mentees,
        }

    def _compatibility_score(self, mentor: Agent, mentee: Agent) -> float:
        """
        Score compatibility between a potential mentor-mentee pair.

        Based on: conscientiousness similarity, complementary skills,
        same community/location bonus, status difference.
        """
        score = 0.0

        # Conscientiousness similarity (both should be reasonably conscientious)
        try:
            cons_idx = self.ts.trait_index("conscientiousness")
            cons_sim = 1.0 - abs(float(mentor.traits[cons_idx] - mentee.traits[cons_idx]))
            score += cons_sim * 0.3
        except KeyError:
            score += 0.15

        # Status difference (mentor should be higher status)
        status_diff = mentor.social_status - mentee.social_status
        if status_diff > 0:
            score += min(status_diff, 0.5) * 0.3
        else:
            score -= 0.1

        # Same location/community bonus
        if mentor.location_id and mentor.location_id == mentee.location_id:
            score += 0.2
        if mentor.community_id and mentor.community_id == mentee.community_id:
            score += 0.1

        # Existing bond bonus
        if mentor.id in mentee.social_bonds:
            score += 0.1

        return max(score, 0.0)
