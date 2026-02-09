"""
Narrative generators for generation summaries and decision explanations.
"""

from __future__ import annotations

from typing import Any

from seldon.llm.client import LLMClient, LLMResponse
from seldon.llm.prompts import (
    NARRATIVE_SYSTEM_PROMPT,
    DECISION_EXPLAIN_SYSTEM_PROMPT,
    build_generation_context,
    build_decision_context,
)


class NarrativeGenerator:
    """Generates prose narratives for simulation generations.

    Parameters
    ----------
    client : LLMClient
        Any LLM provider (Anthropic or Ollama).
    """

    def __init__(self, client: LLMClient) -> None:
        self.client = client

    def narrate_generation(
        self,
        generation: int,
        metrics: dict[str, Any],
        notable_agents: list[dict[str, Any]] | None = None,
        societal_lore: list[dict[str, Any]] | None = None,
        extension_metrics: dict[str, Any] | None = None,
        previous_narrative: str | None = None,
    ) -> LLMResponse:
        """Generate a prose narrative for a simulation generation.

        Parameters
        ----------
        generation : int
            Generation number.
        metrics : dict
            Serialized GenerationMetrics.
        notable_agents : list | None
            Top agents by contribution.
        societal_lore : list | None
            Current societal memories.
        extension_metrics : dict | None
            Extension-provided metrics.
        previous_narrative : str | None
            Previous generation's narrative for continuity.
        """
        context = build_generation_context(
            generation=generation,
            metrics=metrics,
            notable_agents=notable_agents,
            societal_lore=societal_lore,
            extension_metrics=extension_metrics,
        )

        prompt = f"Write a narrative for this generation:\n\n{context}"
        if previous_narrative:
            prompt += f"\n\n--- PREVIOUS GENERATION NARRATIVE ---\n{previous_narrative}\n\nContinue the story, maintaining continuity."

        return self.client.complete(
            system=NARRATIVE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.8,
        )


class DecisionNarrator:
    """Explains agent decisions in psychological terms.

    Parameters
    ----------
    client : LLMClient
        Any LLM provider (Anthropic or Ollama).
    """

    def __init__(self, client: LLMClient) -> None:
        self.client = client

    def explain_decision(
        self,
        agent_data: dict[str, Any],
        decision: dict[str, Any],
        trait_names: list[str],
    ) -> LLMResponse:
        """Generate a psychological explanation for a specific decision.

        Parameters
        ----------
        agent_data : dict
            Serialized agent detail.
        decision : dict
            A single entry from the agent's ``decision_history``.
        trait_names : list[str]
            Ordered trait names.
        """
        context = build_decision_context(agent_data, decision, trait_names)
        prompt = f"Explain this decision:\n\n{context}"

        return self.client.complete(
            system=DECISION_EXPLAIN_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.6,
        )
