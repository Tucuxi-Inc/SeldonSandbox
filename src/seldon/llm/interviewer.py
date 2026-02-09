"""
Agent interviewer — lets users have in-character conversations with agents.

Supports multi-turn dialogue via conversation history.
"""

from __future__ import annotations

from typing import Any

from seldon.llm.client import LLMClient, LLMResponse
from seldon.llm.prompts import INTERVIEW_SYSTEM_PROMPT, build_agent_context


class AgentInterviewer:
    """Conducts in-character interviews with simulation agents.

    Parameters
    ----------
    client : LLMClient
        Any LLM provider (Anthropic or Ollama).
    """

    def __init__(self, client: LLMClient) -> None:
        self.client = client

    def interview(
        self,
        agent_data: dict[str, Any],
        trait_names: list[str],
        question: str,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> LLMResponse:
        """Ask an agent a question and get an in-character response.

        Parameters
        ----------
        agent_data : dict
            Serialized agent detail (from ``serialize_agent_detail``).
        trait_names : list[str]
            Ordered list of trait names.
        question : str
            The user's question to the agent.
        conversation_history : list | None
            Previous exchanges as ``[{"role": "user", "content": ...}, ...]``.
            Only the last 3 exchanges (6 messages) are sent.
        """
        context = build_agent_context(agent_data, trait_names)
        system = f"{INTERVIEW_SYSTEM_PROMPT}\n\n--- CHARACTER CONTEXT ---\n{context}"

        messages: list[dict[str, str]] = []

        # Include recent conversation history (last 3 exchanges)
        if conversation_history:
            recent = conversation_history[-6:]  # 3 exchanges = 6 messages
            messages.extend(recent)

        messages.append({"role": "user", "content": question})

        return self.client.complete(
            system=system,
            messages=messages,
            max_tokens=1024,
            temperature=0.7,
        )
