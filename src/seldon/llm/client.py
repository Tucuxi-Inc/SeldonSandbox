"""
LLM client abstraction supporting Anthropic Claude and local Ollama models.

Provides graceful degradation when providers are unavailable.
Ollama connects via ``host.docker.internal`` when running inside Docker,
falling back to ``localhost`` for native execution.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


class LLMUnavailableError(Exception):
    """Raised when LLM features are requested but no provider is available."""


@dataclass
class LLMResponse:
    """Response from an LLM API call."""

    text: str
    model: str
    input_tokens: int
    output_tokens: int


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class LLMClient(ABC):
    """Common interface for LLM providers."""

    provider: str  # "anthropic" or "ollama"

    @abstractmethod
    def complete(
        self,
        system: str,
        messages: list[dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> LLMResponse: ...


# ---------------------------------------------------------------------------
# Anthropic (Claude)
# ---------------------------------------------------------------------------

DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-20250514"


class ClaudeClient(LLMClient):
    """Thin wrapper around ``anthropic.Anthropic``."""

    provider = "anthropic"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_CLAUDE_MODEL,
    ) -> None:
        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not resolved_key:
            raise LLMUnavailableError(
                "ANTHROPIC_API_KEY not set. Set the environment variable or pass api_key to enable LLM features."
            )

        try:
            import anthropic
        except ImportError:
            raise LLMUnavailableError(
                "The 'anthropic' package is not installed. Run: pip install anthropic"
            )

        self._client = anthropic.Anthropic(api_key=resolved_key)
        self.model = model

    @staticmethod
    def is_available() -> bool:
        """Return True if the Anthropic SDK is installed and an API key is set."""
        if not os.environ.get("ANTHROPIC_API_KEY"):
            return False
        try:
            import anthropic  # noqa: F401
            return True
        except ImportError:
            return False

    def complete(
        self,
        system: str,
        messages: list[dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> LLMResponse:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=messages,
        )

        text = response.content[0].text if response.content else ""
        return LLMResponse(
            text=text,
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )


# ---------------------------------------------------------------------------
# Ollama (local models)
# ---------------------------------------------------------------------------

DEFAULT_OLLAMA_MODEL = "llama3.2"

# Inside Docker, reach the host's Ollama via host.docker.internal.
# Native runs use localhost.
_OLLAMA_HOST_DOCKER = "http://host.docker.internal:11434"
_OLLAMA_HOST_LOCAL = "http://localhost:11434"


def _ollama_base_url() -> str:
    """Determine the Ollama base URL, preferring explicit env var."""
    explicit = os.environ.get("OLLAMA_HOST")
    if explicit:
        # Ensure it has a scheme
        if not explicit.startswith("http"):
            explicit = f"http://{explicit}"
        return explicit
    # Auto-detect: if DOCKER_CONTAINER env var is set or /.dockerenv exists, use Docker host
    if os.environ.get("DOCKER_CONTAINER") or os.path.exists("/.dockerenv"):
        return _OLLAMA_HOST_DOCKER
    return _OLLAMA_HOST_LOCAL


class OllamaClient(LLMClient):
    """Client for locally-running Ollama models.

    No API key required. Connects to Ollama's HTTP API.
    """

    provider = "ollama"

    def __init__(self, model: str = DEFAULT_OLLAMA_MODEL, base_url: str | None = None) -> None:
        self.model = model
        self.base_url = base_url or _ollama_base_url()

    @staticmethod
    def is_available(base_url: str | None = None) -> bool:
        """Check if Ollama is reachable and has at least one model."""
        import urllib.request
        import urllib.error
        url = (base_url or _ollama_base_url()) + "/api/tags"
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=3) as resp:
                return resp.status == 200
        except Exception:
            return False

    @staticmethod
    def list_models(base_url: str | None = None) -> list[str]:
        """Return names of locally available Ollama models."""
        import json
        import urllib.request
        import urllib.error
        url = (base_url or _ollama_base_url()) + "/api/tags"
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode())
                return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []

    def complete(
        self,
        system: str,
        messages: list[dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> LLMResponse:
        import json
        import urllib.request

        # Build Ollama chat payload
        ollama_messages = [{"role": "system", "content": system}]
        for msg in messages:
            ollama_messages.append({"role": msg["role"], "content": msg["content"]})

        payload = json.dumps({
            "model": self.model,
            "messages": ollama_messages,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        }).encode()

        req = urllib.request.Request(
            self.base_url + "/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode())
        except Exception as e:
            raise LLMUnavailableError(f"Ollama request failed: {e}")

        text = data.get("message", {}).get("content", "")
        # Ollama provides token counts in different fields
        input_tokens = data.get("prompt_eval_count", 0) or 0
        output_tokens = data.get("eval_count", 0) or 0

        return LLMResponse(
            text=text,
            model=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_client(
    provider: str = "anthropic",
    model: str | None = None,
    api_key: str | None = None,
    ollama_base_url: str | None = None,
) -> LLMClient:
    """Create an LLM client for the specified provider.

    Parameters
    ----------
    provider : str
        ``"anthropic"`` or ``"ollama"``.
    model : str | None
        Model name. Defaults to provider-specific default.
    api_key : str | None
        API key (Anthropic only).
    ollama_base_url : str | None
        Override Ollama base URL.
    """
    if provider == "anthropic":
        return ClaudeClient(api_key=api_key, model=model or DEFAULT_CLAUDE_MODEL)
    elif provider == "ollama":
        return OllamaClient(model=model or DEFAULT_OLLAMA_MODEL, base_url=ollama_base_url)
    else:
        raise ValueError(f"Unknown provider: {provider!r}. Use 'anthropic' or 'ollama'.")
