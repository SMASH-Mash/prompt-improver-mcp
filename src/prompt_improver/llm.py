"""LLM backends: the Claude API, and parsing for replies obtained through MCP sampling."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import anthropic

DEFAULT_MODEL = "claude-opus-5-5"
DEFAULT_EFFORT = "medium"
# Models that accept server-side `fallbacks: "default"` (a refused request is re-run on Anthropic's
# recommended fallback model). Claude Haiku 5.5 has no server-side fallback.
_FALLBACK_MODELS = frozenset({"claude-opus-5-5", "claude-opus-5", "claude-fable-5-1", "claude-sonnet-5-5"})
_FALLBACK_BETA = "server-side-fallback-2026-07-01"


class RewriteError(Exception):
    """The backend could not produce a rewrite; the caller falls back to host mode."""


def anthropic_credentials_available() -> bool:
    """Whether the Anthropic SDK can likely authenticate (env vars or an `ant auth login` profile)."""
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return True
    if os.environ.get("ANTHROPIC_PROFILE"):
        return True
    config_home = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    profiles = config_home / "anthropic"
    return profiles.is_dir() and any(profiles.iterdir())


class AnthropicRewriter:
    def __init__(
        self,
        model: str | None = None,
        effort: str | None = None,
        client: anthropic.AsyncAnthropic | None = None,
    ) -> None:
        self.model = model or os.environ.get("PROMPT_IMPROVER_MODEL", DEFAULT_MODEL)
        self.effort = effort or os.environ.get("PROMPT_IMPROVER_EFFORT", DEFAULT_EFFORT)
        self.use_fallbacks = (
            self.model in _FALLBACK_MODELS and os.environ.get("PROMPT_IMPROVER_FALLBACKS", "1") != "0"
        )
        self._client = client

    @property
    def client(self) -> anthropic.AsyncAnthropic:
        if self._client is None:
            self._client = anthropic.AsyncAnthropic()
        return self._client

    async def run(self, system: str, user: str, schema: dict[str, Any]) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": 16000,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "output_config": {"effort": self.effort, "format": {"type": "json_schema", "schema": schema}},
        }
        if self.use_fallbacks:
            kwargs["betas"] = [_FALLBACK_BETA]
            kwargs["fallbacks"] = "default"
        try:
            response = await self.client.beta.messages.create(**kwargs)
        except anthropic.AuthenticationError as e:
            raise RewriteError(f"Anthropic authentication failed: {e.message}") from e
        except anthropic.APIStatusError as e:
            raise RewriteError(f"Anthropic API error {e.status_code}: {e.message}") from e
        except anthropic.APIConnectionError as e:
            raise RewriteError(f"Could not reach the Anthropic API: {e}") from e

        if response.stop_reason == "refusal":
            category = getattr(response.stop_details, "category", None) if response.stop_details else None
            raise RewriteError(f"The model declined to rewrite this prompt (category: {category}).")
        if response.stop_reason == "max_tokens":
            raise RewriteError("The rewrite was cut off at max_tokens.")
        text = next((b.text for b in response.content if b.type == "text"), None)
        if text is None:
            raise RewriteError("The response contained no text block.")
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise RewriteError(f"The response was not valid JSON: {e}") from e


def parse_json_reply(text: str) -> dict[str, Any] | None:
    """Pull the JSON object out of a free-form sampling reply (which may wrap it in prose or a code fence)."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or not isinstance(data.get("improved_prompt"), str):
        return None
    return data
