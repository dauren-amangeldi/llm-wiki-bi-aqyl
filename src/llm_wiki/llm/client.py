"""Unified LLM client wrapper with usage tracking and retry logic.

This is the ONLY module that talks to LLM providers. All agents go through here.
Provider is selected by the LLM_PROVIDER env var — never hardcode a provider.
Implemented in LW-4.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal


@dataclass
class LLMUsage:
    """Usage record written to data/usage.log after every LLM call."""

    file_id: str
    agent_type: Literal["search", "writer", "lint"]
    model: str
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int
    cost_usd: float
    timestamp: datetime
    duration_ms: int


class LLMClient:
    """Provider-agnostic async LLM client.

    Supports:
      - ollama (OpenAI-compatible local API)
      - openai (GPT-5.4 / GPT-5.4 Mini)
      - anthropic (Claude, fallback)

    Every call:
      1. Loads the prompt from llm/prompts/*.md (never hardcoded).
      2. Retries up to 3× with exponential backoff on transient errors.
      3. Writes an LLMUsage record to data/usage.log as JSON-lines.
    """

    PROMPTS_DIR = Path(__file__).parent / "prompts"

    def __init__(self) -> None:
        """Initialise the client from environment settings."""
        raise NotImplementedError("Implemented in LW-4")

    def load_prompt(self, name: str, **variables: Any) -> str:
        """Load a prompt from llm/prompts/{name}.md and interpolate variables.

        Args:
            name: Prompt file stem (e.g. 'search', 'writer_create').
            **variables: Placeholder values substituted into the prompt text.

        Returns:
            Fully rendered prompt string ready to send to the LLM.
        """
        template = (self.PROMPTS_DIR / f"{name}.md").read_text(encoding="utf-8")
        return template.format(**variables)

    async def complete(
        self,
        prompt: str,
        system: str,
        file_id: str,
        agent_type: Literal["search", "writer", "lint"],
        response_format: Literal["text", "json"] = "text",
    ) -> tuple[str, LLMUsage]:
        """Send a completion request and return the response with usage.

        Args:
            prompt: User message content.
            system: System message content.
            file_id: Correlation ID for usage tracking.
            agent_type: Which agent is making the call.
            response_format: 'text' or 'json' (structured output).

        Returns:
            Tuple of (response_text, LLMUsage). usage.log is written as a
            side effect.
        """
        raise NotImplementedError("Implemented in LW-4")

    def _compute_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Compute USD cost using the price table from config.

        Args:
            model: Model identifier (must exist in settings.price_table).
            input_tokens: Number of input tokens billed.
            output_tokens: Number of output tokens billed.

        Returns:
            Cost in USD, rounded to 6 decimal places.
        """
        from llm_wiki.config import settings

        prices = settings.price_table.get(model) or settings.price_table.get("ollama", {"input": 0.0, "output": 0.0})
        cost = (input_tokens * prices["input"] + output_tokens * prices["output"]) / 1_000_000
        return round(cost, 6)
