"""Unified LLM client wrapper with usage tracking and retry logic.

This is the ONLY module that talks to LLM providers. All agents go through here.
Provider is selected by the LLM_PROVIDER env var — never hardcode a provider.
"""

import asyncio
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import openai
import structlog
from filelock import FileLock

logger = structlog.get_logger(__name__)

# Non-retryable OpenAI 4xx errors (429 = RateLimitError IS retried; these are not)
_NON_RETRYABLE_OPENAI: tuple[type[Exception], ...] = (
    openai.AuthenticationError,
    openai.PermissionDeniedError,
    openai.BadRequestError,
    openai.NotFoundError,
    openai.UnprocessableEntityError,
)


@dataclass
class LLMUsage:
    """Usage record written to data/usage.log after every LLM call."""

    file_id: str
    agent_type: Literal["search", "writer", "lint", "audit", "embed"]
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
      1. Dispatches to the correct SDK based on LLM_PROVIDER.
      2. Retries up to 3× with exponential backoff on transient errors.
      3. Writes an LLMUsage record to data/usage.log as a JSON-line.
    """

    PROMPTS_DIR = Path(__file__).parent / "prompts"
    _MAX_RETRIES = 3

    def __init__(self) -> None:
        """Initialise the client from environment settings.

        Reads LLM_PROVIDER and creates the appropriate SDK client.
        Provider can be ollama, openai, or anthropic.
        """
        from llm_wiki.config import settings

        self._provider: str = settings.llm_provider
        self._usage_log_path: Path = settings.usage_log_path

        # _client is typed Any because AsyncOpenAI and AsyncAnthropic have
        # different method signatures — dispatch happens in _call_provider.
        # _non_retryable is built per-instance to avoid global mutation.
        match self._provider:
            case "openai":
                self._client: Any = openai.AsyncOpenAI(api_key=settings.openai_api_key)
                self._model: str = settings.openai_model
                self._non_retryable: tuple[type[Exception], ...] = _NON_RETRYABLE_OPENAI
            case "anthropic":
                import anthropic

                self._client = anthropic.AsyncAnthropic(
                    api_key=settings.anthropic_api_key
                )
                self._model = settings.anthropic_model
                self._non_retryable = _NON_RETRYABLE_OPENAI + (
                    anthropic.AuthenticationError,
                    anthropic.PermissionDeniedError,
                    anthropic.BadRequestError,
                )
            case "ollama":
                self._client = openai.AsyncOpenAI(
                    base_url=f"{settings.ollama_base_url}/v1",
                    api_key="ollama",  # Ollama ignores the key but SDK requires non-empty
                )
                self._model = settings.ollama_model
                self._non_retryable = _NON_RETRYABLE_OPENAI
            case _:
                raise ValueError(f"Unknown LLM_PROVIDER: {self._provider!r}")

    async def aclose(self) -> None:
        """Close the underlying SDK client and release all HTTP connections.

        Must be called when the LLMClient is no longer needed, **within the
        same event loop** that was active when ``complete()`` was first called.
        Calling this prevents the ``RuntimeError: Event loop is closed`` warning
        that appears when GC later tries to clean up an open httpx.AsyncClient
        on a dead loop.
        """
        close = getattr(self._client, "aclose", None)
        if callable(close):
            await close()

        # httpx spawns fire-and-forget asyncio.Tasks to tear down TLS streams
        # and clean up the connection pool.  If we return now, asyncio.Runner
        # will close the event loop before those tasks finish, producing
        # "RuntimeError: Event loop is closed".  Gather them explicitly.
        loop = asyncio.get_running_loop()
        pending = [
            t for t in asyncio.all_tasks(loop)
            if t is not asyncio.current_task() and not t.done()
        ]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    def embed(self, texts: list[str], file_id: str = "") -> list[list[float]]:
        """Generate embeddings for *texts* using the OpenAI embeddings API.

        Embeddings always use OpenAI's ``text-embedding-3-small`` regardless of
        the configured chat provider (Anthropic/Ollama do not have standalone
        embedding APIs comparable to OpenAI).

        Retries up to ``_MAX_RETRIES`` times with exponential backoff on
        transient errors.  Non-retryable 4xx errors are raised immediately.
        Usage is appended to ``data/usage.log`` after each successful call.

        Args:
            texts: Strings to embed.  Empty list returns immediately.
            file_id: Correlation ID for usage tracking.

        Returns:
            List of embedding vectors (one per input string), ordered
            to match the input.

        Raises:
            openai.OpenAIError: Re-raised after ``_MAX_RETRIES`` failures, or
                immediately for non-retryable auth/bad-request errors.
            ValueError: If ``OPENAI_API_KEY`` is not configured.
        """
        if not texts:
            return []

        from llm_wiki.config import settings

        if not settings.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is required for embeddings (text-embedding-3-small)."
            )

        sync_client = openai.OpenAI(api_key=settings.openai_api_key)
        model = settings.embedding_model
        batch_size = settings.embedding_batch_size

        all_vectors: list[list[float]] = []
        # Process in batches to stay within OpenAI's input limit
        for batch_start in range(0, len(texts), batch_size):
            batch = texts[batch_start : batch_start + batch_size]
            last_exc: Exception | None = None

            for attempt in range(self._MAX_RETRIES):
                start = time.monotonic()
                try:
                    response = sync_client.embeddings.create(
                        model=model,
                        input=batch,
                        dimensions=settings.embedding_dimensions,
                    )
                    duration_ms = int((time.monotonic() - start) * 1000)
                    input_tokens: int = (
                        response.usage.total_tokens if response.usage else len(batch) * 5
                    )
                    cost = self._compute_cost(model, input_tokens, 0)
                    usage = LLMUsage(
                        file_id=file_id,
                        agent_type="embed",
                        model=model,
                        input_tokens=input_tokens,
                        output_tokens=0,
                        cached_input_tokens=0,
                        cost_usd=cost,
                        timestamp=datetime.now(timezone.utc),
                        duration_ms=duration_ms,
                    )
                    self._write_usage(usage)
                    all_vectors.extend(item.embedding for item in response.data)
                    break  # success — move to next batch
                except Exception as exc:  # noqa: BLE001
                    if isinstance(exc, self._non_retryable):
                        raise
                    last_exc = exc
                    if attempt < self._MAX_RETRIES - 1:
                        backoff = 4**attempt
                        logger.warning(
                            "embed_retry",
                            file_id=file_id,
                            batch_start=batch_start,
                            attempt=attempt + 1,
                            backoff_s=backoff,
                            error=str(exc),
                        )
                        time.sleep(backoff)
            else:
                raise last_exc or RuntimeError("embed() failed after retries")

        return all_vectors

    def load_prompt(self, prompt_name: str, **variables: Any) -> str:
        """Load a prompt from llm/prompts/{prompt_name}.md and interpolate variables.

        Args:
            prompt_name: Prompt file stem (e.g. ``'search'``, ``'writer_create'``).
            **variables: Placeholder values substituted into the prompt text.

        Returns:
            Fully rendered prompt string ready to send to the LLM.
        """
        template = (self.PROMPTS_DIR / f"{prompt_name}.md").read_text(encoding="utf-8")
        return template.format(**variables)

    async def complete(
        self,
        prompt: str,
        system: str,
        file_id: str,
        agent_type: Literal["search", "writer", "lint", "audit"],
        response_format: Literal["text", "json"] = "text",
    ) -> tuple[str, LLMUsage]:
        """Send a completion request and return the response with usage.

        Retries up to 3 times with exponential backoff (1 s, 4 s, 16 s) on
        transient errors (rate limits, connection errors, 5xx).  Never retries
        on 4xx client errors (except 429 which is handled by the SDK as
        RateLimitError and IS retried).

        Args:
            prompt: User message content.
            system: System message content.
            file_id: Correlation ID for usage tracking and structured logs.
            agent_type: Which agent is making the call (used in usage log).
            response_format: ``'text'`` or ``'json'`` (structured JSON output).

        Returns:
            Tuple of ``(response_text, LLMUsage)``.  The usage record is
            appended to ``data/usage.log`` as a side effect.

        Raises:
            Exception: Re-raises after ``_MAX_RETRIES`` failed attempts, or
                immediately for non-retryable errors (auth, bad request, etc.).
        """
        start = time.monotonic()
        last_exc: Exception | None = None

        for attempt in range(self._MAX_RETRIES):
            try:
                text, input_tokens, output_tokens, cached = await self._call_provider(
                    prompt, system, response_format
                )
                duration_ms = int((time.monotonic() - start) * 1000)
                cost = self._compute_cost(self._model, input_tokens, output_tokens)
                usage = LLMUsage(
                    file_id=file_id,
                    agent_type=agent_type,
                    model=self._model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cached_input_tokens=cached,
                    cost_usd=cost,
                    timestamp=datetime.now(timezone.utc),
                    duration_ms=duration_ms,
                )
                self._write_usage(usage)
                logger.info(
                    "llm_call",
                    file_id=file_id,
                    agent_type=agent_type,
                    model=self._model,
                    cost_usd=cost,
                    duration_ms=duration_ms,
                )
                return text, usage

            except Exception as exc:  # noqa: BLE001
                if isinstance(exc, self._non_retryable):
                    raise
                last_exc = exc
                if attempt < self._MAX_RETRIES - 1:
                    backoff = 4**attempt  # 1 s, 4 s, 16 s
                    logger.warning(
                        "llm_retry",
                        file_id=file_id,
                        attempt=attempt + 1,
                        backoff_s=backoff,
                        error=str(exc),
                    )
                    await asyncio.sleep(backoff)

        raise last_exc or RuntimeError("LLM call failed after retries with no recorded exception")

    # ------------------------------------------------------------------
    # Provider dispatch
    # ------------------------------------------------------------------

    async def _call_provider(
        self,
        prompt: str,
        system: str,
        response_format: Literal["text", "json"],
    ) -> tuple[str, int, int, int]:
        """Dispatch to the right SDK and return (text, in_tok, out_tok, cached).

        Args:
            prompt: User message.
            system: System message.
            response_format: ``'text'`` or ``'json'``.

        Returns:
            Tuple of ``(response_text, input_tokens, output_tokens, cached_tokens)``.
        """
        if self._provider in ("openai", "ollama"):
            return await self._call_openai(prompt, system, response_format)
        return await self._call_anthropic(prompt, system, response_format)

    async def _call_openai(
        self,
        prompt: str,
        system: str,
        response_format: Literal["text", "json"],
    ) -> tuple[str, int, int, int]:
        """Call OpenAI-compatible API (OpenAI or Ollama).

        Args:
            prompt: User message.
            system: System message.
            response_format: ``'text'`` or ``'json'``.

        Returns:
            ``(text, input_tokens, output_tokens, cached_tokens)``
        """
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "timeout": 60,
        }
        if response_format == "json":
            kwargs["response_format"] = {"type": "json_object"}

        response = await self._client.chat.completions.create(**kwargs)
        text: str = response.choices[0].message.content or ""
        input_tokens: int = response.usage.prompt_tokens if response.usage else 0
        output_tokens: int = response.usage.completion_tokens if response.usage else 0
        cached: int = 0
        if response.usage and response.usage.prompt_tokens_details:
            cached = response.usage.prompt_tokens_details.cached_tokens or 0
        return text, input_tokens, output_tokens, cached

    async def _call_anthropic(
        self,
        prompt: str,
        system: str,
        response_format: Literal["text", "json"],
    ) -> tuple[str, int, int, int]:
        """Call Anthropic Messages API.

        JSON output is requested via a system-prompt instruction rather than
        a native parameter, since Anthropic does not yet support
        ``response_format=json_object``.

        Args:
            prompt: User message.
            system: System message.
            response_format: ``'text'`` or ``'json'``.

        Returns:
            ``(text, input_tokens, output_tokens, cached_tokens)``
        """
        effective_system = system
        if response_format == "json":
            effective_system += "\nRespond ONLY with valid JSON, no markdown fences."

        response = await self._client.messages.create(
            model=self._model,
            max_tokens=4096,
            system=effective_system,
            messages=[{"role": "user", "content": prompt}],
        )
        text: str = response.content[0].text
        input_tokens: int = response.usage.input_tokens
        output_tokens: int = response.usage.output_tokens
        cached: int = getattr(response.usage, "cache_read_input_tokens", 0) or 0
        return text, input_tokens, output_tokens, cached

    # ------------------------------------------------------------------
    # Usage logging
    # ------------------------------------------------------------------

    def _write_usage(self, usage: LLMUsage) -> None:
        """Append a JSON-line record to usage.log, protected by a file lock.

        Args:
            usage: The usage record to persist.
        """
        record = {
            "file_id": usage.file_id,
            "agent_type": usage.agent_type,
            "model": usage.model,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cached_input_tokens": usage.cached_input_tokens,
            "cost_usd": usage.cost_usd,
            "timestamp": usage.timestamp.isoformat(),
            "duration_ms": usage.duration_ms,
        }
        lock_path = Path(str(self._usage_log_path) + ".lock")
        with FileLock(str(lock_path)):
            with self._usage_log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")

    # ------------------------------------------------------------------
    # Cost computation (not touched by LW-4 — preserved from skeleton)
    # ------------------------------------------------------------------

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

        prices = settings.price_table.get(model) or settings.price_table.get(
            "ollama", {"input": 0.0, "output": 0.0}
        )
        cost = (input_tokens * prices["input"] + output_tokens * prices["output"]) / 1_000_000
        return round(cost, 6)
