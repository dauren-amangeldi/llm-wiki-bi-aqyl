"""Unified LLM client wrapper with usage tracking and retry logic.

This is the ONLY module that talks to LLM providers. All agents go through here.
Provider is selected by the LLM_PROVIDER env var — never hardcode a provider.
"""

import asyncio
import json
import time
import weakref
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import openai
import structlog

from llm_wiki.llm.telemetry import estimate_cost, media_call, response_usage, write_usage
from llm_wiki.observability import context, error_fields, new_id

logger = structlog.get_logger(__name__)
_PROVIDER_USAGE: ContextVar[dict[str, Any] | None] = ContextVar("provider_usage", default=None)

# Non-retryable OpenAI 4xx errors (429 = RateLimitError IS retried; these are not)
_NON_RETRYABLE_OPENAI: tuple[type[Exception], ...] = (
    openai.AuthenticationError,
    openai.PermissionDeniedError,
    openai.BadRequestError,
    openai.NotFoundError,
    openai.UnprocessableEntityError,
)

# Cap on CONCURRENT provider calls per event loop («повар не страдает»): however
# many requests/tasks fan out, at most settings.llm_max_concurrency calls are in
# flight at once — the rest queue here instead of storming the provider (429s).
# Semaphores are loop-bound, so keep one per loop (API process has one loop;
# each prefork worker task runs its own) — a module global would break across
# asyncio.Runner instances.
_LOOP_SEMAPHORES: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore]" = (
    weakref.WeakKeyDictionary()
)


def _llm_semaphore() -> asyncio.Semaphore:
    from llm_wiki.config import settings

    loop = asyncio.get_running_loop()
    sem = _LOOP_SEMAPHORES.get(loop)
    if sem is None:
        sem = asyncio.Semaphore(max(1, settings.llm_max_concurrency))
        _LOOP_SEMAPHORES[loop] = sem
    return sem


@dataclass
class LLMUsage:
    """Usage record written to data/usage.log after every LLM call."""

    file_id: str
    agent_type: Literal["search", "writer", "lint", "audit", "embed", "answer", "advisor", "twins", "tagger", "artifact"]
    model: str
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int
    cost_usd: float
    timestamp: datetime
    duration_ms: int
    call_id: str = field(default_factory=new_id)
    attempts: int = 1
    semaphore_wait_ms: float = 0
    provider_duration_ms: float = 0
    backoff_ms: float = 0
    reasoning_tokens: int | None = None
    usage_known: bool = True


class LLMClient:
    """Provider-agnostic async LLM client.

    Supports:
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
        Provider can be openai or anthropic.
        """
        from llm_wiki.config import settings
        from llm_wiki.quality.budget import BudgetGuard

        self._provider: str = settings.llm_provider
        self._usage_log_path: Path = settings.usage_log_path

        # Budget guard — checks daily limits before every LLM / embed call.
        # Constructed with current settings; limits of None = disabled.
        self._budget = BudgetGuard(
            usage_log_path=settings.usage_log_path,
            daily_cost_limit_usd=settings.daily_cost_limit_usd,
            daily_token_limit=settings.daily_token_limit,
        )

        # _client is typed Any because AsyncOpenAI and AsyncAnthropic have
        # different method signatures — dispatch happens in _call_provider.
        # _non_retryable is built per-instance to avoid global mutation.
        timeout = settings.llm_timeout_s
        match self._provider:
            case "openai":
                self._client: Any = openai.AsyncOpenAI(
                    api_key=settings.openai_api_key,
                    timeout=timeout,
                )
                self._model: str = settings.openai_model
                self._non_retryable: tuple[type[Exception], ...] = _NON_RETRYABLE_OPENAI
            case "anthropic":
                import anthropic

                self._client = anthropic.AsyncAnthropic(
                    api_key=settings.anthropic_api_key,
                    timeout=timeout,
                )
                self._model = settings.anthropic_model
                self._non_retryable = _NON_RETRYABLE_OPENAI + (
                    anthropic.AuthenticationError,
                    anthropic.PermissionDeniedError,
                    anthropic.BadRequestError,
                )
            case _:
                raise ValueError(f"Unknown LLM_PROVIDER: {self._provider!r}")

    async def aclose(self) -> None:
        """Close the underlying SDK client and release all HTTP connections.

        Must be called when the LLMClient is no longer needed, **within the
        same event loop** that was active when ``complete()`` was first called.

        We snapshot the event loop's task set BEFORE calling the SDK's
        ``aclose()`` and gather only tasks that appeared *during* that call —
        these are the httpx fire-and-forget TLS-shutdown / pool-cleanup tasks
        we actually want to drain.

        Gathering *all* pending tasks (the previous behaviour) is safe under
        ``asyncio.Runner`` (Celery), where the loop runs a single coroutine,
        but **deadlocks under FastAPI/uvicorn**, where ``asyncio.all_tasks()``
        includes the long-lived server task and other in-flight request
        handlers — waiting on those never returns and the HTTP response never
        reaches the client.
        """
        loop = asyncio.get_running_loop()

        # Snapshot tasks that exist BEFORE teardown — uvicorn server task,
        # other request handlers, lifespan tasks, etc. We must NOT wait on
        # these; they belong to the process, not to this LLMClient instance.
        pre_existing: set[asyncio.Task[Any]] = set(asyncio.all_tasks(loop))

        close = getattr(self._client, "aclose", None)
        if callable(close):
            await close()

        # Only the diff: tasks spawned by httpx during its teardown (TLS
        # stream close, connection-pool drain, etc.).
        current = asyncio.current_task()
        new_tasks = [
            t
            for t in asyncio.all_tasks(loop)
            if t not in pre_existing and t is not current and not t.done()
        ]
        if new_tasks:
            # Bounded wait — if a cleanup task is somehow wedged, we'd rather
            # leak it than hang the caller. 2 s is ample for local TLS teardown.
            try:
                await asyncio.wait_for(
                    asyncio.gather(*new_tasks, return_exceptions=True),
                    timeout=2.0,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "llm_aclose_cleanup_timeout",
                    stuck_tasks=len(new_tasks),
                )

    def embed(self, texts: list[str], file_id: str = "") -> list[list[float]]:
        """Generate embeddings for *texts* using the OpenAI embeddings API.

        Embeddings always use OpenAI's ``text-embedding-3-small`` regardless of
        the configured chat provider (Anthropic does not have a standalone
        embedding API comparable to OpenAI).

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

        # Budget check — raises BudgetExceeded before any API call is made.
        from llm_wiki.quality.budget import BudgetExceeded

        try:
            self._budget.check()
        except BudgetExceeded as exc:
            logger.error("llm_call_blocked_by_budget", agent_type="embed", error=str(exc))
            raise

        from llm_wiki.config import settings

        if not settings.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is required for embeddings (text-embedding-3-small)."
            )

        sync_client = openai.OpenAI(
            api_key=settings.openai_api_key,
            timeout=settings.llm_timeout_s,
        )
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
                    logger.info(
                        "embed_batch_start",
                        file_id=file_id,
                        batch_start=batch_start,
                        batch_size=len(batch),
                        attempt=attempt + 1,
                    )
                    with media_call(model, "embed", file_id, usage_log_path=self._usage_log_path, batch_start=batch_start,
                                    batch_size=len(batch), attempt=attempt + 1) as telemetry:
                        response = sync_client.embeddings.create(
                            model=model, input=batch, dimensions=settings.embedding_dimensions)
                        telemetry["response"] = response
                    duration_ms = int((time.monotonic() - start) * 1000)
                    all_vectors.extend(item.embedding for item in response.data)
                    logger.info(
                        "embed_batch_done",
                        file_id=file_id,
                        batch_start=batch_start,
                        duration_ms=duration_ms,
                    )
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
                            **error_fields(exc),
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
        agent_type: Literal["search", "writer", "lint", "audit", "answer", "advisor", "twins", "tagger", "artifact"],
        response_format: Literal["text", "json"] = "text",
        json_schema: dict[str, Any] | None = None,
        schema_name: str = "response",
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
            response_format: ``'text'`` or ``'json'`` (loose JSON mode).
            json_schema: When provided, request **strict Structured Outputs** —
                the model is constrained to return JSON that exactly matches this
                JSON Schema (OpenAI ``response_format=json_schema, strict=true``).
                Takes precedence over ``response_format``. For Anthropic the
                schema is injected into the system prompt as a fallback.
            schema_name: Name for the schema (OpenAI requires a non-empty name).

        Returns:
            Tuple of ``(response_text, LLMUsage)``.  The usage record is
            appended to ``data/usage.log`` as a side effect.

        Raises:
            BudgetExceeded: When the daily cost or token budget is exhausted.
            Exception: Re-raises after ``_MAX_RETRIES`` failed attempts, or
                immediately for non-retryable errors (auth, bad request, etc.).
        """
        from llm_wiki.quality.budget import BudgetExceeded

        call_id = new_id()
        fields = dict(call_id=call_id, model=self._model, agent_type=agent_type, file_id=file_id,
                      provider=self._provider)
        # SDK calls may internally retry; attempts below counts wrapper attempts.
        sdk_retries = getattr(self._client, "max_retries", None)
        fields["sdk_max_retries"] = sdk_retries if isinstance(sdk_retries, int) else None
        started = time.perf_counter()
        provider_ms = wait_ms = backoff_ms = 0.0
        attempts = 0
        outcome = "failed"
        failure = {}
        token = _PROVIDER_USAGE.set(None)
        logger.info("ai_call_started", **fields)
        try:
            self._budget.check()
            for attempt in range(self._MAX_RETRIES):
                wait_started = time.perf_counter()
                provider_started = None
                try:
                    async with _llm_semaphore():
                        wait_ms += (time.perf_counter() - wait_started) * 1000
                        attempts += 1
                        provider_started = time.perf_counter()
                        try:
                            text, incoming, outgoing, cached = await self._call_provider(
                                prompt, system, response_format, json_schema, schema_name)
                        finally:
                            provider_ms += (time.perf_counter() - provider_started) * 1000
                except Exception as exc:
                    retry = not isinstance(exc, self._non_retryable) and attempt < self._MAX_RETRIES - 1
                    logger.warning("ai_attempt_failed", **fields, attempt=attempts,
                                   retrying=retry, **error_fields(exc))
                    if not retry:
                        raise
                    backoff = 4**attempt
                    logger.warning("llm_retry", **fields, attempt=attempts, backoff_s=backoff,
                                   **error_fields(exc))
                    backoff_started = time.perf_counter()
                    await asyncio.sleep(backoff)
                    backoff_ms += (time.perf_counter() - backoff_started) * 1000
                    continue
                details = _PROVIDER_USAGE.get()
                usage = LLMUsage(
                    file_id=file_id, agent_type=agent_type, model=self._model,
                    input_tokens=incoming, output_tokens=outgoing, cached_input_tokens=cached,
                    cost_usd=self._compute_cost(self._model, incoming, outgoing, cached),
                    timestamp=datetime.now(timezone.utc), duration_ms=int((time.perf_counter()-started)*1000),
                    call_id=call_id, attempts=attempts, semaphore_wait_ms=round(wait_ms, 2),
                    provider_duration_ms=round(provider_ms, 2), backoff_ms=round(backoff_ms, 2),
                    reasoning_tokens=details.get("reasoning_tokens") if details else None,
                    usage_known=details is None or (details.get("input_tokens") is not None and details.get("output_tokens") is not None),
                )
                self._write_usage(usage)
                outcome = "success"
                return text, usage
            raise RuntimeError("LLM call failed after retries")
        except BudgetExceeded as exc:
            outcome = "blocked"
            failure = error_fields(exc)
            logger.error("llm_call_blocked_by_budget", **fields, **failure)
            raise
        except asyncio.CancelledError:
            outcome = "cancelled"
            raise
        except BaseException as exc:
            failure = error_fields(exc)
            raise
        finally:
            logger.info("ai_call_finished", **fields, outcome=outcome, attempts=attempts,
                        duration_ms=round((time.perf_counter()-started)*1000, 2),
                        provider_duration_ms=round(provider_ms, 2), semaphore_wait_ms=round(wait_ms, 2),
                        backoff_ms=round(backoff_ms, 2), **failure)
            _PROVIDER_USAGE.reset(token)

    # ------------------------------------------------------------------
    # Image generation (infographic artifact)
    # ------------------------------------------------------------------

    async def generate_image(self, prompt: str) -> str:
        """Generate an image via the OpenAI Images API; return a PNG data URI.

        Used by the infographic artifact. OpenAI provider only. Raises on any
        provider/config/API error so the caller can fall back to a non-image
        rendering. The art-director prompt includes the infographic text;
        structured data is also rendered as HTML cards beside the picture.

        Handles both response shapes: ``b64_json`` (gpt-image-1) and a temporary
        ``url`` (dall-e-3), fetching + encoding the latter so the stored artifact
        is self-contained (no expiring external link). ``response_format`` is not
        passed — some image models reject it as an unknown parameter.
        """
        import base64

        from llm_wiki.config import settings

        if self._provider != "openai":
            raise RuntimeError("Image generation requires the OpenAI provider")

        from llm_wiki.llm.image_limits import image_slot

        with media_call(settings.image_model, "image", context().get("document_id", "image"),
                        usage_log_path=self._usage_log_path,
                        image_count=1, image_size=settings.image_size, image_quality=settings.image_quality) as telemetry:
            async with image_slot():
                client = self._client.with_options(max_retries=0) if settings.visual_presentations_enabled else self._client
                response = await client.images.generate(
                    model=settings.image_model,
                    prompt=prompt,
                    size=settings.image_size,
                    quality=settings.image_quality,
                    n=1,
                    timeout=min(max(settings.llm_timeout_s, 180), 300),
                )
            telemetry["response"] = response
        item = response.data[0] if response.data else None
        if item is None:
            raise RuntimeError("Image API returned no image")

        b64 = getattr(item, "b64_json", None)
        if not b64:
            url = getattr(item, "url", None)
            if not url:
                raise RuntimeError("Image API returned neither b64_json nor url")
            import httpx

            async with httpx.AsyncClient(timeout=60) as http:
                resp = await http.get(url)
                resp.raise_for_status()
                b64 = base64.b64encode(resp.content).decode("ascii")

        logger.info("image_generated", model=settings.image_model, b64_len=len(b64))
        return f"data:image/png;base64,{b64}"

    # ------------------------------------------------------------------
    # Provider dispatch
    # ------------------------------------------------------------------

    async def _call_provider(
        self,
        prompt: str,
        system: str,
        response_format: Literal["text", "json"],
        json_schema: dict[str, Any] | None = None,
        schema_name: str = "response",
    ) -> tuple[str, int, int, int]:
        """Dispatch to the right SDK and return (text, in_tok, out_tok, cached).

        Args:
            prompt: User message.
            system: System message.
            response_format: ``'text'`` or ``'json'``.
            json_schema: Strict Structured Outputs schema (see ``complete``).
            schema_name: Name for the schema.

        Returns:
            Tuple of ``(response_text, input_tokens, output_tokens, cached_tokens)``.
        """
        if self._provider == "openai":
            return await self._call_openai(
                prompt, system, response_format, json_schema, schema_name
            )
        return await self._call_anthropic(
            prompt, system, response_format, json_schema
        )

    async def _call_openai(
        self,
        prompt: str,
        system: str,
        response_format: Literal["text", "json"],
        json_schema: dict[str, Any] | None = None,
        schema_name: str = "response",
    ) -> tuple[str, int, int, int]:
        """Call the OpenAI Chat Completions API.

        Args:
            prompt: User message.
            system: System message.
            response_format: ``'text'`` or ``'json'``.
            json_schema: When set, use strict Structured Outputs instead of the
                loose ``json_object`` mode.
            schema_name: Name for the schema.

        Returns:
            ``(text, input_tokens, output_tokens, cached_tokens)``
        """
        from llm_wiki.config import settings

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "timeout": settings.llm_timeout_s,
        }
        if json_schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "schema": json_schema,
                    "strict": True,
                },
            }
        elif response_format == "json":
            kwargs["response_format"] = {"type": "json_object"}

        response = await self._client.chat.completions.create(**kwargs)
        _PROVIDER_USAGE.set(response_usage(response))
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
        json_schema: dict[str, Any] | None = None,
    ) -> tuple[str, int, int, int]:
        """Call Anthropic Messages API.

        JSON output is requested via a system-prompt instruction rather than
        a native parameter, since Anthropic does not support OpenAI's
        ``response_format`` shapes. When a ``json_schema`` is given it is
        injected into the system prompt as a best-effort strictness hint.

        Args:
            prompt: User message.
            system: System message.
            response_format: ``'text'`` or ``'json'``.
            json_schema: Optional schema injected into the system prompt.

        Returns:
            ``(text, input_tokens, output_tokens, cached_tokens)``
        """
        effective_system = system
        if json_schema is not None:
            effective_system += (
                "\nRespond ONLY with valid JSON (no markdown fences) that matches "
                "this JSON Schema:\n" + json.dumps(json_schema, ensure_ascii=False)
            )
        elif response_format == "json":
            effective_system += "\nRespond ONLY with valid JSON, no markdown fences."

        from llm_wiki.config import settings

        response = await self._client.messages.create(
            model=self._model,
            max_tokens=4096,
            system=effective_system,
            messages=[{"role": "user", "content": prompt}],
            timeout=settings.llm_timeout_s,
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
        record = asdict(usage)
        record["timestamp"] = usage.timestamp.isoformat()
        record["provider"] = self._provider if usage.agent_type != "embed" else "openai"
        record["outcome"] = "success"
        if not usage.usage_known:
            record["input_tokens"] = record["output_tokens"] = None
        record["cost_usd"], record["cost_status"] = estimate_cost(usage.model, record)
        write_usage(record, self._usage_log_path)

    # ------------------------------------------------------------------
    # Cost computation (not touched by LW-4 — preserved from skeleton)
    # ------------------------------------------------------------------

    def _compute_cost(self, model: str, input_tokens: int, output_tokens: int, cached_input_tokens: int = 0) -> float:
        # Legacy response DTOs require a float. The usage ledger separately
        # reports null + cost_status for missing prices instead of implying $0.
        cost, _ = estimate_cost(model, {"input_tokens": input_tokens,
                                      "output_tokens": output_tokens,
                                      "cached_input_tokens": cached_input_tokens})
        return round(cost or 0.0, 6)
