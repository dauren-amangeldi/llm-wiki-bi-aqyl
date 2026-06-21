"""FastAPI dependency injection — DB sessions and shared resources."""

from collections.abc import AsyncGenerator

from fastapi import Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from llm_wiki.api.rate_limit import InMemoryRateLimiter
from llm_wiki.api.schemas import CurrentUser
from llm_wiki.config import settings
from llm_wiki.storage.metadata import ensure_dev_user, get_or_create_user

_engine = create_async_engine(settings.database_url, echo=False)
_SessionLocal: async_sessionmaker[AsyncSession] = async_sessionmaker(
    bind=_engine,
    expire_on_commit=False,
    autoflush=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Yield an async SQLAlchemy session, closing it after the request."""
    async with _SessionLocal() as session:
        yield session


async def get_current_user(
    request: Request,
    session: AsyncSession = Depends(get_db),
) -> CurrentUser:
    """Resolve the caller from ``X-User-Id`` (dev stub — no JWT).

    When the header is absent, falls back to the stable dev user
    ``dev-user``.  Creates the user row on first sight so ``user.id`` is
    always persisted and can flow into notebook ownership.
    """
    user_id = request.headers.get("X-User-Id", "dev-user")
    name = request.headers.get("X-User-Name", "Dev User")
    role = request.headers.get("X-User-Role", "admin")

    if user_id == "dev-user" and "X-User-Id" not in request.headers:
        user = await ensure_dev_user(session)
    else:
        user = await get_or_create_user(session, user_id, name, role)

    return CurrentUser(id=user.id, name=user.name, role=user.role)


# ---------------------------------------------------------------------------
# Rate-limiter singletons (one per endpoint group, initialised lazily)
# ---------------------------------------------------------------------------

_files_rate_limiter: InMemoryRateLimiter | None = None
_ask_rate_limiter: InMemoryRateLimiter | None = None


def get_files_rate_limiter() -> InMemoryRateLimiter:
    """Return the singleton rate limiter for POST /files."""
    global _files_rate_limiter
    if _files_rate_limiter is None:
        _files_rate_limiter = InMemoryRateLimiter(
            max_requests=settings.ingestion_rate_limit_per_min,
            window_seconds=60,
        )
    return _files_rate_limiter


def get_ask_rate_limiter() -> InMemoryRateLimiter:
    """Return the singleton rate limiter for POST /ask."""
    global _ask_rate_limiter
    if _ask_rate_limiter is None:
        _ask_rate_limiter = InMemoryRateLimiter(
            max_requests=settings.ask_rate_limit_per_min,
            window_seconds=60,
        )
    return _ask_rate_limiter


def check_files_rate_limit(
    request: Request,
    limiter: InMemoryRateLimiter = Depends(get_files_rate_limiter),
) -> None:
    """FastAPI dependency that enforces the per-IP rate limit on POST /files.

    Raises:
        HTTPException 429: With ``Retry-After`` header when limit is exceeded.
    """
    key = request.client.host if request.client else "unknown"
    allowed, retry_after = limiter.check(key)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded: {settings.ingestion_rate_limit_per_min} files/min per IP",
            headers={"Retry-After": str(retry_after)},
        )


def check_ask_rate_limit(
    request: Request,
    limiter: InMemoryRateLimiter = Depends(get_ask_rate_limiter),
) -> None:
    """FastAPI dependency that enforces the per-IP rate limit on POST /ask.

    Raises:
        HTTPException 429: With ``Retry-After`` header when limit is exceeded.
    """
    key = request.client.host if request.client else "unknown"
    allowed, retry_after = limiter.check(key)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded: {settings.ask_rate_limit_per_min} requests/min per IP",
            headers={"Retry-After": str(retry_after)},
        )
