"""Atomic shared image admission. Fail closed when Redis cannot enforce limits.

The request counter is intentionally conservative: attempted provider calls count
even if they fail. Unknown prices are never treated as free. This is a request
limit, not a dollar budget; existing telemetry retains actual usage separately.
"""

from __future__ import annotations

import asyncio
import uuid
from contextlib import asynccontextmanager, suppress
from datetime import UTC, datetime

import openai
from redis.asyncio import Redis

from llm_wiki.config import settings

ACQUIRE = """
local now = tonumber(redis.call('TIME')[1])
redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', now)
if redis.call('ZCARD', KEYS[1]) >= tonumber(ARGV[1]) then return 0 end
if tonumber(redis.call('GET', KEYS[2]) or '0') >= tonumber(ARGV[2]) then return -1 end
redis.call('ZADD', KEYS[1], now + 900, ARGV[3])
redis.call('EXPIRE', KEYS[1], 1000)
redis.call('INCR', KEYS[2])
redis.call('EXPIRE', KEYS[2], 172800)
return 1
"""


@asynccontextmanager
async def image_slot():
    # A disabled visual feature keeps the existing infographic deployment intact.
    if not settings.visual_presentations_enabled:
        yield
        return
    token = uuid.uuid4().hex
    day = datetime.now(UTC).strftime("%Y-%m-%d")
    slots, count = "aqyl:images:{budget}:slots", f"aqyl:images:{{budget}}:{day}"
    client = Redis.from_url(settings.redis_url, socket_connect_timeout=2, socket_timeout=2)
    acquired = False
    unknown = False
    try:
        try:
            admitted = await client.eval(
                ACQUIRE,
                2,
                slots,
                count,
                settings.image_global_concurrency,
                settings.image_daily_request_limit,
                token,
            )
        except Exception as exc:
            raise ValueError("image_limiter_unavailable") from exc
        if admitted != 1:
            raise ValueError("image_capacity" if admitted == 0 else "image_daily_limit")
        acquired = True
        try:
            yield
        except (TimeoutError, openai.APIConnectionError, asyncio.CancelledError):
            # The provider may still be computing. Keep its lease until expiry.
            unknown = True
            raise
    finally:
        if acquired and not unknown:
            with suppress(Exception):  # bounded expiry also releases a stranded lease
                await client.zrem(slots, token)
        await client.aclose()
