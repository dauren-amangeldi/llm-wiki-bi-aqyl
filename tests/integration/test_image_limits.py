"""Run with TEST_IMAGE_REDIS_URL pointing at an isolated Redis database."""

import asyncio
import os
from datetime import UTC, datetime

import pytest
import pytest_asyncio
from redis.asyncio import Redis

from llm_wiki.config import settings
from llm_wiki.llm.image_limits import image_slot


@pytest_asyncio.fixture
async def limiter(monkeypatch):
    url = os.environ.get("TEST_IMAGE_REDIS_URL")
    if not url:
        pytest.skip("isolated image limiter Redis is not configured")
    monkeypatch.setattr(settings, "redis_url", url)
    monkeypatch.setattr(settings, "visual_presentations_enabled", True)
    monkeypatch.setattr(settings, "image_global_concurrency", 2)
    monkeypatch.setattr(settings, "image_daily_request_limit", 20)
    client = Redis.from_url(url)
    keys = [
        "aqyl:images:{budget}:slots",
        "aqyl:images:{budget}:" + datetime.now(UTC).strftime("%Y-%m-%d"),
    ]
    await client.delete(*keys)
    yield client, keys
    await client.delete(*keys)
    await client.aclose()


@pytest.mark.asyncio
async def test_replicas_share_the_same_concurrency(limiter):
    maximum = active = allowed = 0
    release = asyncio.Event()

    async def attempt():
        nonlocal active, maximum, allowed
        try:
            async with image_slot():
                active += 1
                allowed += 1
                maximum = max(maximum, active)
                await release.wait()
                active -= 1
        except ValueError as exc:
            assert str(exc) == "image_capacity"

    tasks = [asyncio.create_task(attempt()) for _ in range(10)]
    for _ in range(100):
        if sum(t.done() for t in tasks) == 8:
            break
        await asyncio.sleep(0.01)
    assert allowed == maximum == 2
    release.set()
    await asyncio.gather(*tasks)
    assert await limiter[0].zcard(limiter[1][0]) == 0


@pytest.mark.asyncio
async def test_daily_limit_counts_attempts_not_only_successes(limiter, monkeypatch):
    monkeypatch.setattr(settings, "image_daily_request_limit", 2)
    for _ in range(2):
        with pytest.raises(RuntimeError):
            async with image_slot():
                raise RuntimeError("provider failed")
    with pytest.raises(ValueError, match="image_daily_limit"):
        async with image_slot():
            pytest.fail("must not call the paid provider")


@pytest.mark.asyncio
async def test_unknown_outcome_holds_slot_until_expiry(limiter):
    client, keys = limiter
    with pytest.raises(TimeoutError):
        async with image_slot():
            raise TimeoutError()
    assert await client.zcard(keys[0]) == 1
    token = (await client.zrange(keys[0], 0, -1))[0]
    await client.zadd(keys[0], {token: 1})
    async with image_slot():
        assert await client.zcard(keys[0]) == 1
    assert await client.zcard(keys[0]) == 0


@pytest.mark.asyncio
async def test_redis_outage_never_bypasses_limits(monkeypatch):
    monkeypatch.setattr(settings, "visual_presentations_enabled", True)
    monkeypatch.setattr(settings, "redis_url", "redis://127.0.0.1:1/0")
    with pytest.raises(ValueError, match="image_limiter_unavailable"):
        async with image_slot():
            pytest.fail("must not call the paid provider")
