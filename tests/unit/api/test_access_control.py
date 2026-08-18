"""Tests for the Keycloak access whitelist (allowed_users) + AuthGate middleware.

The middleware tests exercise the real ``access_for_email`` against the test DB
(via ``_SessionLocal``), stubbing only token verification so no real JWT/JWKS is
needed.
"""

from __future__ import annotations

import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from llm_wiki.config import settings
from llm_wiki.storage.metadata import (
    AllowedUser,
    access_for_email,
    allowed_users_count,
    seed_allowed_users,
)

# ---------------------------------------------------------------------------
# Whitelist decision helpers
# ---------------------------------------------------------------------------


async def test_open_access_allows_unknown_email(db_session: AsyncSession) -> None:
    # Default (open): any authenticated user is allowed, no whitelist row needed.
    d = await access_for_email(db_session, "nobody@bi.group")
    assert d.allowed is True
    assert d.is_admin is False
    assert d.reason == "ok"


async def test_access_allowed_and_admin_case_insensitive(db_session: AsyncSession) -> None:
    db_session.add(AllowedUser(email="alice@bi.group", is_admin=True))
    await db_session.commit()
    d = await access_for_email(db_session, "Alice@BI.Group")  # different case
    assert d.allowed is True
    assert d.is_admin is True
    assert d.reason == "ok"


async def test_open_access_ignores_blocked_but_drops_admin(db_session: AsyncSession) -> None:
    db_session.add(AllowedUser(email="bob@bi.group", is_admin=True, blocked=True))
    await db_session.commit()
    d = await access_for_email(db_session, "bob@bi.group")
    assert d.allowed is True  # open mode has no block list
    assert d.is_admin is False  # a blocked row doesn't get admin


async def test_strict_mode_denies_unknown_and_blocked(
    db_session: AsyncSession, monkeypatch
) -> None:
    monkeypatch.setattr(settings, "auth_strict_allowlist", True)
    unknown = await access_for_email(db_session, "ghost@bi.group")
    assert unknown.allowed is False and unknown.reason == "not_whitelisted"

    db_session.add(AllowedUser(email="carol@bi.group", is_admin=True))
    db_session.add(AllowedUser(email="dave@bi.group", blocked=True))
    await db_session.commit()
    ok = await access_for_email(db_session, "carol@bi.group")
    assert ok.allowed is True and ok.is_admin is True
    blocked = await access_for_email(db_session, "dave@bi.group")
    assert blocked.allowed is False and blocked.reason == "blocked"


async def test_seed_allowed_users_seeds_demo_and_is_idempotent(
    db_session: AsyncSession,
) -> None:
    n1 = await seed_allowed_users(db_session)
    assert n1 >= 1
    demo = await access_for_email(db_session, "demo@bi.group")
    assert demo.allowed is True
    assert demo.is_admin is True
    # Second run inserts nothing.
    assert await seed_allowed_users(db_session) == 0
    assert await allowed_users_count(db_session) == n1


# ---------------------------------------------------------------------------
# AuthGate middleware
# ---------------------------------------------------------------------------


def _build_app() -> FastAPI:
    from llm_wiki.api.middleware import AuthGateMiddleware

    app = FastAPI()
    app.add_middleware(AuthGateMiddleware)

    @app.get("/api/v1/protected")
    async def protected() -> dict[str, bool]:
        return {"ok": True}

    @app.get("/api/v1/auth/config")
    async def open_cfg() -> dict[str, bool]:
        return {"open": True}

    @app.get("/healthz")
    @app.get("/health-ams")
    @app.get("/readiness")
    @app.get("/api/healthz")
    @app.get("/api/health-ams")
    @app.get("/api/readiness")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/v1/ops/generations")
    async def ops_stub() -> dict[str, bool]:
        # Реальный роут защищён собственным X-Ops-Token; здесь важно одно —
        # Keycloak-гейт его НЕ трогает (Grafana не умеет OIDC-хендшейк).
        return {"ops": True}

    return app


@pytest_asyncio.fixture
async def gate_client(db_engine, monkeypatch):  # type: ignore[no-untyped-def]
    """Client for an app behind AuthGate, with auth ON and token→email stub.

    Depends on ``db_engine`` so the shared test DB has the tables the real
    ``access_for_email`` (invoked inside the middleware) queries.
    """
    monkeypatch.setattr(settings, "auth_enabled", True)
    # Present the desired email as the bearer token (verify is stubbed).
    import llm_wiki.api.auth as auth_mod

    monkeypatch.setattr(auth_mod, "verify_access_token", lambda t: {"email": t})

    app = _build_app()
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


async def test_gate_open_auth_path_needs_no_token(gate_client: AsyncClient) -> None:
    r = await gate_client.get("/api/v1/auth/config")
    assert r.status_code == 200


async def test_gate_health_probe_open(gate_client: AsyncClient) -> None:
    r = await gate_client.get("/healthz")
    assert r.status_code == 200


async def test_gate_bi_standard_probes_open(gate_client: AsyncClient) -> None:
    # Стандарт BI: health / readiness / health-ams опрашиваются k8s и AMS без
    # каких-либо кредов — 401 здесь читается как «приложение лежит».
    for path in ("/health-ams", "/readiness", "/api/healthz", "/api/health-ams", "/api/readiness"):
        r = await gate_client.get(path)
        assert r.status_code == 200, f"{path} must be open, got {r.status_code}"


async def test_gate_ops_prefix_bypasses_keycloak(gate_client: AsyncClient) -> None:
    # /api/v1/ops/* несёт собственный X-Ops-Token-гейт — Keycloak-гейт его
    # пропускает (иначе Grafana никогда не достучится).
    r = await gate_client.get("/api/v1/ops/generations")
    assert r.status_code == 200


async def test_gate_missing_token_is_401(gate_client: AsyncClient) -> None:
    r = await gate_client.get("/api/v1/protected")
    assert r.status_code == 401


async def test_gate_any_authenticated_email_passes_when_open(gate_client: AsyncClient) -> None:
    # Open by default: a valid token with any email gets in — no whitelist row.
    r = await gate_client.get(
        "/api/v1/protected", headers={"Authorization": "Bearer ghost@bi.group"}
    )
    assert r.status_code == 200
    assert r.json() == {"ok": True}


async def test_gate_allowed_email_passes(
    gate_client: AsyncClient, db_session: AsyncSession
) -> None:
    db_session.add(AllowedUser(email="alice@bi.group", is_admin=False))
    await db_session.commit()
    r = await gate_client.get(
        "/api/v1/protected", headers={"Authorization": "Bearer alice@bi.group"}
    )
    assert r.status_code == 200
    assert r.json() == {"ok": True}


async def test_gate_unknown_email_is_403_in_strict_mode(
    gate_client: AsyncClient, monkeypatch
) -> None:
    monkeypatch.setattr(settings, "auth_strict_allowlist", True)
    r = await gate_client.get(
        "/api/v1/protected", headers={"Authorization": "Bearer ghost@bi.group"}
    )
    assert r.status_code == 403


async def test_gate_blocked_email_is_403_in_strict_mode(
    gate_client: AsyncClient, db_session: AsyncSession, monkeypatch
) -> None:
    monkeypatch.setattr(settings, "auth_strict_allowlist", True)
    db_session.add(AllowedUser(email="bob@bi.group", blocked=True))
    await db_session.commit()
    r = await gate_client.get(
        "/api/v1/protected", headers={"Authorization": "Bearer bob@bi.group"}
    )
    assert r.status_code == 403


async def test_gate_disabled_passes_through(db_engine, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(settings, "auth_enabled", False)
    app = _build_app()
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        r = await ac.get("/api/v1/protected")  # no token, but auth is off
    assert r.status_code == 200
