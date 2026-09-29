"""Real token signatures + isolated PostgreSQL + complete HTTP auth gate."""

import time

import jwt
import pytest
from fastapi import HTTPException
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import async_sessionmaker
from starlette.requests import Request

from llm_wiki.api import auth, deps, load_test_auth as lt
from llm_wiki.config import Settings, settings
from llm_wiki.storage.metadata import AllowedUser, User, ensure_column_migrations

LOGIN_SECRET = "test-login-secret-" + "a" * 32
SIGNING_SECRET = "test-signing-secret-" + "b" * 32


@pytest.fixture
def enabled(monkeypatch):
    values = dict(
        auth_enabled=True, load_test_auth_enabled=True, app_environment="test",
        public_base_url="https://aqyl.test.bi.group", load_test_login_secret=LOGIN_SECRET,
        load_test_signing_secret=SIGNING_SECRET, load_test_user_count=2,
        load_test_token_ttl_s=900,
    )
    for key, value in values.items():
        monkeypatch.setattr(settings, key, value)
    return values


@pytest.fixture
async def api(enabled, db_engine, db_session, monkeypatch):
    from llm_wiki.main import app

    monkeypatch.setattr(deps, "_SessionLocal", async_sessionmaker(db_engine, expire_on_commit=False))
    await lt.seed_load_test_users(db_session)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="https://aqyl.test.bi.group") as client:
        yield client


async def login(api, email=None, secret=LOGIN_SECRET):
    return await api.post(
        "/api/v1/auth/load-test/token", json={"email": email or lt.account_email(1)},
        headers={"X-Load-Test-Secret": secret, "X-Request-ID": "load-test-auth-test"},
    )


async def test_login_real_gate_profile_and_read_routes(api):
    response = await login(api)
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-request-id"] == "load-test-auth-test"
    payload = response.json()
    assert payload["expires_in"] == 900
    token = payload["access_token"]
    assert auth.verify_access_token(token)["email"] == lt.account_email(1)
    headers = {"Authorization": "Bearer " + token, "X-User-Email": "admin@bi.group"}
    me = await api.get("/api/v1/auth/me", headers=headers)
    assert me.status_code == 200
    assert me.json()["email"] == lt.account_email(1)
    assert me.json()["role"] == "employee"
    for path in ("/api/v1/cases?limit=200&offset=0", "/api/v1/documents", "/api/v1/tags"):
        response = await api.get(path, headers=headers)
        assert response.status_code == 200
        assert isinstance(response.json(), list)


@pytest.mark.parametrize("field,value", [
    ("load_test_auth_enabled", False), ("public_base_url", "https://aqyl.bi.group"),
    ("public_base_url", ""), ("auth_enabled", False),
    ("load_test_login_secret", ""), ("load_test_signing_secret", ""),
])
async def test_disabled_environment_never_issues_or_verifies(api, monkeypatch, field, value):
    token = lt.issue_load_test_token(lt.account_email(1))
    monkeypatch.setattr(settings, field, value)
    assert (await login(api)).status_code == 404
    with pytest.raises(HTTPException) as exc:
        auth.verify_access_token(token)
    assert exc.value.status_code == 401


@pytest.mark.parametrize("secret", ["", "wrong-secret"])
async def test_login_requires_secret(api, secret):
    assert (await login(api, secret=secret)).status_code == 401


@pytest.mark.parametrize("email", [
    "employee@bi.group", "demo@bi.group", "loadtest-0000@aqyl.test.invalid",
    "loadtest-1001@aqyl.test.invalid", "loadtest-0003@aqyl.test.invalid",
])
async def test_login_denies_non_test_or_non_whitelisted_identity(api, db_session, email):
    if not email.startswith("loadtest-"):
        db_session.add(AllowedUser(email=email, load_test_enabled=True, is_admin=True))
        await db_session.commit()
    assert (await login(api, email)).status_code == 403


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("change", ["blocked", "disabled", "removed"])
async def test_revocation_applies_to_already_issued_tokens(api, db_session, monkeypatch, strict, change):
    monkeypatch.setattr(settings, "auth_strict_allowlist", strict)
    token = (await login(api)).json()["access_token"]
    row = await db_session.get(AllowedUser, lt.account_email(1))
    if change == "removed":
        await db_session.delete(row)
    elif change == "blocked":
        row.blocked = True
    else:
        row.load_test_enabled = False
    await db_session.commit()
    headers = {"Authorization": "Bearer " + token}
    assert (await login(api)).status_code == 403
    assert (await api.get("/api/v1/cases", headers=headers)).status_code == 403
    assert (await api.get("/api/v1/auth/me", headers=headers)).status_code == 403


async def test_load_test_token_never_inherits_admin(api, db_session):
    row = await db_session.get(AllowedUser, lt.account_email(1))
    row.is_admin = True
    db_session.add(User(id=row.email, name="Old admin", role="admin"))
    await db_session.commit()
    token = (await login(api)).json()["access_token"]
    req = Request({"type": "http", "headers": [(b"authorization", ("Bearer " + token).encode())]})
    current = await deps.get_current_user(req, db_session)
    assert current.role == "employee"
    response = await api.get("/api/v1/auth/me", headers={"Authorization": "Bearer " + token})
    assert response.json()["role"] == "employee"


@pytest.mark.parametrize("change", ["expired", "audience", "issuer", "email", "subject", "missing", "ttl", "signature", "algorithm"])
def test_forged_or_invalid_tokens_rejected(enabled, change):
    token = lt.issue_load_test_token(lt.account_email(1))
    claims = lt.verify_load_test_token(token)
    key, algorithm = SIGNING_SECRET, "HS256"
    if change == "expired":
        claims.update(iat=int(time.time()) - 1000, nbf=int(time.time()) - 1000, exp=int(time.time()) - 1)
    elif change == "audience":
        claims["aud"] = "production"
    elif change == "issuer":
        claims["iss"] = "attacker"
    elif change == "email":
        claims["email"] = claims["sub"] = "admin@bi.group"
    elif change == "subject":
        claims["sub"] = lt.account_email(2)
    elif change == "missing":
        del claims["exp"]
    elif change == "ttl":
        claims["exp"] += 9999
    elif change == "signature":
        key = LOGIN_SECRET  # knowing the login secret does not permit signing tokens
    elif change == "algorithm":
        algorithm = "HS384"
    forged = jwt.encode(claims, key, algorithm=algorithm, headers={"kid": lt.TOKEN_KID})
    with pytest.raises(HTTPException) as exc:
        auth.verify_access_token(forged)
    assert exc.value.status_code == 401


def test_signing_key_rotation_revokes_tokens(enabled, monkeypatch):
    token = lt.issue_load_test_token(lt.account_email(1))
    monkeypatch.setattr(settings, "load_test_signing_secret", "c" * 64)
    with pytest.raises(HTTPException):
        auth.verify_access_token(token)


async def test_startup_seed_is_idempotent_and_preserves_revocation(enabled, db_session, monkeypatch):
    assert await lt.seed_load_test_users(db_session) == 2
    row = await db_session.get(AllowedUser, lt.account_email(1))
    row.blocked = True
    row.load_test_enabled = False
    await db_session.commit()
    assert await lt.seed_load_test_users(db_session) == 0
    await db_session.refresh(row)
    assert row.blocked and not row.load_test_enabled
    monkeypatch.setattr(settings, "load_test_user_count", 3)
    assert await lt.seed_load_test_users(db_session) == 1
    monkeypatch.setattr(settings, "public_base_url", "https://aqyl.bi.group")
    monkeypatch.setattr(settings, "load_test_user_count", 4)
    assert await lt.seed_load_test_users(db_session) == 0
    assert len((await db_session.scalars(select(AllowedUser))).all()) == 3


async def test_migration_existing_table_is_idempotent(db_engine):
    async with db_engine.begin() as conn:
        await conn.execute(text("ALTER TABLE allowed_users DROP COLUMN load_test_enabled"))
        await conn.execute(text("INSERT INTO allowed_users (email, is_admin, blocked, created_at) VALUES ('old@bi.group', true, false, now())"))
        await ensure_column_migrations(conn)
        await ensure_column_migrations(conn)
        row = (await conn.execute(text("SELECT is_admin, load_test_enabled FROM allowed_users WHERE email='old@bi.group'"))).one()
        assert tuple(row) == (True, False)


@pytest.mark.parametrize("overrides", [
    {"public_base_url": "https://aqyl.bi.group"},
    {"auth_enabled": False}, {"load_test_signing_secret": "short"},
    {"load_test_signing_secret": LOGIN_SECRET},
    {"load_test_login_secret": ""}, {"load_test_signing_secret": ""},
])
def test_misconfigured_enabled_mode_starts_with_load_test_access_disabled(enabled, overrides, monkeypatch):
    values = {**enabled, **overrides}
    values["PUBLIC_BASE_URL"] = values.pop("public_base_url")
    configured = Settings(_env_file=None, **values)
    assert not configured.load_test_auth_active
    monkeypatch.setattr(lt, "settings", configured)
    with pytest.raises(HTTPException) as exc:
        lt.require_load_test_login(LOGIN_SECRET)
    assert exc.value.status_code == 404
    with pytest.raises(HTTPException):
        lt.issue_load_test_token(lt.account_email(1))


def test_disabled_default_and_valid_test_configuration(enabled):
    assert not Settings(_env_file=None, load_test_auth_enabled=False).load_test_auth_active
    values = {**enabled}
    values["PUBLIC_BASE_URL"] = values.pop("public_base_url")
    assert Settings(_env_file=None, **values).load_test_auth_active


@pytest.mark.parametrize("environment", [None, "production", "development", "test"])
def test_temporary_environment_waiver_keeps_test_domain_required(enabled, monkeypatch, environment):
    monkeypatch.delenv("APP_ENVIRONMENT", raising=False)
    values = {**enabled}
    values.pop("app_environment")
    if environment is not None:
        values["app_environment"] = environment
    values["PUBLIC_BASE_URL"] = values.pop("public_base_url")
    configured = Settings(_env_file=None, **values)
    monkeypatch.setattr(lt, "settings", configured)
    assert configured.load_test_auth_active
    lt.require_load_test_login(LOGIN_SECRET)
    token = lt.issue_load_test_token(lt.account_email(1))
    assert lt.verify_load_test_token(token)["email"] == lt.account_email(1)
    configured.public_base_url = "https://aqyl.bi.group"
    assert not configured.load_test_auth_active
    with pytest.raises(HTTPException):
        lt.verify_load_test_token(token)
