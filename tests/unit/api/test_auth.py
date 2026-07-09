"""Unit tests for Keycloak/OIDC token verification + the auth-off fallback."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import jwt as pyjwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import HTTPException

from llm_wiki.api import auth as auth_mod
from llm_wiki.api.deps import get_user_key
from llm_wiki.config import settings


def _keypair() -> rsa.RSAPrivateKey:
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


def _token(priv: rsa.RSAPrivateKey, **overrides: object) -> str:
    claims: dict[str, object] = {
        "iss": settings.keycloak_realm_url,
        "exp": int(time.time()) + 3600,
        "email": "user@bi.group",
        "preferred_username": "user",
    }
    claims.update(overrides)
    return pyjwt.encode(claims, priv, algorithm="RS256")


def _patch_jwks(priv: rsa.RSAPrivateKey):  # type: ignore[no-untyped-def]
    """Patch the JWKS client so it returns *priv*'s public key."""
    fake_key = MagicMock()
    fake_key.key = priv.public_key()
    patcher = patch.object(auth_mod, "_jwks")
    m = patcher.start()
    m.return_value.get_signing_key_from_jwt.return_value = fake_key
    return patcher


# ---------------------------------------------------------------------------
# verify_access_token
# ---------------------------------------------------------------------------


def test_verify_valid_token_returns_claims() -> None:
    priv = _keypair()
    token = _token(priv)
    patcher = _patch_jwks(priv)
    try:
        claims = auth_mod.verify_access_token(token)
    finally:
        patcher.stop()
    assert claims["email"] == "user@bi.group"
    assert auth_mod.claims_email(claims) == "user@bi.group"


def test_verify_rejects_expired_token() -> None:
    priv = _keypair()
    token = _token(priv, exp=int(time.time()) - 10)
    patcher = _patch_jwks(priv)
    try:
        with pytest.raises(HTTPException) as exc:
            auth_mod.verify_access_token(token)
    finally:
        patcher.stop()
    assert exc.value.status_code == 401


def test_verify_rejects_wrong_issuer() -> None:
    priv = _keypair()
    token = _token(priv, iss="https://evil.example/realms/x")
    patcher = _patch_jwks(priv)
    try:
        with pytest.raises(HTTPException) as exc:
            auth_mod.verify_access_token(token)
    finally:
        patcher.stop()
    assert exc.value.status_code == 401


def test_verify_rejects_token_signed_by_other_key() -> None:
    """A token signed by an attacker's key must fail signature verification."""
    real, attacker = _keypair(), _keypair()
    token = _token(attacker)  # signed by the wrong key
    patcher = _patch_jwks(real)  # JWKS serves the real public key
    try:
        with pytest.raises(HTTPException) as exc:
            auth_mod.verify_access_token(token)
    finally:
        patcher.stop()
    assert exc.value.status_code == 401


# ---------------------------------------------------------------------------
# bearer_token helper
# ---------------------------------------------------------------------------


def test_bearer_token_parsing() -> None:
    req = MagicMock()
    req.headers = {"Authorization": "Bearer abc.def.ghi"}
    assert auth_mod.bearer_token(req) == "abc.def.ghi"

    req.headers = {"Authorization": "Basic xxx"}
    assert auth_mod.bearer_token(req) is None

    req.headers = {}
    assert auth_mod.bearer_token(req) is None


# ---------------------------------------------------------------------------
# get_user_key — auth on/off
# ---------------------------------------------------------------------------


def test_get_user_key_uses_header_when_auth_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "auth_enabled", False)
    req = MagicMock()
    req.headers = {"X-User-Email": "a@b.c"}
    assert get_user_key(req) == "a@b.c"


def test_get_user_key_requires_bearer_when_auth_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "auth_enabled", True)
    req = MagicMock()
    req.headers = {}  # no Authorization header
    with pytest.raises(HTTPException) as exc:
        get_user_key(req)
    assert exc.value.status_code == 401


def test_get_user_key_reads_email_from_token_when_auth_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "auth_enabled", True)
    priv = _keypair()
    token = _token(priv, email="jwt-user@bi.group")
    req = MagicMock()
    req.headers = {"Authorization": f"Bearer {token}"}
    patcher = _patch_jwks(priv)
    try:
        assert get_user_key(req) == "jwt-user@bi.group"
    finally:
        patcher.stop()
