"""Keycloak / OIDC access-token verification.

The API trusts an access token only after verifying its **signature** against
the realm's public keys (JWKS) plus the issuer and expiry — decoding alone is
not enough (a JWT is just base64 and could be forged).

Everything here is inert unless ``settings.auth_enabled`` is true; the OIDC
login/callback endpoints live in ``api/v1/auth.py``.
"""

from __future__ import annotations

from typing import Any

import jwt
import structlog
from fastapi import HTTPException, Request
from jwt import PyJWKClient

from llm_wiki.config import settings

logger = structlog.get_logger(__name__)

_jwks_client: PyJWKClient | None = None


def _jwks() -> PyJWKClient:
    """Lazily create the cached JWKS client for the configured realm."""
    global _jwks_client
    if _jwks_client is None:
        _jwks_client = PyJWKClient(settings.keycloak_jwks_url)
    return _jwks_client


def verify_access_token(token: str) -> dict[str, Any]:
    """Verify a Keycloak access-token JWT and return its claims.

    Checks the RS256 signature (via JWKS), the issuer, and expiry. Keycloak
    sets the access-token ``aud`` to ``account`` (not the client id), so
    audience is intentionally not verified.

    Raises:
        HTTPException 401: If the token is missing a key, forged, or expired.
    """
    try:
        signing_key = _jwks().get_signing_key_from_jwt(token)
        return jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            issuer=settings.keycloak_realm_url,
            options={"verify_aud": False},
        )
    except Exception as exc:  # noqa: BLE001
        logger.info("token_rejected", error=str(exc))
        raise HTTPException(status_code=401, detail="Invalid or expired token") from exc


def bearer_token(request: Request) -> str | None:
    """Return the bearer token from the Authorization header, or None."""
    header = request.headers.get("Authorization", "")
    if header[:7].lower() == "bearer ":
        return header[7:].strip() or None
    return None


def claims_email(claims: dict[str, Any]) -> str:
    """Best-effort user identity (email) from token claims."""
    return str(
        claims.get("email")
        or claims.get("preferred_username")
        or claims.get("sub")
        or "unknown"
    )


def require_email(request: Request) -> str:
    """Verify the request's bearer token and return the caller's email.

    Raises HTTPException 401 when the token is missing or invalid.
    """
    token = bearer_token(request)
    if not token:
        raise HTTPException(status_code=401, detail="Missing bearer token")
    return claims_email(verify_access_token(token))
