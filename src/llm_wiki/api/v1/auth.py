"""OIDC login endpoints for Keycloak (confidential client, backend-mediated).

Flow (Authorization Code):
  1. Frontend sends the browser to ``GET /api/v1/auth/login``.
  2. We redirect to Keycloak; the user logs in there.
  3. Keycloak redirects back to ``GET /api/v1/auth/callback?code=...``.
  4. We exchange the code + client_secret for tokens (server-side — the secret
     never touches the browser), then redirect to the SPA with the access token
     in the URL fragment (fragments are not sent to servers).
  5. The SPA stores it and sends ``Authorization: Bearer <jwt>``; the API
     verifies it (see ``api/auth.py``).

Inert unless ``settings.auth_enabled`` is true.
"""

from __future__ import annotations

import secrets
from urllib.parse import urlencode

import httpx
import structlog
from fastapi import Depends, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse
from sqlalchemy.ext.asyncio import AsyncSession

from llm_wiki.api.auth import (
    bearer_token,
    claims_email,
    claims_given_name,
    claims_title,
    verify_access_token,
)
from llm_wiki.api.deps import get_db
from llm_wiki.api.v1 import router
from llm_wiki.config import settings
from llm_wiki.storage.metadata import access_for_email

logger = structlog.get_logger(__name__)

_STATE_COOKIE = "kc_state"


def _public_base(request: Request) -> str:
    """Browser-facing origin of this app (config override, else request base)."""
    if settings.public_base_url:
        return settings.public_base_url.rstrip("/")
    return str(request.base_url).rstrip("/")


def _redirect_uri(request: Request) -> str:
    return f"{_public_base(request)}/api/v1/auth/callback"


@router.get("/auth/config")
async def auth_config() -> JSONResponse:
    """Tell the frontend whether auth is enabled and where to start login."""
    return JSONResponse(
        {"enabled": settings.auth_enabled, "login_url": "/api/v1/auth/login"}
    )


@router.get("/auth/login")
async def auth_login(request: Request) -> RedirectResponse:
    """Redirect the browser to the Keycloak login page."""
    if not settings.auth_enabled:
        raise HTTPException(status_code=404, detail="Auth is disabled")

    state = secrets.token_urlsafe(24)
    redirect_uri = _redirect_uri(request)
    params = {
        "client_id": settings.keycloak_client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": "openid email profile",
        "state": state,
    }
    resp = RedirectResponse(f"{settings.keycloak_auth_url}?{urlencode(params)}", status_code=307)
    # Short-lived CSRF state, checked in the callback.
    resp.set_cookie(
        _STATE_COOKIE,
        state,
        max_age=600,
        httponly=True,
        samesite="lax",
        secure=redirect_uri.startswith("https"),
    )
    return resp


@router.get("/auth/callback")
async def auth_callback(request: Request, code: str = "", state: str = "") -> RedirectResponse:
    """Exchange the authorization code for tokens and hand it to the SPA."""
    if not settings.auth_enabled:
        raise HTTPException(status_code=404, detail="Auth is disabled")

    expected = request.cookies.get(_STATE_COOKIE)
    if not code or not state or state != expected:
        raise HTTPException(status_code=400, detail="Invalid OAuth state or missing code")

    data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": _redirect_uri(request),
        "client_id": settings.keycloak_client_id,
        "client_secret": settings.keycloak_client_secret,
    }
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.post(settings.keycloak_token_url, data=data)
    except httpx.HTTPError as exc:
        logger.warning("oidc_token_exchange_error", error=str(exc))
        raise HTTPException(status_code=502, detail="Keycloak unreachable") from exc

    if r.status_code != 200:
        logger.warning("oidc_token_exchange_failed", status=r.status_code, body=r.text[:200])
        raise HTTPException(status_code=401, detail="Token exchange failed")

    access_token = str(r.json().get("access_token", ""))
    # Fragment (#…) stays client-side — the SPA reads it, stores it, and cleans
    # the URL; it is never sent to any server.
    resp = RedirectResponse(f"{_public_base(request)}/#access_token={access_token}", status_code=307)
    resp.delete_cookie(_STATE_COOKIE)
    return resp


@router.get("/auth/logout")
async def auth_logout(request: Request) -> RedirectResponse:
    """RP-initiated logout: end the Keycloak SSO session, then return to the app.

    Clearing only the app's local token is not enough — the Keycloak SSO cookie
    survives, so the next /auth/login silently re-authenticates (the "logout
    loops back in" bug). Redirecting through the end-session endpoint kills that
    session; Keycloak then sends the browser to ``post_logout_redirect_uri``
    (must be registered on the client), where the SPA shows the login form.
    """
    base = _public_base(request)
    if not settings.auth_enabled:
        return RedirectResponse(base + "/", status_code=307)
    params = {
        "client_id": settings.keycloak_client_id,
        "post_logout_redirect_uri": base + "/",
    }
    resp = RedirectResponse(
        f"{settings.keycloak_logout_url}?{urlencode(params)}", status_code=307
    )
    resp.delete_cookie(_STATE_COOKIE)
    return resp


@router.get("/auth/me")
async def auth_me(
    request: Request, session: AsyncSession = Depends(get_db)
) -> JSONResponse:
    """Return the verified caller's identity + role (for the SPA session).

    Also runs ``access_for_email`` here (this endpoint is outside the middleware
    gate). Access is OPEN by default (any authenticated user); under
    ``AUTH_STRICT_ALLOWLIST`` a valid-but-not-allowed account gets a clear 403
    the SPA renders as «Нет доступа». The admin role comes from
    ``allowed_users.is_admin``, not the Keycloak realm roles.
    """
    token = bearer_token(request)
    if not token:
        raise HTTPException(status_code=401, detail="Missing bearer token")
    claims = verify_access_token(token)
    email = claims_email(claims)
    decision = await access_for_email(session, email)
    if not decision.allowed:
        raise HTTPException(
            status_code=403, detail="Access is not allowed for this account"
        )
    return JSONResponse(
        {
            "email": email,
            "name": claims.get("name") or claims.get("preferred_username") or email,
            # given_name → SPA greeting; title (job title) → LLM personalization.
            # Both are best-effort ("" when the token/mapper doesn't carry them).
            "given_name": claims_given_name(claims),
            "title": claims_title(claims),
            "role": "admin" if decision.is_admin else "employee",
        }
    )
