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
# The refresh token lives in an HttpOnly cookie — never in the URL fragment or JS,
# so a stolen access token (short-lived) can't be turned into a lasting session.
# Scoped to /api/v1/auth so it's only sent to /auth/refresh and /auth/logout.
_REFRESH_COOKIE = "kc_refresh"
_REFRESH_COOKIE_PATH = "/api/v1/auth"


def _set_refresh_cookie(resp: RedirectResponse | JSONResponse, token: str, *, secure: bool) -> None:
    resp.set_cookie(
        _REFRESH_COOKIE,
        token,
        max_age=settings.keycloak_refresh_cookie_max_age_s,
        httponly=True,
        samesite="lax",  # blocks cross-site POST → CSRF-safe for /auth/refresh
        secure=secure,
        path=_REFRESH_COOKIE_PATH,
    )


def _clear_refresh_cookie(resp: RedirectResponse | JSONResponse) -> None:
    resp.delete_cookie(_REFRESH_COOKIE, path=_REFRESH_COOKIE_PATH)


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

    payload = r.json()
    access_token = str(payload.get("access_token", ""))
    # The refresh token stays server-side in an HttpOnly cookie (see below); the
    # SPA never sees it and calls /auth/refresh to rotate the access token.
    refresh_token = str(payload.get("refresh_token", ""))
    # Keep the callback response headers small: ONLY the access token rides in the
    # fragment. Carrying the id_token here too, on top of the refresh Set-Cookie,
    # pushed the response headers past the ingress proxy_buffer_size → nginx
    # "upstream sent too big header" → 502 on a real login. The id_token only
    # enabled the silent-logout hint; drop it here — the SPA picks up a fresh
    # id_token from /auth/refresh (stored as bi_id_token), so silent logout still
    # works once the first refresh has run; until then Keycloak may show its page.
    frag = f"access_token={access_token}"
    base = _public_base(request)
    resp = RedirectResponse(f"{base}/#{frag}", status_code=307)
    resp.delete_cookie(_STATE_COOKIE)
    if refresh_token:
        _set_refresh_cookie(resp, refresh_token, secure=base.startswith("https"))
    return resp


@router.post("/auth/refresh")
async def auth_refresh(request: Request) -> JSONResponse:
    """Swap the HttpOnly refresh cookie for a fresh access token (silent refresh).

    The SPA calls this shortly before its short-lived access token expires (and
    once reactively on a 401), so users keep a seamless session without a full
    Keycloak re-login every few minutes. The refresh token never leaves the
    server side. On an expired/invalid refresh token we return 401 and clear the
    cookie, and the SPA falls back to a real login."""
    if not settings.auth_enabled:
        raise HTTPException(status_code=404, detail="Auth is disabled")

    refresh_token = request.cookies.get(_REFRESH_COOKIE)
    if not refresh_token:
        raise HTTPException(status_code=401, detail="No refresh token")

    data = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": settings.keycloak_client_id,
        "client_secret": settings.keycloak_client_secret,
    }
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.post(settings.keycloak_token_url, data=data)
    except httpx.HTTPError as exc:
        logger.warning("oidc_refresh_error", error=str(exc))
        raise HTTPException(status_code=502, detail="Keycloak unreachable") from exc

    if r.status_code != 200:
        # Refresh token expired/revoked → drop the cookie and make the SPA re-login.
        logger.info("oidc_refresh_failed", status=r.status_code)
        resp = JSONResponse({"detail": "Refresh failed"}, status_code=401)
        _clear_refresh_cookie(resp)
        return resp

    payload = r.json()
    resp = JSONResponse(
        {
            "access_token": str(payload.get("access_token", "")),
            "id_token": str(payload.get("id_token", "")),
        }
    )
    # Keycloak rotates the refresh token by default — persist the new one.
    new_refresh = str(payload.get("refresh_token", ""))
    if new_refresh:
        _set_refresh_cookie(resp, new_refresh, secure=_public_base(request).startswith("https"))
    return resp


@router.get("/auth/logout")
async def auth_logout(request: Request, id_token_hint: str = "") -> RedirectResponse:
    """RP-initiated logout: end the Keycloak SSO session, then return to the app.

    Clearing only the app's local token is not enough — the Keycloak SSO cookie
    survives, so the next /auth/login silently re-authenticates (the "logout
    loops back in" bug). Redirecting through the end-session endpoint kills that
    session; Keycloak then sends the browser to ``post_logout_redirect_uri``
    (must be registered on the client), where the SPA shows the login form.

    ``id_token_hint`` (the login id_token, forwarded by the SPA) is what lets
    Keycloak log out SILENTLY. Without it Keycloak shows a logout-confirmation
    page — which this realm's theme renders as a blank white screen, and the
    session isn't ended until confirmed (hence "logout → white screen, then
    reopening logs me back in").
    """
    base = _public_base(request)
    if not settings.auth_enabled:
        return RedirectResponse(base + "/", status_code=307)
    params = {
        "client_id": settings.keycloak_client_id,
        "post_logout_redirect_uri": base + "/",
    }
    if id_token_hint:
        params["id_token_hint"] = id_token_hint
    resp = RedirectResponse(
        f"{settings.keycloak_logout_url}?{urlencode(params)}", status_code=307
    )
    resp.delete_cookie(_STATE_COOKIE)
    _clear_refresh_cookie(resp)
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
