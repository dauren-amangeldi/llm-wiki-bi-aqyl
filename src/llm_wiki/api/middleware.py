"""HTTP middleware: assign a request_id to every request and propagate it.

The request_id is bound to structlog's contextvars so every log record
emitted while handling the request automatically includes it.  It is also
returned in the ``X-Request-ID`` response header so clients can include it
in bug reports.

If the caller already supplies an ``X-Request-ID`` header (e.g. from a load
balancer or a tracing proxy), that value is reused so the ID is consistent
end-to-end.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from uuid import uuid4

import structlog
from fastapi import HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

_logger = structlog.get_logger(__name__)

# Paths reachable without a token even when auth is enabled: liveness/readiness
# probes (incl. the BI-standard aliases /health-ams and /readiness — k8s probes
# and AMS poll them without any credentials, a 401 here reads as "app down"),
# API docs, and the OIDC login handshake itself (which is how a caller obtains
# a token in the first place).
_OPEN_EXACT = frozenset(
    {
        "/",
        "/health",
        "/healthz",
        "/health-ams",
        "/readyz",
        "/readiness",
        # Те же пробы под /api/ — так их видно через фронт-nginx/ingress
        # (проксируется только /api), и пробу можно вешать на публичный URL.
        "/api/health",
        "/api/healthz",
        "/api/health-ams",
        "/api/readyz",
        "/api/readiness",
        "/docs",
        "/redoc",
        "/openapi.json",
    }
)
# /api/v1/ops/ carries its own X-Ops-Token gate (see api/v1/ops.py) — Grafana
# polls it headlessly and can't do the Keycloak handshake.
_OPEN_PREFIXES = ("/api/v1/auth/", "/api/v1/ops/")


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Generate and propagate a per-request correlation ID.

    For each incoming HTTP request:
    1. Extract ``X-Request-ID`` from headers or generate a new 16-char hex ID.
    2. Bind ``request_id``, ``method``, and ``path`` to structlog contextvars
       so every log line emitted during request handling includes them.
    3. Set the ``X-Request-ID`` response header.
    4. Emit a single ``request_handled`` access-log record after the response.
    5. Clear contextvars so the next request starts clean.
    """

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        request_id = request.headers.get("X-Request-ID") or uuid4().hex[:16]

        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
        )

        try:
            response = await call_next(request)
        finally:
            # Capture context before clearing so it appears on the access log.
            ctx = structlog.contextvars.get_contextvars()
            structlog.contextvars.clear_contextvars()

        response.headers["X-Request-ID"] = request_id

        _logger.info(
            "request_handled",
            status_code=response.status_code,
            **{k: v for k, v in ctx.items() if k != "request_id"},
            request_id=request_id,
        )
        return response


class AuthGateMiddleware(BaseHTTPMiddleware):
    """Enforce Keycloak auth on every protected route.

    When ``settings.auth_enabled`` is off (dev/demo, the default) this is a
    no-op. When on, each request outside the open-list must carry a valid
    Keycloak access token; the caller is then admitted by ``access_for_email``
    — OPEN by default (any authenticated user), or deny-by-default whitelist
    when ``AUTH_STRICT_ALLOWLIST`` is set. Rejected with 401 (missing/invalid
    token) or 403 (denied). This is the single, uniform gate — individual
    routes need not re-check — so upload/wiki/case endpoints are protected too.

    On success the verified identity is stashed on ``request.state``
    (``user_email``, ``user_is_admin``, ``user_claims``) for downstream deps.
    """

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        from llm_wiki.config import settings

        if not settings.auth_enabled:
            return await call_next(request)

        path = request.url.path
        if (
            request.method == "OPTIONS"  # CORS preflight
            or path in _OPEN_EXACT
            or path.startswith(_OPEN_PREFIXES)
        ):
            return await call_next(request)

        from llm_wiki.api.auth import bearer_token, claims_email, verify_access_token

        token = bearer_token(request)
        if not token:
            return JSONResponse({"detail": "Missing bearer token"}, status_code=401)
        try:
            claims = verify_access_token(token)
        except HTTPException as exc:
            return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)

        email = claims_email(claims)

        from llm_wiki.api.deps import _SessionLocal
        from llm_wiki.storage.metadata import access_for_email

        async with _SessionLocal() as session:
            decision = await access_for_email(session, email)
        if not decision.allowed:
            _logger.info("access_denied", email=email, reason=decision.reason)
            return JSONResponse(
                {"detail": "Access is not allowed for this account"}, status_code=403
            )

        request.state.user_email = email
        request.state.user_is_admin = decision.is_admin
        request.state.user_claims = claims
        return await call_next(request)
