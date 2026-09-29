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

import asyncio
import time
from collections.abc import Awaitable, Callable

import structlog
from fastapi import HTTPException
from starlette.datastructures import Headers, MutableHeaders
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from llm_wiki.observability import bind_entities, error_fields, new_id, safe_id, trace_scope

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


class RequestIDMiddleware:
    """Measure the whole ASGI response, including streams and unexpected errors."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = Headers(scope=scope)
        request_id = safe_id(headers.get("x-request-id")) or new_id()
        fields = {"request_id": request_id,
                  "operation_id": safe_id(headers.get("x-operation-id")) or request_id,
                  "method": scope["method"], "path": scope["path"]}
        for name in ("run_id", "scenario_id"):
            value = safe_id(headers.get("x-" + name.replace("_", "-")))
            if value:
                fields[name] = value
        scope.setdefault("state", {})["request_id"] = request_id
        started = time.perf_counter()
        status = 500
        response_started = complete = disconnected = streaming = False
        headers_ms = None
        failure = {}

        async def observed_receive() -> Message:
            nonlocal disconnected
            message = await receive()
            if message["type"] == "http.disconnect":
                disconnected = True
            return message

        async def observed_send(message: Message) -> None:
            nonlocal status, response_started, complete, headers_ms, streaming, disconnected
            if message["type"] == "http.response.start":
                status = message["status"]
                response_headers = MutableHeaders(scope=message)
                response_headers["X-Request-ID"] = request_id
                streaming = "text/event-stream" in response_headers.get("content-type", "")
                headers_ms = round((time.perf_counter() - started) * 1000, 2)
                response_started = True
            try:
                await send(message)
            except OSError:
                disconnected = True
                raise
            if message["type"] == "http.response.body" and not message.get("more_body", False):
                complete = True

        with trace_scope(fields) as trace:
            _logger.info("request_started", **fields)
            try:
                await self.app(scope, observed_receive, observed_send)
            except asyncio.CancelledError:
                disconnected = True
                raise
            except Exception as exc:
                failure = error_fields(exc)
                trace.outcome = "failed"
                if not response_started:
                    # Preserve the exception for the server while ensuring its
                    # generic 500 has the same request ID as our terminal log.
                    await JSONResponse({"detail": "Internal Server Error"}, status_code=500)(
                        scope, observed_receive, observed_send)
                raise
            finally:
                route = getattr(scope.get("route"), "path", None)
                outcome = ("failed" if status >= 500 else "rejected" if status >= 400
                           else trace.outcome or "success")
                if disconnected and not complete:
                    outcome = "cancelled"
                elif not failure and streaming and not trace.stream_done and outcome == "success":
                    outcome = "incomplete"
                _logger.info(
                    "request_handled", status_code=status, route=route,
                    duration_ms=round((time.perf_counter() - started) * 1000, 2),
                    response_headers_ms=headers_ms, response_complete=complete,
                    streaming=streaming, outcome=outcome,
                    stream_events=trace.stream_events if streaming else None,
                    stream_done=trace.stream_done if streaming else None,
                    first_event_ms=(round((trace.first_event_at - started) * 1000, 2)
                                    if trace.first_event_at is not None else None),
                    **trace.fields, **failure,
                )


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
        from llm_wiki.api.load_test_auth import access_for_claims

        async with _SessionLocal() as session:
            decision = await access_for_claims(session, claims)
        if not decision.allowed:
            _logger.info("access_denied", email=email, reason=decision.reason)
            return JSONResponse(
                {"detail": "Access is not allowed for this account"}, status_code=403
            )

        request.state.user_email = email
        request.state.user_is_admin = decision.is_admin
        request.state.user_claims = claims
        return await call_next(request)


async def bind_request_entities(request: Request) -> None:
    """Runs after routing so nested AI calls inherit resource identifiers."""
    bind_entities(**request.path_params)
