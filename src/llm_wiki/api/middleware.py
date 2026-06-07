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
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

_logger = structlog.get_logger(__name__)


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
