"""Integration tests for RequestIDMiddleware (LW-17 lite)."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from llm_wiki.api.middleware import RequestIDMiddleware
from llm_wiki.logging_config import configure_logging


def _make_app() -> FastAPI:
    configure_logging()
    app = FastAPI()
    app.add_middleware(RequestIDMiddleware)

    @app.get("/ping")
    def ping() -> dict[str, str]:
        return {"status": "ok"}

    return app


def test_request_id_generated_when_absent() -> None:
    client = TestClient(_make_app())
    resp = client.get("/ping")
    assert resp.status_code == 200
    assert "X-Request-ID" in resp.headers
    assert len(resp.headers["X-Request-ID"]) == 16


def test_request_id_propagated_when_provided() -> None:
    client = TestClient(_make_app())
    resp = client.get("/ping", headers={"X-Request-ID": "client-supplied-id"})
    assert resp.headers["X-Request-ID"] == "client-supplied-id"


def test_different_requests_get_different_ids() -> None:
    client = TestClient(_make_app())
    ids = {client.get("/ping").headers["X-Request-ID"] for _ in range(5)}
    assert len(ids) == 5, "Each request should get a unique request_id"


def test_request_id_is_hex() -> None:
    client = TestClient(_make_app())
    rid = client.get("/ping").headers["X-Request-ID"]
    assert all(c in "0123456789abcdef" for c in rid)
