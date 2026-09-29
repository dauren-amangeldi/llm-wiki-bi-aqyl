"""Offline regressions for the load-test tracing contract. No DB/broker/provider."""
import asyncio
import json
import io
from datetime import datetime, timezone
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, patch

import httpx
import openai
import pytest
import structlog
from celery import Celery, Task
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient

from llm_wiki.api.middleware import AuthGateMiddleware, RequestIDMiddleware
from llm_wiki.config import settings
from llm_wiki.llm.client import LLMClient
from llm_wiki.llm.telemetry import estimate_cost, media_call, response_usage, write_usage
from llm_wiki.logging_config import configure_logging
from llm_wiki.observability import context, mark_outcome, observe_sse, trace_scope
from llm_wiki.orchestrator.observed_task import ObservedTask
from llm_wiki.quality.budget import compute_budget_snapshot


@pytest.fixture
def log_output(monkeypatch):
    buffer = io.StringIO()
    factory = structlog.PrintLoggerFactory
    monkeypatch.setattr(structlog, "PrintLoggerFactory", lambda **kwargs: factory(file=buffer))
    return buffer


@pytest.fixture(autouse=True)
def offline(monkeypatch, log_output, tmp_path):
    import socket
    def reject(*args, **kwargs):
        raise AssertionError("This test must not contact a network service")
    monkeypatch.setattr(socket.socket, "connect", reject)
    monkeypatch.setattr(settings, "data_dir", tmp_path / "isolated-data")
    monkeypatch.setattr(settings, "auth_enabled", False)
    monkeypatch.setattr(settings, "daily_cost_limit_usd", None)
    monkeypatch.setattr(settings, "daily_token_limit", None)
    configure_logging()


def records(log_output, event=None):
    rows = [json.loads(line) for line in log_output.getvalue().splitlines() if line.startswith("{")]
    log_output.seek(0)
    log_output.truncate()
    return [r for r in rows if event is None or r.get("event") == event]


def app():
    result = FastAPI()
    result.add_middleware(AuthGateMiddleware)
    result.add_middleware(RequestIDMiddleware)
    return result


@pytest.mark.parametrize("status", [200, 401, 403, 404, 409, 422, 429, 503])
def test_http_terminal_log_and_headers(status, log_output):
    api = app()
    @api.get("/cases/{case_id}")
    async def route(case_id: str):
        if status != 200:
            raise HTTPException(status, "public error")
        return {"ok": True}
    response = TestClient(api).get("/cases/a", headers={"X-Request-ID": "req-a", "X-Run-ID": "run-a"})
    assert response.status_code == status
    assert response.headers["X-Request-ID"] == "req-a"
    [log] = records(log_output, "request_handled")
    assert log["route"] == "/cases/{case_id}"
    assert log["run_id"] == "run-a"
    assert log["duration_ms"] >= 0
    assert log["outcome"] == ("failed" if status >= 500 else "rejected" if status >= 400 else "success")


def test_unhandled_exception_logs_once_and_returns_request_id(log_output):
    api = app()
    @api.get("/broken")
    async def broken():
        raise ValueError("sk-thismustnotappear")
    response = TestClient(api, raise_server_exceptions=False).get("/broken")
    assert response.status_code == 500
    [log] = records(log_output, "request_handled")
    assert log["request_id"] == response.headers["X-Request-ID"]
    assert log["error_type"] == "ValueError"
    assert "sk-thismustnotappear" not in json.dumps(log)


@pytest.mark.parametrize("payload,outcome", [({"done": True}, "success"),
    ({"error": "unavailable"}, "failed"), ({"done": True, "timeout": True}, "failed"),
    ({"done": True, "status": "FAILED"}, "failed"), ({"done": True, "refusal": True}, "rejected"),
    ({"done": True, "content": {"failed": True}}, "degraded"), ({"status": "working"}, "incomplete")])
def test_sse_logs_actual_end_and_semantic_outcome(payload, outcome, log_output):
    api = app()
    @api.get("/stream")
    async def stream():
        async def chunks():
            await asyncio.sleep(0.02)
            observe_sse(payload)
            yield "data: " + json.dumps(payload) + "\n\n"
        return StreamingResponse(chunks(), media_type="text/event-stream")
    assert TestClient(api).get("/stream").status_code == 200
    [log] = records(log_output, "request_handled")
    assert log["outcome"] == outcome
    assert log["stream_events"] == 1
    assert log["duration_ms"] >= 15
    assert log["first_event_ms"] >= log["response_headers_ms"]


async def test_cancelled_stream_has_terminal_log(log_output):
    async def disconnected(scope, receive, send):
        await send({"type": "http.response.start", "status": 200,
                    "headers": [(b"content-type", b"text/event-stream")]})
        raise asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        await RequestIDMiddleware(disconnected)(
            {"type": "http", "method": "GET", "path": "/stream", "headers": []},
            AsyncMock(), AsyncMock())
    [log] = records(log_output, "request_handled")
    assert log["outcome"] == "cancelled" and not log["response_complete"]


async def test_concurrent_request_contexts_do_not_mix(log_output):
    api = app()
    @api.get("/check")
    async def route():
        await asyncio.sleep(0.01)
        structlog.get_logger().info("nested")
        return context()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api), base_url="http://test") as client:
        results = await asyncio.gather(*[client.get("/check", headers={"X-Run-ID": f"run-{n}"}) for n in range(2)])
    assert [r.json()["run_id"] for r in results] == ["run-0", "run-1"]
    assert not context()
    assert {r["run_id"] for r in records(log_output, "nested")} == {"run-0", "run-1"}


def test_invalid_correlation_header_is_not_logged(log_output):
    api = app()
    response = TestClient(api).get("/missing", headers={"X-Request-ID": "x" * 1024})
    assert len(response.headers["X-Request-ID"]) == 16
    assert "x" * 1024 not in json.dumps(records(log_output))


def test_task_publish_execution_and_child_propagation(log_output):
    celery = Celery("observed", broker="memory://", task_cls=ObservedTask)
    @celery.task(name="test.process")
    def process(file_id):
        structlog.contextvars.clear_contextvars()  # existing task bodies do this
        structlog.get_logger().info("inside_task")
        return {"status": "failed"}
    with trace_scope({"request_id": "r1", "run_id": "run1", "scenario_id": "upload"}):
        with patch.object(Task, "apply_async", return_value=NS(id="job")) as publish:
            process.apply_async(args=("file1",), task_id="job")
    headers = publish.call_args.kwargs["headers"]
    assert headers["observability"]["run_id"] == "run1"
    process.push_request(id="job", headers=headers, delivery_info={"routing_key": "ingest"}, retries=0)
    try:
        assert process("file1")["status"] == "failed"
    finally:
        process.pop_request()
    rows = records(log_output)
    for name in ("task_queued", "task_execution_started", "inside_task", "task_execution_finished"):
        log = next(r for r in rows if r["event"] == name)
        assert log["request_id"] == "r1" and log["run_id"] == "run1"
        assert log["task_id"] == "job" and log["file_id"] == "file1"
    final = next(r for r in rows if r["event"] == "task_execution_finished")
    assert final["outcome"] == "failed" and final["queue_wait_ms"] >= 0
    assert not context()


def test_publish_failure_is_visible_and_reraised(log_output):
    celery = Celery("observed", broker="memory://", task_cls=ObservedTask)
    @celery.task
    def work(case_id):
        return None
    with patch.object(Task, "apply_async", side_effect=ConnectionError("broker down")):
        with pytest.raises(ConnectionError):
            work.delay("case1")
    [log] = records(log_output, "task_enqueue_failed")
    assert log["case_id"] == "case1" and log["error_type"] == "ConnectionError"


def test_usage_keeps_tokens_when_cost_unknown_and_propagates_context(tmp_path, log_output):
    path = tmp_path / "ledger.jsonl"
    with trace_scope({"run_id": "run1", "task_id": "job1", "artifact_id": "art1"}):
        write_usage({"model": "unpriced-model", "agent_type": "artifact", "input_tokens": 100,
                     "output_tokens": 50, "cached_input_tokens": 0}, path)
    [log] = records(log_output, "llm_usage")
    saved = json.loads(path.read_text())
    assert log["call_id"] == saved["call_id"]
    assert log["cost_usd"] is None and log["cost_status"] == "unknown_model"
    assert log["artifact_id"] == "art1" and log["task_id"] == "job1"
    snapshot = compute_budget_snapshot(path)
    assert snapshot.tokens_today == 150 and snapshot.unpriced_calls_today == 1


def test_cached_pricing_and_reasoning_not_counted_twice(monkeypatch):
    monkeypatch.setattr(settings, "price_table", {"test": {"input": 2, "cached_input": 0.2, "output": 8}})
    response = NS(usage=NS(prompt_tokens=100, completion_tokens=50,
                          prompt_tokens_details=NS(cached_tokens=80),
                          completion_tokens_details=NS(reasoning_tokens=30)))
    usage = response_usage(response)
    assert usage["output_tokens"] == 50 and usage["reasoning_tokens"] == 30
    cost, status = estimate_cost("test", usage)
    assert cost == pytest.approx((20*2 + 80*0.2 + 50*8)/1e6)
    assert status == "estimated"


@pytest.mark.parametrize("kind", ["ocr", "image", "transcription", "embed"])
def test_media_has_usage_and_terminal_event(kind, log_output):
    with trace_scope({"request_id": "r1"}):
        with media_call("unknown-model", kind, "file1") as trace:
            trace["response"] = NS(usage=NS(input_tokens=100, output_tokens=20), duration=5)
    rows = records(log_output)
    [usage] = [r for r in rows if r["event"] == "llm_usage"]
    assert usage["input_tokens"] == 100 and usage["audio_seconds"] == 5
    assert usage["cost_usd"] is None and usage["request_id"] == "r1"
    assert sum(r["event"] == "ai_call_finished" for r in rows) == 1


async def test_failed_completion_has_terminal_error_without_provider_body(tmp_path, log_output):
    with patch("llm_wiki.llm.client.openai.AsyncOpenAI"):
        client = LLMClient()
    client._usage_log_path = tmp_path / "usage.log"
    error = openai.AuthenticationError("bad sk-testsecret", response=httpx.Response(401, request=httpx.Request("POST", "https://test")), body=None)
    client._call_provider = AsyncMock(side_effect=error)
    with pytest.raises(openai.AuthenticationError):
        await client.complete("private prompt", "system", "file1", "writer")
    rows = records(log_output)
    [log] = [r for r in rows if r["event"] == "ai_call_finished"]
    assert log["outcome"] == "failed" and log["attempts"] == 1
    assert log["provider_status_code"] == 401
    assert "private prompt" not in json.dumps(rows) and "sk-testsecret" not in json.dumps(rows)
    assert not client._usage_log_path.exists()


async def test_ledger_failure_does_not_repeat_paid_completion(log_output):
    with patch("llm_wiki.llm.client.openai.AsyncOpenAI"):
        client = LLMClient()
    client._call_provider = AsyncMock(return_value=("result", 100, 50, 0))
    with patch.object(client._budget, "check"), patch("pathlib.Path.open", side_effect=PermissionError("read only")):
        result, _ = await client.complete("prompt", "system", "file1", "writer")
    assert result == "result" and client._call_provider.await_count == 1
    assert any(r["event"] == "usage_persistence_failed" for r in records(log_output))


async def test_retry_is_correlated_and_success_is_charged_once(tmp_path, log_output):
    with patch("llm_wiki.llm.client.openai.AsyncOpenAI"):
        client = LLMClient()
    client._usage_log_path = tmp_path / "ledger.jsonl"
    client._call_provider = AsyncMock(side_effect=[RuntimeError("temporary"), ("answer", 50, 10, 0)])
    with patch("llm_wiki.llm.client.asyncio.sleep", new=AsyncMock()):
        _, usage = await client.complete("prompt", "system", "file1", "writer")
    rows = records(log_output)
    failed = next(r for r in rows if r["event"] == "ai_attempt_failed")
    final = next(r for r in rows if r["event"] == "ai_call_finished")
    assert failed["call_id"] == final["call_id"] == usage.call_id
    assert failed["retrying"] and final["attempts"] == 2 and final["outcome"] == "success"
    assert len(client._usage_log_path.read_text().splitlines()) == 1


async def test_semaphore_wait_is_separate_from_provider_time(tmp_path, log_output):
    with patch("llm_wiki.llm.client.openai.AsyncOpenAI"):
        client = LLMClient()
    client._usage_log_path = tmp_path / "ledger.jsonl"
    client._call_provider = AsyncMock(return_value=("answer", 50, 10, 0))
    semaphore = asyncio.Semaphore(0)
    async def release():
        await asyncio.sleep(0.025)
        semaphore.release()
    with patch("llm_wiki.llm.client._llm_semaphore", return_value=semaphore):
        (_, usage), _ = await asyncio.gather(client.complete("p", "s", "file1", "writer"), release())
    assert usage.semaphore_wait_ms >= 15
    assert usage.provider_duration_ms < usage.semaphore_wait_ms


def test_audio_parser_records_duration_and_requests_structured_response(tmp_path, monkeypatch, log_output):
    from llm_wiki.parsers.audio import transcribe_audio
    monkeypatch.setattr(settings, "transcription_model", "whisper-1")
    monkeypatch.setattr(settings, "openai_api_key", "sk-offline-test")
    path = tmp_path / "sample.wav"
    path.write_bytes(b"mocked audio")
    with patch("openai.OpenAI") as sdk:
        sdk.return_value.audio.transcriptions.create.return_value = NS(text="words", duration=30)
        assert transcribe_audio(path, "file1") == "words"
        assert sdk.return_value.audio.transcriptions.create.call_args.kwargs["response_format"] == "verbose_json"
    [usage] = records(log_output, "llm_usage")
    assert usage["audio_seconds"] == 30 and usage["input_tokens"] is None
    assert usage["agent_type"] == "transcription" and usage["file_id"] == "file1"


def test_ocr_parser_records_each_page(log_output):
    from llm_wiki.parsers.ocr import _transcribe_images
    with patch("openai.OpenAI") as sdk:
        sdk.return_value.chat.completions.create.return_value = NS(
            choices=[NS(message=NS(content="page"))],
            usage=NS(prompt_tokens=100, completion_tokens=10))
        assert _transcribe_images([("image/png", b"mock"), ("image/png", b"mock2")], "file1") == "page\n\npage"
    rows = records(log_output, "llm_usage")
    assert [r["page"] for r in rows] == [1, 2]
    assert all(r["input_tokens"] == 100 and r["agent_type"] == "ocr" for r in rows)


async def test_image_generation_records_provider_usage(log_output):
    with patch("llm_wiki.llm.client.openai.AsyncOpenAI"):
        client = LLMClient()
    client._client.images.generate = AsyncMock(return_value=NS(
        data=[NS(b64_json="ZmFrZQ==")], usage=NS(input_tokens=100, output_tokens=200)))
    assert (await client.generate_image("private prompt")).startswith("data:image/png;base64,")
    [usage] = records(log_output, "llm_usage")
    assert usage["agent_type"] == "image" and usage["output_tokens"] == 200
    assert usage["image_count"] == 1


def test_media_budget_blocks_before_sdk_call(monkeypatch, log_output):
    from llm_wiki.quality.budget import BudgetExceeded
    monkeypatch.setattr(settings, "daily_cost_limit_usd", 0)
    reached = False
    with pytest.raises(BudgetExceeded):
        with media_call("test", "image", "file1"):
            reached = True
    assert not reached
    [event] = records(log_output, "ai_call_finished")
    assert event["error_type"] == "BudgetExceeded"
    assert event["outcome"] == "blocked"


async def test_transport_disconnect_is_not_logged_as_server_failure(log_output):
    async def stream(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"text/event-stream")]})
        await send({"type": "http.response.body", "body": b"data: {}\n\n", "more_body": True})
    calls = 0
    async def disconnected_send(message):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("connection closed")
    with pytest.raises(OSError):
        await RequestIDMiddleware(stream)({"type": "http", "method": "GET", "path": "/s", "headers": []}, AsyncMock(), disconnected_send)
    [log] = records(log_output, "request_handled")
    assert log["outcome"] == "cancelled"


def test_celery_tracer_preserves_request_and_propagates_to_child(log_output):
    from celery.app.trace import build_tracer
    celery = Celery("trace-test", broker="memory://", task_cls=ObservedTask)
    @celery.task(name="test.child")
    def child(case_id):
        return None
    @celery.task(bind=True, name="test.parent", ignore_result=True)
    def parent(self, case_id):
        assert self.request.id == "parent-id"
        child.delay(case_id)
        mark_outcome("degraded")
        return {"status": "tagged"}
    tracer = build_tracer(parent.name, parent, app=celery)
    with patch.object(Task, "apply_async", return_value=NS(id="child-id")) as publish:
        result = tracer("parent-id", ("case1",), {}, {"id": "parent-id", "retries": 0,
            "headers": {"observability": {"request_id": "r1", "run_id": "run1"}}})
    assert result.info is None
    assert publish.call_args.kwargs["headers"]["observability"]["run_id"] == "run1"
    [finished] = records(log_output, "task_execution_finished")
    assert finished["outcome"] == "degraded" and finished["task_id"] == "parent-id"
