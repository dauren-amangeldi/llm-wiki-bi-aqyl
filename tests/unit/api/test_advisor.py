"""Smoke tests for POST /api/v1/advisor SSE (LW-N8)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from llm_wiki.agents.advisor import AdvisorPoint, AdvisorResponse
from llm_wiki.main import app


@pytest.fixture
def mock_advisor_response() -> AdvisorResponse:
    return AdvisorResponse(
        title="Insights",
        summary="Two cases analysed.",
        points=[
            AdvisorPoint(
                heading="Lean",
                body="Applied lean methods.",
                metric="12%",
                tag="Timeline",
                case_id="case-001",
            )
        ],
        source="Cases: lean (1)",
        caseCount=1,
        cost_usd=0.002,
    )


async def test_advisor_sse_stream_returns_done_event(mock_advisor_response: AdvisorResponse) -> None:
    """POST /advisor?stream=true yields SSE with final done payload."""
    mock_agent = MagicMock()
    mock_agent.advise = AsyncMock(return_value=mock_advisor_response)

    with patch("llm_wiki.agents.advisor.AdvisorAgent", return_value=mock_agent), patch(
        "llm_wiki.llm.chunk_store.ChunkStore"
    ), patch("llm_wiki.llm.client.LLMClient") as mock_llm_cls:
        mock_llm = MagicMock()
        mock_llm.aclose = AsyncMock()
        mock_llm_cls.return_value = mock_llm

        async with AsyncClient(
            transport=ASGITransport(app=app),  # type: ignore[arg-type]
            base_url="http://test",
        ) as client:
            resp = await client.post(
                "/api/v1/advisor",
                params={"stream": "true"},
                json={"query": "How to reduce timelines?", "role": "pm", "language": "en"},
            )

    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers.get("content-type", "")

    body = resp.text
    assert '"status": "searching"' in body
    assert '"done": true' in body
    assert '"title": "Insights"' in body
    assert '"case_id": "case-001"' in body
    mock_agent.advise.assert_awaited_once()


async def test_advisor_sse_refusal_event() -> None:
    """Off-topic query streams a refusal done event."""
    refusal = AdvisorResponse(refusal=True, refusal_message="No materials found.")

    mock_agent = MagicMock()
    mock_agent.advise = AsyncMock(return_value=refusal)

    with patch("llm_wiki.agents.advisor.AdvisorAgent", return_value=mock_agent), patch(
        "llm_wiki.llm.chunk_store.ChunkStore"
    ), patch("llm_wiki.llm.client.LLMClient") as mock_llm_cls:
        mock_llm = MagicMock()
        mock_llm.aclose = AsyncMock()
        mock_llm_cls.return_value = mock_llm

        async with AsyncClient(
            transport=ASGITransport(app=app),  # type: ignore[arg-type]
            base_url="http://test",
        ) as client:
            resp = await client.post(
                "/api/v1/advisor",
                params={"stream": "true"},
                json={"query": "unrelated topic xyz", "language": "en"},
            )

    assert resp.status_code == 200
    assert '"refusal": true' in resp.text
    assert "No materials found" in resp.text
