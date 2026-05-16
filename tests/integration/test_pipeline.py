"""Integration test: end-to-end file upload → wiki page.

Exercises the full pipeline: upload → Celery task → agents → wiki write.
Implemented in LW-9.
"""

import pytest


@pytest.mark.xfail(reason="Implemented in LW-9")
async def test_upload_pdf_creates_wiki_page() -> None:
    """POST /files with a PDF should eventually produce a wiki page in data/wiki/."""
    ...


@pytest.mark.xfail(reason="Implemented in LW-9")
async def test_pipeline_idempotent_on_rerun() -> None:
    """Re-processing the same file_id must not create duplicate wiki pages."""
    ...


@pytest.mark.xfail(reason="Implemented in LW-9")
async def test_pipeline_failed_state_on_llm_error() -> None:
    """When the LLM is unreachable, the file should end in FAILED state after 3 retries."""
    ...
