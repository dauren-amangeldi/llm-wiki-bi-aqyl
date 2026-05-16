"""Shared pytest fixtures for all tests."""

import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    """Return a temporary directory pre-populated with wiki data structure."""
    (tmp_path / "raw").mkdir()
    (tmp_path / "wiki").mkdir()
    (tmp_path / "chroma").mkdir()
    (tmp_path / "index.md").write_text("# Wiki Index\n")
    (tmp_path / "log.md").write_text("# Ingestion Log\n")
    (tmp_path / "issues.md").write_text("# Lint Agent Issues\n")
    return tmp_path


@pytest.fixture
def sample_pdf_path() -> Path:
    """Return the path to the sample PDF test fixture."""
    return Path(__file__).parent / "fixtures" / "sample.pdf"


@pytest.fixture
def sample_md_path() -> Path:
    """Return the path to the sample Markdown test fixture."""
    return Path(__file__).parent / "fixtures" / "sample.md"
