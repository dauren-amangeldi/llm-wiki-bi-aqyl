"""Unit tests for the storage layer.

Tests cover: atomic writes, file locking, and idempotency.
Full implementation added in LW-2.
"""

from pathlib import Path

import pytest

from llm_wiki.storage.filesystem import atomic_write, ensure_dirs


def test_atomic_write_creates_file(tmp_path: Path) -> None:
    """atomic_write should create a file with the given content."""
    target = tmp_path / "output.md"
    atomic_write(target, "# Hello\n")
    assert target.exists()
    assert target.read_text() == "# Hello\n"


def test_atomic_write_overwrites_existing(tmp_path: Path) -> None:
    """atomic_write should replace an existing file atomically."""
    target = tmp_path / "page.md"
    target.write_text("old content")
    atomic_write(target, "new content")
    assert target.read_text() == "new content"


def test_atomic_write_no_temp_file_left(tmp_path: Path) -> None:
    """No .tmp files should remain after a successful atomic_write."""
    target = tmp_path / "page.md"
    atomic_write(target, "content")
    leftover = list(tmp_path.glob("*.tmp"))
    assert leftover == [], f"Unexpected temp files: {leftover}"


def test_ensure_dirs_creates_nested(tmp_path: Path) -> None:
    """ensure_dirs should create nested directories if they do not exist."""
    deep = tmp_path / "a" / "b" / "c"
    ensure_dirs(deep)
    assert deep.is_dir()


def test_ensure_dirs_idempotent(tmp_path: Path) -> None:
    """ensure_dirs should not raise when directories already exist."""
    d = tmp_path / "existing"
    d.mkdir()
    ensure_dirs(d)  # second call must not raise
