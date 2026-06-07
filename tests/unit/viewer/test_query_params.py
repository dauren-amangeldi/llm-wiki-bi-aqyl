"""Smoke tests for the Streamlit viewer (LW-16 deep-linking).

These tests verify the module-level logic and helper functions that are
*independent* of Streamlit's runtime.  Full Streamlit rendering requires a
running server, which is outside the scope of unit tests.

Streamlit is only present in the ``viewer`` optional-dependency group, so
all tests in this module are skipped when the package is not installed.
"""

from __future__ import annotations

import importlib
import sys
import types
from unittest.mock import MagicMock

import pytest

# Skip the entire module if streamlit is not installed
streamlit = pytest.importorskip("streamlit", reason="streamlit not installed (viewer extra)")


def _make_streamlit_stub() -> types.ModuleType:
    """Return a minimal Streamlit stub so app.py can be imported without a
    running Streamlit server.  Only the symbols used at module level need
    to be stubbed.
    """
    stub = types.ModuleType("streamlit")

    # query_params: dict-like, writable
    stub.query_params = {}  # type: ignore[attr-defined]

    # session_state: dict-like
    stub.session_state = {}  # type: ignore[attr-defined]

    # Callables that do nothing at import time
    for name in (
        "set_page_config",
        "sidebar",
        "title",
        "subheader",
        "info",
        "warning",
        "error",
        "divider",
        "metric",
        "columns",
        "button",
        "radio",
        "caption",
        "markdown",
        "expander",
        "rerun",
        "write",
        "stop",
    ):
        setattr(stub, name, MagicMock(return_value=MagicMock()))

    # sidebar returns an object with the same attributes
    stub.sidebar = stub  # type: ignore[assignment]
    return stub


class TestViewerImport:
    """App.py must be importable (no runtime errors at module level)."""

    def test_app_imports_without_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Import app.py with a stubbed Streamlit to verify there are no
        import-time errors (missing modules, syntax errors, bad attribute
        access on stub objects, etc.).
        """
        # Remove cached module if already loaded
        for key in list(sys.modules):
            if "llm_wiki.viewer.app" in key or "viewer.app" in key:
                del sys.modules[key]

        # Inject the stub so ``import streamlit as st`` in app.py gets our stub
        monkeypatch.setitem(sys.modules, "streamlit", _make_streamlit_stub())

        # This should not raise
        import llm_wiki.viewer.app  # noqa: F401


class TestNavHelper:
    """_nav() must write both session_state and query_params."""

    def test_nav_sets_session_state_and_query_params(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """After _nav('page', 'transformers'), both session_state and
        query_params must be updated.
        """
        # Build a minimal stub with mutable state
        stub = _make_streamlit_stub()
        session: dict[str, str] = {}
        stub.session_state = session  # type: ignore[assignment]
        qp: dict[str, str] = {}
        stub.query_params = qp  # type: ignore[assignment]

        for key in list(sys.modules):
            if "llm_wiki.viewer.app" in key or "viewer.app" in key:
                del sys.modules[key]

        monkeypatch.setitem(sys.modules, "streamlit", stub)
        import llm_wiki.viewer.app as viewer_app

        viewer_app._nav("page", "transformers")

        assert session.get("nav") == "page"
        assert session.get("slug") == "transformers"
        assert qp.get("nav") == "page"
        assert qp.get("slug") == "transformers"

    def test_nav_removes_slug_from_query_params_when_empty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stub = _make_streamlit_stub()
        stub.session_state = {}  # type: ignore[assignment]
        stub.query_params = {"slug": "old-slug"}  # type: ignore[assignment]

        for key in list(sys.modules):
            if "llm_wiki.viewer.app" in key or "viewer.app" in key:
                del sys.modules[key]

        monkeypatch.setitem(sys.modules, "streamlit", stub)
        import llm_wiki.viewer.app as viewer_app

        viewer_app._nav("index")  # no slug arg

        assert "slug" not in stub.query_params
        assert stub.query_params.get("nav") == "index"
