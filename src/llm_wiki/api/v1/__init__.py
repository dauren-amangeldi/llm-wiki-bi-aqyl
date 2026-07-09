"""API v1 — bridge sub-router for the llm-wiki-frontend."""

from fastapi import APIRouter

router = APIRouter()

# Sub-modules register their routes on `router` above.
# Imports must come AFTER `router` is defined (circular-import pattern).
from llm_wiki.api.v1 import auth, ask, cases, chat, documents, mocks, wiki  # noqa: E402, F401
