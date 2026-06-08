"""API v1 adapter — endpoints for the llm-wiki-frontend (BI AQYL UI).

Phase 0: CORS + routing scaffold.
Phase 1: Materials list, document detail, sources, dossier, raw file download.
Phase 2: File upload (DropZone), status polling, source deletion.
Phase 3: Document chat with SSE streaming (POST /documents/{id}/ask).
See README.md "Роадмап интеграции с фронтендом" for the full plan.
"""

from fastapi import APIRouter

from llm_wiki.api.v1 import chat, files, materials, tags, uploads

router = APIRouter(tags=["v1"])

router.include_router(materials.router)
router.include_router(tags.router)
router.include_router(files.router)
router.include_router(uploads.router)
router.include_router(chat.router)
