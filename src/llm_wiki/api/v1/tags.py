"""Phase 1 — GET /tags stub.

Returns an empty list.  Phase 6 adds the tags table and live tag management.
"""

from fastapi import APIRouter

router = APIRouter()


@router.get("/tags", tags=["tags"])
async def list_tags() -> list[dict[str, str]]:
    """Return all available tags.  Phase 1: always returns [] (stub for Phase 6)."""
    return []
