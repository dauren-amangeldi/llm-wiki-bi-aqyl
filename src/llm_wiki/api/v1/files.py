"""Phase 1 — GET /files/{file_id}/raw.

Serves the original uploaded file directly from data/raw/.
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from llm_wiki.config import settings

router = APIRouter()

_EXT_MEDIA: dict[str, str] = {
    ".pdf": "application/pdf",
    ".md": "text/markdown; charset=utf-8",
}


@router.get("/files/{file_id}/raw", tags=["files"])
async def download_raw(file_id: str) -> FileResponse:
    """Stream the original uploaded file.

    Looks up the raw file by globbing ``{file_id}.*`` inside ``data/raw/``.
    Sets the correct ``Content-Type`` based on the file extension.

    Raises:
        HTTPException 400: ``file_id`` contains path-traversal characters.
        HTTPException 404: No matching raw file found on disk.
    """
    if "/" in file_id or ".." in file_id or "\\" in file_id:
        raise HTTPException(status_code=400, detail="Invalid file_id.")

    raw_dir = settings.raw_dir
    matches = list(raw_dir.glob(f"{file_id}.*")) if raw_dir.exists() else []
    if not matches:
        raise HTTPException(
            status_code=404,
            detail=f"Raw file for {file_id!r} not found.",
        )

    file_path = matches[0]
    media_type = _EXT_MEDIA.get(file_path.suffix.lower(), "application/octet-stream")
    return FileResponse(
        path=str(file_path),
        media_type=media_type,
        filename=file_path.name,
    )
