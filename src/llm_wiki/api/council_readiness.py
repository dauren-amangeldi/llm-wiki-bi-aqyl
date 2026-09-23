"""One readiness rule for council pickers, chat and summaries; no model calls."""

from collections.abc import Sequence

from fastapi import HTTPException
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from llm_wiki.storage.metadata import CaseRecord, FileRecord

PROCESSING_STATUSES = frozenset({"RECEIVED", "STORED", "SEARCHED", "WRITTEN", "LINTED", "LOGGED"})


async def council_readiness(
    db: AsyncSession,
    cases: Sequence[CaseRecord],
    caller: str,
) -> dict[str, dict[str, list[str]]]:
    ids = {fid for case in cases for fid in case.doc_ids or []}
    rows = (
        (
            await db.execute(
                select(FileRecord.file_id, FileRecord.status).where(
                    FileRecord.file_id.in_(ids),
                    or_(FileRecord.sensitive.is_(False), FileRecord.owner == caller),
                )
            )
        ).all()
        if ids
        else []
    )
    statuses = dict(rows)
    return {
        case.id: {
            "ready_doc_ids": [
                fid for fid in dict.fromkeys(case.doc_ids or []) if statuses.get(fid) == "DONE"
            ],
            "processing_doc_ids": [
                fid
                for fid in dict.fromkeys(case.doc_ids or [])
                if statuses.get(fid) in PROCESSING_STATUSES
            ],
        }
        for case in cases
    }


async def require_ready_case(
    db: AsyncSession,
    case_id: str,
    caller: str,
) -> tuple[CaseRecord, list[FileRecord]]:
    # Refresh ORM objects too: materials may be removed during an SSE round.
    case = await db.get(CaseRecord, case_id, populate_existing=True)
    if case is None or (case.sensitive and case.owner != caller):
        raise HTTPException(404, "case_not_available")
    documents = list(
        (
            await db.scalars(
                select(FileRecord)
                .where(
                    FileRecord.file_id.in_(case.doc_ids or []),
                    FileRecord.status == "DONE",
                    or_(FileRecord.sensitive.is_(False), FileRecord.owner == caller),
                )
                .execution_options(populate_existing=True)
            )
        ).all()
    )
    if not documents:
        raise HTTPException(422, "council_no_ready_materials")
    return case, documents
