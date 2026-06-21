"""Seed the system with demo files and default skills for local development.

Run inside the container:
    docker compose exec api uv run python scripts/seed.py
    docker compose exec api uv run python scripts/seed.py --skills-only
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent.parent / "tests" / "fixtures"


async def seed_skills_db() -> None:
    """Insert default role prompts into the skills table."""
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from llm_wiki.config import settings
    from llm_wiki.storage.metadata import Base, seed_skills, skills_count

    engine = create_async_engine(settings.database_url, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(bind=engine, expire_on_commit=False, autoflush=False)
    async with factory() as session:
        before = await skills_count(session)
        inserted = await seed_skills(session)
        after = await skills_count(session)
        print(f"Skills: {before} → {after} rows ({inserted} inserted)")

    await engine.dispose()


async def seed_files() -> None:
    """Upload all fixture files through the ingestion pipeline."""
    import httpx

    files = list(FIXTURES_DIR.glob("*.pdf")) + list(FIXTURES_DIR.glob("*.md"))
    if not files:
        print("No fixture files found in tests/fixtures/")
        return

    async with httpx.AsyncClient(base_url="http://localhost:8000") as client:
        for f in files:
            with f.open("rb") as fh:
                resp = await client.post(
                    "/api/v1/files",
                    files={"file": (f.name, fh, "application/octet-stream")},
                )
            print(f"{f.name} → {resp.status_code} {resp.json()}")


async def main(skills_only: bool = False, files_only: bool = False) -> None:
    """Run selected seed steps."""
    if not files_only:
        await seed_skills_db()
    if not skills_only:
        await seed_files()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed demo data")
    parser.add_argument("--skills-only", action="store_true", help="Seed skills table only")
    parser.add_argument("--files-only", action="store_true", help="Upload fixture files only")
    args = parser.parse_args()
    asyncio.run(main(skills_only=args.skills_only, files_only=args.files_only))
