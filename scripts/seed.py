"""Seed the system with demo files for local development.

Run inside the container:
    docker compose exec api uv run python scripts/seed.py
"""

import asyncio
from pathlib import Path


FIXTURES_DIR = Path(__file__).parent.parent / "tests" / "fixtures"


async def main() -> None:
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


if __name__ == "__main__":
    asyncio.run(main())
