"""ChromaDB client factory — embedded (local disk) or HTTP (external server).

In production the vector store runs as its own service (``chroma_backend=http``)
so the app/worker pods stay stateless. Locally it defaults to an embedded
``PersistentClient`` on disk.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def make_chroma_client(chroma_path: Path) -> Any:
    """Return a ChromaDB client based on ``settings.chroma_backend``.

    Args:
        chroma_path: On-disk persistence directory (used only for the local
            embedded backend).
    """
    import chromadb

    from llm_wiki.config import settings

    if settings.chroma_backend == "http":
        return chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)

    chroma_path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(chroma_path))
