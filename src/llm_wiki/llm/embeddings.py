"""Embedding generation and ChromaDB vector store wrapper.

Implemented in LW-11 (indexing) and LW-12 (pre-filter queries).
"""

from pathlib import Path


class EmbeddingStore:
    """Wraps OpenAI text-embedding-3-small + ChromaDB for vector search.

    Used by Search Agent v2 (LW-12) to pre-filter index.md headings before
    LLM re-ranking, keeping cost linear rather than quadratic in wiki size.
    """

    COLLECTION_NAME = "wiki_headings"

    def __init__(self, chroma_dir: Path) -> None:
        """Initialise the store with the ChromaDB persistence directory.

        Args:
            chroma_dir: Path to the ChromaDB data directory.
        """
        raise NotImplementedError("Implemented in LW-11")

    async def embed_and_store(self, slug: str, title: str, content: str) -> None:
        """Generate an embedding for a wiki page heading and persist it.

        Args:
            slug: Page slug (used as the ChromaDB document ID).
            title: Page title to embed.
            content: Optional extended content for richer embeddings.
        """
        raise NotImplementedError("Implemented in LW-11")

    async def query(self, text: str, top_k: int = 20) -> list[tuple[str, float]]:
        """Return the top-k closest headings by cosine similarity.

        Args:
            text: Query text (typically the first 2 000 tokens of a document).
            top_k: Number of candidates to return before LLM re-ranking.

        Returns:
            List of (slug, similarity_score) tuples, descending by score.
        """
        raise NotImplementedError("Implemented in LW-11")
