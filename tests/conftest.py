"""Shared pytest fixtures for all tests."""

import os
from collections.abc import AsyncGenerator
from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

# Tests require a PostgreSQL instance (no SQLite). Defaults to the storage
# sandbox; override with TEST_DATABASE_URL. The test DB is wiped per test.
TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql+psycopg://llmwiki:devpassword@postgres:5432/llmwiki",
)


@pytest_asyncio.fixture
async def db_engine() -> AsyncGenerator[AsyncEngine, None]:
    """A Postgres engine with a freshly-created schema (clean slate per test)."""
    from llm_wiki.storage.metadata import Base
    from llm_wiki.storage.wiki_fts import ensure_wiki_fts_table

    engine = create_async_engine(TEST_DATABASE_URL)
    async with engine.begin() as conn:
        # pgvector must exist before create_all builds the vector() columns.
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        # pg_trgm powers fuzzy (word_similarity) search in cases/documents/wiki.
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))
        await conn.execute(text("DROP TABLE IF EXISTS wiki_fts"))
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
        await ensure_wiki_fts_table(conn)
    yield engine
    await engine.dispose()


@pytest.fixture
def vector_engine(db_engine: AsyncEngine):  # type: ignore[no-untyped-def]
    """A synchronous engine to the test DB for the (sync) vector stores.

    Depends on ``db_engine`` so the pgvector extension + tables already exist.
    """
    from sqlalchemy import create_engine

    engine = create_engine(TEST_DATABASE_URL)
    yield engine
    engine.dispose()


@pytest_asyncio.fixture
async def db_session(db_engine: AsyncEngine) -> AsyncGenerator[AsyncSession, None]:
    """An ``AsyncSession`` bound to the per-test Postgres engine."""
    factory: async_sessionmaker[AsyncSession] = async_sessionmaker(
        bind=db_engine, expire_on_commit=False, autoflush=False
    )
    async with factory() as session:
        yield session


@pytest.fixture(autouse=True)
def _isolate_global_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Per-test isolation: object store at a temp dir + fresh rate limiters."""
    from llm_wiki.config import settings
    from llm_wiki.storage import object_store
    import llm_wiki.api.deps as deps

    data = tmp_path / "_objstore"
    (data / "raw").mkdir(parents=True, exist_ok=True)
    (data / "wiki").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(settings, "data_dir", data, raising=False)
    monkeypatch.setattr(settings, "storage_backend", "local", raising=False)
    object_store.reset_object_store()
    # Rate limiters are module-level singletons that accumulate per-IP counts
    # across tests — reset so upload/ask tests don't trip 429 on each other.
    deps._files_rate_limiter = None
    deps._ask_rate_limiter = None
    yield
    object_store.reset_object_store()


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    """Return a temporary directory pre-populated with wiki data structure."""
    (tmp_path / "raw").mkdir()
    (tmp_path / "wiki").mkdir()
    (tmp_path / "chroma").mkdir()
    (tmp_path / "index.md").write_text("# Wiki Index\n")
    (tmp_path / "log.md").write_text("# Ingestion Log\n")
    (tmp_path / "issues.md").write_text("# Lint Agent Issues\n")
    return tmp_path


@pytest.fixture
def sample_pdf_path() -> Path:
    """Return the path to the sample PDF test fixture, generating it if absent.

    Uses fpdf2 (dev dependency) to generate a 3-page technical PDF on first run.
    The result is cached in tests/fixtures/sample.pdf so subsequent runs are instant.
    If fpdf2 is not installed the test is skipped rather than failing with an ImportError.
    """
    fixtures_dir = Path(__file__).parent / "fixtures"
    pdf_path = fixtures_dir / "sample.pdf"

    if not pdf_path.exists():
        try:
            from fpdf import FPDF  # type: ignore[import-untyped]
        except ImportError:
            pytest.skip("fpdf2 is not installed — run `uv sync --all-extras` inside the container")

        pages = [
            (
                "Transformer Architecture",
                [
                    "The Transformer was introduced in 'Attention is All You Need' (2017).",
                    "It replaced recurrent networks with a pure attention mechanism,",
                    "enabling parallelism during training and yielding state-of-the-art",
                    "results on NLP benchmarks. Self-attention allows the model to relate",
                    "tokens to each other regardless of their distance in the sequence.",
                ],
            ),
            (
                "Self-Attention and Multi-Head Attention",
                [
                    "Self-attention computes Attention(Q,K,V) = softmax(QK^T/sqrt(d_k))V.",
                    "Multi-head attention runs h attention functions in parallel across",
                    "different learned subspaces, concatenates the results, and projects.",
                    "This allows the model to attend to information from different",
                    "representation subspaces simultaneously.",
                ],
            ),
            (
                "Applications and Variants",
                [
                    "Encoder-only models (BERT) are used for classification tasks.",
                    "Decoder-only models (GPT) are used for language generation.",
                    "Encoder-decoder models handle sequence-to-sequence tasks.",
                    "Transformers power modern LLMs, vision models, and code assistants.",
                    "Fine-tuning on downstream tasks achieves strong performance.",
                ],
            ),
        ]

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        for title, lines in pages:
            pdf.add_page()
            pdf.set_font("Helvetica", style="B", size=16)
            pdf.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", size=11)
            pdf.ln(4)
            for line in lines:
                pdf.cell(0, 8, line, new_x="LMARGIN", new_y="NEXT")
        pdf.output(str(pdf_path))

    return pdf_path


@pytest.fixture
def sample_md_path() -> Path:
    """Return the path to the sample Markdown test fixture."""
    return Path(__file__).parent / "fixtures" / "sample.md"
