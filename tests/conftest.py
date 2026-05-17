"""Shared pytest fixtures for all tests."""

from pathlib import Path

import pytest


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
