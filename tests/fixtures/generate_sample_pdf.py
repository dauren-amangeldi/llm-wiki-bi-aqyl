"""Script to generate tests/fixtures/sample.pdf for parser tests.

Run manually (inside the container) after adding fpdf2 to dev dependencies:
    docker compose exec api uv run python tests/fixtures/generate_sample_pdf.py

The PDF contains 3 pages of technical text about Transformer architecture,
totalling >200 words so that parser tests are representative.
"""

from pathlib import Path


PAGES = [
    (
        "Transformer Architecture",
        [
            "The Transformer model was introduced in 'Attention is All You Need'",
            "(Vaswani et al., 2017). It replaced recurrent neural networks",
            "with a mechanism based entirely on attention, enabling parallelism",
            "during training and dramatically improved performance on NLP tasks.",
            "",
            "The core insight is that global dependencies between input and output",
            "can be captured without processing tokens sequentially.",
        ],
    ),
    (
        "Self-Attention Mechanism",
        [
            "Self-attention computes a weighted sum of all positions in the input",
            "sequence. Given queries Q, keys K and values V, the output is:",
            "    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V",
            "",
            "Multi-head attention runs this operation h times in parallel across",
            "different learned subspaces, then concatenates the results.",
            "This allows the model to attend to different representation subspaces",
            "simultaneously, improving expressiveness over single-head attention.",
        ],
    ),
    (
        "Training and Applications",
        [
            "Transformers are trained on large corpora using masked language",
            "modelling (BERT) or next-token prediction (GPT). Pre-trained models",
            "are fine-tuned on downstream tasks such as question answering,",
            "summarisation, translation, and code generation.",
            "",
            "Encoder-only models like BERT are used for classification tasks.",
            "Decoder-only models like GPT are used for text generation.",
            "Encoder-decoder models (the original Transformer) are used for",
            "sequence-to-sequence tasks such as machine translation.",
        ],
    ),
]


def generate(output_path: Path) -> None:
    """Generate a multi-page PDF with technical content and write to *output_path*.

    Args:
        output_path: Destination path for the generated PDF.
    """
    from fpdf import FPDF  # type: ignore[import-untyped]

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    for title, lines in PAGES:
        pdf.add_page()
        pdf.set_font("Helvetica", style="B", size=16)
        pdf.cell(0, 10, title, ln=True)
        pdf.set_font("Helvetica", size=11)
        pdf.ln(4)
        for line in lines:
            pdf.cell(0, 8, line, ln=True)

    pdf.output(str(output_path))
    print(f"Generated {output_path} ({output_path.stat().st_size} bytes)")


if __name__ == "__main__":
    dest = Path(__file__).parent / "sample.pdf"
    generate(dest)
