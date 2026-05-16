# Transformer Architecture

The Transformer model was introduced in "Attention is All You Need" (Vaswani et al., 2017).

## Key Components

- **Self-Attention**: computes attention scores between all token pairs in the input sequence
- **Multi-Head Attention**: runs attention in parallel across multiple learned subspaces
- **Positional Encoding**: adds position information since Transformers have no recurrence

## Encoder–Decoder Structure

The original architecture consists of stacked encoder and decoder blocks.
Encoder-only variants (e.g. [[bert]]) are used for classification tasks.
Decoder-only variants (e.g. [[gpt]]) are used for language generation.

## References

- [[attention-mechanism]]
- [[bert]]
- [[gpt]]
