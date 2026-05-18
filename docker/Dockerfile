FROM python:3.11-slim

# Install system dependencies for pdfplumber (poppler) and file locking
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpoppler-cpp-dev \
    poppler-utils \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast Python package manager)
RUN curl -Ls https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app

# Copy dependency manifests first for layer caching
COPY pyproject.toml uv.lock* ./

# Install all dependencies (including dev extras for the test/worker images)
RUN uv sync --all-extras --frozen || uv sync --all-extras

# Copy application source and tests
COPY src/ src/
COPY scripts/ scripts/
COPY tests/ tests/

# Copy entrypoint and make it executable
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Pre-create data directories so they exist even when no volume is mounted.
# Running as root here (Variant A — MVP/dev mode).
# TODO: switch to non-root user (appuser) for production hardening once the
#       volume ownership strategy is decided (init-container or bind mount).
RUN mkdir -p /app/data/raw /app/data/wiki /app/data/chroma

EXPOSE 8000

ENTRYPOINT ["/entrypoint.sh"]
CMD ["uv", "run", "uvicorn", "llm_wiki.main:app", "--host", "0.0.0.0", "--port", "8000"]
