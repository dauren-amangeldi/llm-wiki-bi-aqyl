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

# Non-root user for production safety
RUN groupadd -r appuser && useradd -r -g appuser appuser \
    && chown -R appuser:appuser /app \
    && mkdir -p /home/appuser/.cache/uv \
    && chown -R appuser:appuser /home/appuser
USER appuser

ENV UV_CACHE_DIR=/home/appuser/.cache/uv

# Data directory is mounted as a volume at runtime
VOLUME ["/app/data"]

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "llm_wiki.main:app", "--host", "0.0.0.0", "--port", "8000"]
