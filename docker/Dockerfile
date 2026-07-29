FROM python:3.11-slim

# System dependencies for pdfplumber (poppler). curl is only needed to fetch uv.
# fonts-dejavu-core supplies the Unicode TTF (Cyrillic) that fpdf2 embeds when
# exporting artifacts to PDF — the fpdf2 core fonts are latin-1 only.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpoppler-cpp-dev \
    poppler-utils \
    fonts-dejavu-core \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast Python package manager) into a shared path so the non-root
# runtime user can run it too.
RUN curl -Ls https://astral.sh/uv/install.sh | sh \
    && cp /root/.local/bin/uv /root/.local/bin/uvx /usr/local/bin/
ENV PATH="/usr/local/bin:${PATH}"

WORKDIR /app

# Copy dependency manifests first for layer caching.
COPY pyproject.toml uv.lock* ./

# Production dependencies only — optional extras (dev/test) are opt-in, so a
# plain `uv sync` installs just the main deps. Tests run in CI with
# `uv sync --extra dev`; they are not shipped to prod.
RUN uv sync --frozen || uv sync

# Application source + runtime scripts (no tests in the prod image).
COPY src/ src/
COPY scripts/ scripts/

# Copy entrypoint and make it executable.
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Non-root user (production hardening). Pre-create the data dirs and hand the
# whole /app tree to appuser so a freshly-mounted empty volume inherits a
# writable owner. NOTE: in k8s, prefer external storage (Postgres/S3) over a
# pod-local volume — pods are ephemeral and restart together.
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/data/raw /app/data/wiki /app/data/chroma \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

ENTRYPOINT ["/entrypoint.sh"]
CMD ["uv", "run", "uvicorn", "llm_wiki.main:app", "--host", "0.0.0.0", "--port", "8000"]
