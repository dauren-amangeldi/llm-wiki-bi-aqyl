"""Application settings loaded from environment variables."""

from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for llm-wiki.

    All values are read from environment variables (or .env file).
    Never hardcode secrets — use .env.* files that are gitignored.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- LLM provider ---
    llm_provider: Literal["openai", "anthropic"] = "openai"
    openai_api_key: str = Field(default="", repr=False)
    openai_model: str = "gpt-5.4-mini"
    # Image generation for the infographic artifact (OpenAI Images API). The
    # picture is decorative/thematic (text-free prompt) — the accurate data
    # lives in the cards the frontend renders under it. Falls back to a
    # self-contained SVG when generation is unavailable.
    image_model: str = "gpt-image-2"  # richer/denser than gpt-image-1 for infographics
    image_size: str = "1536x1024"  # landscape 16:9 — the art-director infographic slide
    image_quality: str = "high"  # gpt-image-1: low|medium|high|auto — the main quality lever
    # Speech-to-text for audio uploads (mp3/ogg/wav/m4a/webm). OpenAI Whisper.
    transcription_model: str = "whisper-1"
    # OCR for scanned/photo PDFs with no text layer: render each page and read
    # it with a vision model. Only runs as a fallback when pypdf+pdfplumber
    # extract < ocr_min_text_chars, so text PDFs are untouched (no extra cost).
    # ocr_model must be a VISION-capable model.
    ocr_enabled: bool = True
    ocr_model: str = "gpt-5.4-mini"
    ocr_max_pages: int = 20            # cost/latency cap per document
    ocr_min_text_chars: int = 100      # below this, treat the PDF as scanned → OCR
    ocr_render_scale: float = 2.0      # pypdfium2 render scale (~144 DPI at 2.0)
    anthropic_api_key: str = Field(default="", repr=False)
    anthropic_model: str = "claude-3-5-sonnet-20241022"
    llm_timeout_s: float = 60.0  # HTTP timeout for LLM + embedding API calls
    # Max CONCURRENT LLM calls per process/event loop (see _llm_semaphore).
    # Bounds provider load however many workers/requests fan out.
    llm_max_concurrency: int = 4
    # Token for GET /api/v1/ops/generations (Grafana JSON datasource sends it
    # as X-Ops-Token). Empty → the endpoint is open only while auth is off
    # (local/demo); with auth on it returns 403 until a token is configured.
    ops_token: str = Field(default="", repr=False)

    # --- Infrastructure ---
    redis_url: str = "redis://redis:6379/0"
    database_url: str = (
        "postgresql+psycopg://llmwiki:llmwiki@postgres:5432/llmwiki"
    )
    # Managed-DB parts (the platform injects these). When POSTGRES_HOST is set,
    # DATABASE_URL is assembled from them (see the validator below).
    postgres_host: str = ""
    postgres_port: int = 5432
    postgres_user: str = "llmwiki"
    postgres_password: str = Field(default="", repr=False)
    postgres_db: str = "llmwiki"
    data_dir: Path = Path("/app/data")

    # --- Vector store ---
    # Vectors live inside PostgreSQL via the pgvector extension (see
    # heading_embeddings / chunk_embeddings tables). No separate vector DB.

    # --- Object storage (raw uploads + wiki pages) ---
    storage_backend: Literal["local", "s3"] = "local"
    s3_endpoint: str = "minio:9000"          # host:port, no scheme
    s3_access_key: str = ""
    s3_secret_key: str = Field(default="", repr=False)
    s3_bucket: str = "llm-wiki"
    s3_secure: bool = False                   # True for https endpoints

    # --- API ---
    max_file_size_mb: int = 50
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:5173"])
    allowed_extensions: frozenset[str] = frozenset(
        {
            ".pdf",
            ".md",
            ".txt",
            ".docx",
            # audio → transcribed to text before ingestion
            ".mp3",
            ".ogg",
            ".wav",
            ".m4a",
            ".webm",
            # images (фото конспектов/заметок) → vision-OCR before ingestion
            ".jpg",
            ".jpeg",
            ".png",
            ".webp",
        }
    )

    # --- Auth (Keycloak / OIDC) ---
    # When False (default) the app trusts the X-User-Email header (dev/demo) —
    # tests and the current demo rely on this. When True, API requests must
    # carry a valid Keycloak access-token (bearer JWT, verified via JWKS).
    auth_enabled: bool = False
    # Access model once auth is on. Default OPEN (Zebo-style): ANY user who
    # passes Keycloak SSO is allowed in — the allowed_users table only grants
    # the admin role. Set AUTH_STRICT_ALLOWLIST=true to restore deny-by-default,
    # where an email must have a non-blocked allowed_users row to use the API.
    auth_strict_allowlist: bool = False
    keycloak_url: str = "https://sso.test.bi.group"
    keycloak_realm: str = "bi-group"
    # Full realm issuer (e.g. https://sso.test.bi.group/realms/bi-group). When
    # set (KEYCLOAK_ISSUER) it overrides KEYCLOAK_URL + KEYCLOAK_REALM.
    keycloak_issuer: str = ""
    keycloak_client_id: str = "ai-office-bi-aqyl"
    keycloak_client_secret: str = Field(default="", repr=False)
    # Public browser-facing origin of this app, used for the OIDC redirect_uri
    # and the post-login redirect (e.g. https://ai-office.bi.group). Empty →
    # derived from the incoming request. Also accepts FRONTEND_URL.
    public_base_url: str = Field(
        default="", validation_alias=AliasChoices("PUBLIC_BASE_URL", "FRONTEND_URL")
    )
    # How long the refresh-token cookie lives (seconds). The SPA silently swaps a
    # short-lived access token for a new one via /auth/refresh; this only bounds
    # how long that works without a fresh login. Keycloak still enforces the real
    # refresh-token / SSO-session lifetime — if it expires sooner, refresh 401s.
    keycloak_refresh_cookie_max_age_s: int = 43_200  # 12h

    @model_validator(mode="after")
    def _assemble_database_url(self) -> "Settings":
        """Build DATABASE_URL from POSTGRES_* parts when POSTGRES_HOST is given."""
        if self.postgres_host:
            self.database_url = (
                f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}"
                f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
            )
        return self

    # --- Wiki output language ---
    wiki_language: str = "ru"

    # --- Observability ---
    log_level: str = "INFO"
    service_name: str = "llm-wiki"

    # --- Embeddings (always OpenAI, regardless of chat provider) ---
    embedding_model: str = "text-embedding-3-small"
    embedding_batch_size: int = 100
    embedding_dimensions: int = 1536

    # --- Rate limiting & budget (LW-19) ---
    ingestion_enabled: bool = True
    ingestion_rate_limit_per_min: int = 10        # POST /files: per source IP
    ask_rate_limit_per_min: int = 30              # POST /ask: per source IP
    daily_cost_limit_usd: float | None = None     # None = disabled
    daily_token_limit: int | None = None          # None = disabled
    budget_warning_threshold_pct: float = 0.80    # log warning at 80% of limit

    # --- Chunk store (LW-20.1) ---
    chunk_max_chars: int = 2000        # ~500 tokens; per-chunk context window
    chunk_overlap_chars: int = 200     # overlap between consecutive chunks in a long section
    chunk_retrieval_top_k: int = 8     # how many chunks AnswerAgent pulls per query

    # --- Search tuning ---
    search_top_k: int = 20
    search_similarity_threshold: float = 0.3
    search_final_k_max: int = 10
    search_summary_max_chars: int = 8_000  # ~2 000 tokens at 4 chars/token

    # --- Cost tracking (USD per 1M tokens, May 2026) ---
    price_table: dict[str, dict[str, float]] = Field(
        default={
            "gpt-5.4-mini": {"input": 0.75, "output": 4.50},
            "gpt-5.4": {"input": 2.50, "output": 15.00},
            "text-embedding-3-small": {"input": 0.02, "output": 0.00},
        }
    )

    @property
    def raw_dir(self) -> Path:
        """Directory for uploaded source files."""
        return self.data_dir / "raw"

    @property
    def wiki_dir(self) -> Path:
        """Directory for generated wiki pages."""
        return self.data_dir / "wiki"

    @property
    def index_path(self) -> Path:
        """Path to the wiki index file."""
        return self.data_dir / "index.md"

    @property
    def log_path(self) -> Path:
        """Path to the ingestion log file."""
        return self.data_dir / "log.md"

    @property
    def issues_path(self) -> Path:
        """Path to the Lint Agent issues report."""
        return self.data_dir / "issues.md"

    # --- Keycloak / OIDC derived URLs ---
    @property
    def keycloak_realm_url(self) -> str:
        """Realm base URL / expected ``iss`` — the KEYCLOAK_ISSUER override, or
        assembled from KEYCLOAK_URL + KEYCLOAK_REALM."""
        return (
            self.keycloak_issuer.rstrip("/")
            or f"{self.keycloak_url.rstrip('/')}/realms/{self.keycloak_realm}"
        )

    @property
    def keycloak_jwks_url(self) -> str:
        """JWKS endpoint used to verify token signatures."""
        return f"{self.keycloak_realm_url}/protocol/openid-connect/certs"

    @property
    def keycloak_auth_url(self) -> str:
        """Authorization endpoint (login redirect)."""
        return f"{self.keycloak_realm_url}/protocol/openid-connect/auth"

    @property
    def keycloak_token_url(self) -> str:
        """Token endpoint (code → token exchange)."""
        return f"{self.keycloak_realm_url}/protocol/openid-connect/token"

    @property
    def keycloak_logout_url(self) -> str:
        """End-session endpoint (RP-initiated logout — kills the SSO session)."""
        return f"{self.keycloak_realm_url}/protocol/openid-connect/logout"

    @property
    def usage_log_path(self) -> Path:
        """Path to the LLM usage JSONL log."""
        return self.data_dir / "usage.log"


settings = Settings()
