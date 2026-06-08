"""Application settings loaded from environment variables."""

from pathlib import Path
from typing import Literal

from pydantic import Field
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
    llm_provider: Literal["ollama", "openai", "anthropic"] = "ollama"
    ollama_base_url: str = "http://ollama:11434"
    ollama_model: str = "qwen2.5-coder:14b"
    openai_api_key: str = Field(default="", repr=False)
    openai_model: str = "gpt-5.4-mini"
    anthropic_api_key: str = Field(default="", repr=False)
    anthropic_model: str = "claude-3-5-sonnet-20241022"

    # --- Infrastructure ---
    redis_url: str = "redis://redis:6379/0"
    database_url: str = "sqlite+aiosqlite:///data/metadata.db"
    data_dir: Path = Path("/app/data")

    # --- API ---
    max_file_size_mb: int = 50
    allowed_extensions: frozenset[str] = frozenset({".pdf", ".md"})

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
    chunk_retrieval_top_k: int = 8     # how many chunks AnswerAgent pulls from Chroma

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
            "ollama": {"input": 0.00, "output": 0.00},
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
    def chroma_dir(self) -> Path:
        """Directory for ChromaDB persistence."""
        return self.data_dir / "chroma"

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

    @property
    def usage_log_path(self) -> Path:
        """Path to the LLM usage JSONL log."""
        return self.data_dir / "usage.log"


settings = Settings()
