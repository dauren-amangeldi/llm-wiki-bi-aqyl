"""Audio transcription (mp3/ogg/wav/m4a/webm) via the OpenAI transcription API.

Turns a voice recording into plain text so it flows through the same wiki
ingestion pipeline as documents. Uses a synchronous OpenAI client (like the
embedding path) so it is callable from the synchronous ``_load_raw_text`` step.
"""

from __future__ import annotations

from pathlib import Path

import structlog

from llm_wiki.config import settings
from llm_wiki.llm.telemetry import media_call
from llm_wiki.observability import error_fields

logger = structlog.get_logger(__name__)

# OpenAI's transcription endpoint rejects files larger than 25 MB.
_MAX_AUDIO_BYTES = 25 * 1024 * 1024


class TranscriptionError(Exception):
    """Raised when audio cannot be transcribed."""


def transcribe_audio(path: Path, file_id: str = "ask") -> str:
    """Transcribe an audio file to plain text.

    Args:
        path: Absolute path to the audio file.
        file_id: Correlation ID for structured logging.

    Returns:
        The transcribed text.

    Raises:
        TranscriptionError: If the API key is missing, the file exceeds the
            25 MB limit, the API call fails, or no text was produced.
    """
    if not settings.openai_api_key:
        raise TranscriptionError("OPENAI_API_KEY is required to transcribe audio.")

    size = path.stat().st_size
    if size > _MAX_AUDIO_BYTES:
        raise TranscriptionError(
            f"Audio file is {size // (1024 * 1024)} MB; the transcription limit "
            "is 25 MB. Please upload a shorter or compressed recording."
        )

    import openai

    client = openai.OpenAI(
        api_key=settings.openai_api_key, timeout=settings.llm_timeout_s
    )
    logger.info(
        "transcription_start",
        file_id=file_id,
        model=settings.transcription_model,
        size_bytes=size,
    )
    try:
        with media_call(settings.transcription_model, "transcription", file_id, size_bytes=size) as telemetry:
            with path.open("rb") as handle:
                result = client.audio.transcriptions.create(
                    model=settings.transcription_model, file=handle,
                    response_format="verbose_json" if settings.transcription_model == "whisper-1" else "json",
                )
                telemetry["response"] = result
    except Exception as exc:  # noqa: BLE001 - surface any OpenAI/transport error
        logger.error("transcription_failed", file_id=file_id, **error_fields(exc))
        raise TranscriptionError("Transcription failed") from exc

    # Structured responses preserve usage/duration; tolerate legacy string responses.
    text = (result if isinstance(result, str) else getattr(result, "text", "")).strip()
    if not text:
        raise TranscriptionError(
            "Transcription produced no text (silent or unsupported audio)."
        )
    logger.info("transcription_done", file_id=file_id, chars=len(text))
    return text
