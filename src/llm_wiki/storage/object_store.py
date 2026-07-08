"""Object storage abstraction — local filesystem or S3/MinIO.

Raw uploads (``YYYY/MM/DD/{file_id}{ext}``) and generated wiki pages
(``wiki/{slug}.md``) are addressed by *key*, not by filesystem path, so the
same code works whether the bytes live on a local volume (dev) or in
S3/MinIO (production — pods are ephemeral and must not hold data).

Backend is chosen by ``settings.storage_backend`` (``local`` | ``s3``).
A module-level singleton is exposed via :func:`get_object_store`.
"""

from __future__ import annotations

import io
import os
import tempfile
from collections.abc import Iterator
from datetime import datetime
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import structlog

logger = structlog.get_logger(__name__)

# Key prefixes (forward-slash keys, S3-style — also valid local subpaths).
# RAW_PREFIX is the *legacy* raw location; new uploads are date-partitioned
# (YYYY/MM/DD/...) per the BI S3 storage standard.
RAW_PREFIX = "raw/"
WIKI_PREFIX = "wiki/"


def raw_key(file_id: str, ext: str, created_at: datetime) -> str:
    """Date-partitioned storage key for an uploaded source file.

    Layout: ``YYYY/MM/DD/<file_id><ext>`` (BI S3 standard — ГОД/Месяц/День).
    The filename is the UUID file_id + extension: only letters/digits/hyphens,
    no spaces, well under the 150-character limit.
    """
    return f"{created_at:%Y/%m/%d}/{file_id}{ext}"


def legacy_raw_key(file_id: str, ext: str) -> str:
    """Pre-date-partitioning key (``raw/<file_id><ext>``) for reading old uploads."""
    return f"{RAW_PREFIX}{file_id}{ext}"


def wiki_key(slug: str) -> str:
    """Storage key for a generated wiki page."""
    return f"{WIKI_PREFIX}{slug}.md"


def slug_from_wiki_key(key: str) -> str:
    """Inverse of :func:`wiki_key` — extract the slug from a wiki object key."""
    return key[len(WIKI_PREFIX):][:-3] if key.endswith(".md") else key[len(WIKI_PREFIX):]


@dataclass(frozen=True)
class ObjectMeta:
    """Lightweight metadata for a stored object."""

    key: str
    size: int
    last_modified: float  # unix timestamp (seconds)


class ObjectStore(Protocol):
    """Minimal blob store: put/get bytes+text, exists, delete, list."""

    def put_bytes(self, key: str, data: bytes) -> None: ...
    def put_text(self, key: str, text: str) -> None: ...
    def get_bytes(self, key: str) -> bytes: ...
    def get_text(self, key: str) -> str | None: ...
    def exists(self, key: str) -> bool: ...
    def delete(self, key: str) -> None: ...
    def list_objects(self, prefix: str) -> list[ObjectMeta]: ...

    @contextmanager
    def as_local_path(self, key: str) -> Iterator[Path]:
        """Yield a real local path for *key* (downloads S3 keys to a tempfile)."""
        ...


# ---------------------------------------------------------------------------
# Local filesystem backend
# ---------------------------------------------------------------------------


class LocalObjectStore:
    """Stores objects as files under ``base_dir/<key>``."""

    def __init__(self, base_dir: Path) -> None:
        self._base = base_dir

    def _path(self, key: str) -> Path:
        return self._base / key

    def put_bytes(self, key: str, data: bytes) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        # write-to-temp → os.replace so readers never see a partial file
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, delete=False, suffix=".tmp"
        ) as tmp:
            tmp.write(data)
            tmp_path = Path(tmp.name)
        os.replace(tmp_path, path)

    def put_text(self, key: str, text: str) -> None:
        self.put_bytes(key, text.encode("utf-8"))

    def get_bytes(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def get_text(self, key: str) -> str | None:
        path = self._path(key)
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8")

    def exists(self, key: str) -> bool:
        return self._path(key).exists()

    def delete(self, key: str) -> None:
        self._path(key).unlink(missing_ok=True)

    def list_objects(self, prefix: str) -> list[ObjectMeta]:
        root = self._path(prefix)
        base_dir = root if root.is_dir() else root.parent
        if not base_dir.exists():
            return []
        out: list[ObjectMeta] = []
        for path in base_dir.rglob("*"):
            if not path.is_file():
                continue
            key = path.relative_to(self._base).as_posix()
            if not key.startswith(prefix):
                continue
            st = path.stat()
            out.append(ObjectMeta(key=key, size=st.st_size, last_modified=st.st_mtime))
        return out

    @contextmanager
    def as_local_path(self, key: str) -> Iterator[Path]:
        yield self._path(key)


# ---------------------------------------------------------------------------
# S3 / MinIO backend
# ---------------------------------------------------------------------------


class S3ObjectStore:
    """Stores objects in an S3-compatible bucket via the MinIO client."""

    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        bucket: str,
        secure: bool,
    ) -> None:
        from minio import Minio

        self._client = Minio(
            endpoint, access_key=access_key, secret_key=secret_key, secure=secure
        )
        self._bucket = bucket
        if not self._client.bucket_exists(bucket):
            self._client.make_bucket(bucket)

    def put_bytes(self, key: str, data: bytes) -> None:
        self._client.put_object(self._bucket, key, io.BytesIO(data), length=len(data))

    def put_text(self, key: str, text: str) -> None:
        self.put_bytes(key, text.encode("utf-8"))

    def get_bytes(self, key: str) -> bytes:
        resp = self._client.get_object(self._bucket, key)
        try:
            return resp.read()
        finally:
            resp.close()
            resp.release_conn()

    def get_text(self, key: str) -> str | None:
        from minio.error import S3Error

        try:
            return self.get_bytes(key).decode("utf-8")
        except S3Error:
            return None

    def exists(self, key: str) -> bool:
        from minio.error import S3Error

        try:
            self._client.stat_object(self._bucket, key)
            return True
        except S3Error:
            return False

    def delete(self, key: str) -> None:
        self._client.remove_object(self._bucket, key)

    def list_objects(self, prefix: str) -> list[ObjectMeta]:
        out: list[ObjectMeta] = []
        for obj in self._client.list_objects(self._bucket, prefix=prefix, recursive=True):
            lm = obj.last_modified.timestamp() if obj.last_modified else 0.0
            out.append(ObjectMeta(key=obj.object_name, size=obj.size or 0, last_modified=lm))
        return out

    @contextmanager
    def as_local_path(self, key: str) -> Iterator[Path]:
        suffix = Path(key).suffix
        fd, tmp_name = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        tmp_path = Path(tmp_name)
        try:
            tmp_path.write_bytes(self.get_bytes(key))
            yield tmp_path
        finally:
            tmp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Singleton factory
# ---------------------------------------------------------------------------

_store: ObjectStore | None = None


def get_object_store() -> ObjectStore:
    """Return the configured object store (constructed once)."""
    global _store
    if _store is None:
        from llm_wiki.config import settings

        if settings.storage_backend == "s3":
            _store = S3ObjectStore(
                endpoint=settings.s3_endpoint,
                access_key=settings.s3_access_key,
                secret_key=settings.s3_secret_key,
                bucket=settings.s3_bucket,
                secure=settings.s3_secure,
            )
            logger.info("object_store_init", backend="s3", bucket=settings.s3_bucket)
        else:
            _store = LocalObjectStore(settings.data_dir)
            logger.info("object_store_init", backend="local", base=str(settings.data_dir))
    return _store


def reset_object_store() -> None:
    """Drop the cached singleton (tests / settings changes)."""
    global _store
    _store = None
