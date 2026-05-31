"""File hashing utilities for content-based deduplication."""

import hashlib
from typing import BinaryIO


def sha256_stream(stream: BinaryIO, chunk_size: int = 64 * 1024) -> str:
    """Compute the SHA-256 hex digest of *stream* without loading it all into memory.

    After reading, the stream position is reset to 0 (``stream.seek(0)``) so
    the caller can immediately write or re-read the content.

    Args:
        stream: Readable binary stream.  Must support ``read(n)`` and ``seek()``.
        chunk_size: Number of bytes read per iteration.  Default 64 KiB is a
            good balance between memory use and syscall overhead.

    Returns:
        Lowercase hex-encoded SHA-256 digest (64 characters).

    Example::

        with open("document.pdf", "rb") as f:
            digest = sha256_stream(f)
    """
    hasher = hashlib.sha256()
    while True:
        chunk = stream.read(chunk_size)
        if not chunk:
            break
        hasher.update(chunk)
    stream.seek(0)
    return hasher.hexdigest()
