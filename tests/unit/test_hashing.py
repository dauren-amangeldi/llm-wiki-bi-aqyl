"""Unit tests for SHA-256 streaming hash utility (LW-12.1)."""

import io

from llm_wiki.utils.hashing import sha256_stream


def test_sha256_stream_deterministic() -> None:
    """Same content always yields the same digest."""
    content = b"hello world"
    digests = {sha256_stream(io.BytesIO(content)) for _ in range(5)}
    assert len(digests) == 1  # all identical


def test_sha256_stream_known_value() -> None:
    """Verify against the canonical SHA-256 of 'hello world'."""
    expected = "b94d27b9934d3e08a52e52d7da7dabfac484efe04294e576fb76fa3a6cc3c8b"
    # Note: Python's hashlib gives the correct value:
    import hashlib

    expected = hashlib.sha256(b"hello world").hexdigest()
    assert sha256_stream(io.BytesIO(b"hello world")) == expected


def test_sha256_stream_seek_resets() -> None:
    """After sha256_stream(), stream.tell() == 0 so it can be re-read."""
    buf = io.BytesIO(b"content to hash")
    sha256_stream(buf)
    assert buf.tell() == 0


def test_sha256_stream_different_content_different_digest() -> None:
    """Two different byte sequences produce different digests."""
    d1 = sha256_stream(io.BytesIO(b"file one"))
    d2 = sha256_stream(io.BytesIO(b"file two"))
    assert d1 != d2


def test_sha256_stream_empty_file() -> None:
    """Empty stream returns the well-known SHA-256 of an empty string."""
    import hashlib

    expected = hashlib.sha256(b"").hexdigest()
    assert sha256_stream(io.BytesIO(b"")) == expected


def test_sha256_stream_chunked_matches_single_read() -> None:
    """chunk_size parameter does not change the digest."""
    data = b"A" * 200_000  # ~200 KB
    d_small = sha256_stream(io.BytesIO(data), chunk_size=1024)
    d_large = sha256_stream(io.BytesIO(data), chunk_size=200_000)
    assert d_small == d_large
