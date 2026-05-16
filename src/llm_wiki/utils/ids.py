"""UUID generation for file IDs.

Uses UUID7 (time-ordered) so that file_ids sort chronologically,
which simplifies log parsing and pagination.
"""

import uuid


def new_file_id() -> str:
    """Generate a new time-ordered UUID7 string for a file record.

    Returns:
        Hyphenated UUID string, e.g. '01HXYZ...'.
    """
    # uuid6 library provides uuid7(); fall back to uuid4 if not installed.
    try:
        import uuid6  # type: ignore[import-untyped]

        return str(uuid6.uuid7())
    except ImportError:
        return str(uuid.uuid4())
