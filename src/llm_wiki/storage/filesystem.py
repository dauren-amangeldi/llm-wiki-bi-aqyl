"""Atomic filesystem operations for /raw/ and /wiki/ directories.

All writes use the write-to-temp → os.replace() pattern so that a crash
mid-write never leaves a partially written file.
Implemented in LW-2.
"""

import os
import tempfile
from pathlib import Path


def atomic_write(path: Path, content: str, encoding: str = "utf-8") -> None:
    """Write *content* to *path* atomically.

    Writes to a sibling temp file, then renames into place. The rename is
    atomic on POSIX systems, so readers never see a partial file.

    Args:
        path: Destination file path. Parent directory must exist.
        content: Text content to write.
        encoding: File encoding (default UTF-8).
    """
    dir_ = path.parent
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding=encoding,
        dir=dir_,
        delete=False,
        suffix=".tmp",
    ) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


def ensure_dirs(*dirs: Path) -> None:
    """Create *dirs* (and parents) if they do not already exist.

    Args:
        dirs: One or more directory paths to create.
    """
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
