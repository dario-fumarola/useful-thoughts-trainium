"""Small, local artifact helpers. No network calls or cloud mutations."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import tempfile
from datetime import datetime, timezone
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def atomic_bytes(path: str | Path, data: bytes, *, overwrite: bool = False) -> None:
    """Publish a complete file on the same filesystem; refuse accidental overwrite."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    tmp_path = Path(tmp)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(tmp_path, path)
        else:
            # Atomic no-clobber publication on POSIX; link fails if dest exists.
            os.link(tmp_path, path)
            tmp_path.unlink()
    finally:
        tmp_path.unlink(missing_ok=True)


def atomic_json(path: str | Path, value: Any, *, overwrite: bool = False) -> None:
    atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True,
                                  allow_nan=False) + "\n").encode(), overwrite=overwrite)


def source_fingerprint(root: str | Path) -> dict:
    """Hash active code only. Historical material and generated outputs are excluded."""
    root = Path(root).resolve()
    files = {}
    for directory in ("weekend_thinking", "scripts", "train"):
        location = root / directory
        if not location.exists():
            continue
        for path in sorted(location.rglob("*")):
            if path.suffix not in {".py", ".sh"} or not path.is_file():
                continue
            if path.is_symlink():
                raise ValueError(f"Symlink in active source: {path}")
            files[str(path.relative_to(root))] = sha256_file(path)
    if not files:
        raise ValueError("No active source files found.")
    return {"sha256": canonical_hash(files), "files": files}


def require_utc_deadline(value: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Resolve an explicit timezone-aware deadline before freezing.")
    date = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if date.tzinfo is None:
        raise ValueError("Deadline requires an explicit UTC offset or Z.")
    return date.astimezone(timezone.utc)
