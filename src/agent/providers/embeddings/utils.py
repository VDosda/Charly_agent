from __future__ import annotations

import hashlib
import struct
from typing import Iterable, List


def ensure_dims(vec: List[float], expected_dims: int) -> None:
    if len(vec) != expected_dims:
        raise ValueError(f"Embedding dims mismatch: got {len(vec)} expected {expected_dims}")


def pack_f32(vec: List[float]) -> bytes:
    """
    Pack Python floats into little-endian float32 bytes for SQLite BLOB storage.
    """
    return struct.pack("<" + "f" * len(vec), *vec)


def unpack_f32(blob: bytes, dims: int) -> List[float]:
    """
    Unpack float32 bytes into Python floats.
    """
    floats = struct.unpack("<" + "f" * dims, blob)
    return list(floats)


def fingerprint_text(text: str) -> str:
    """
    Stable fingerprint for deduplication.
    """
    normalized = " ".join(text.strip().split()).lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()