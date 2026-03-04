from __future__ import annotations

import sqlite3
from typing import List, Optional, Sequence, Tuple

from agent.providers.embeddings.base import EmbeddingProvider


def upsert_episode_vec(db: sqlite3.Connection, episode_id: int, vector: List[float]) -> None:
    # sqlite-vec: vec tables accept vectors via JSON array string or as a blob depending on wrapper.
    # The simplest portable input is a JSON array string.
    import json
    with db:
        db.execute(
            "INSERT OR REPLACE INTO episodes_vec(episode_id, embedding) VALUES (?, ?)",
            (int(episode_id), json.dumps(vector)),
        )


def upsert_item_vec(db: sqlite3.Connection, item_id: int, vector: List[float]) -> None:
    import json
    with db:
        db.execute(
            "INSERT OR REPLACE INTO memory_items_vec(item_id, embedding) VALUES (?, ?)",
            (int(item_id), json.dumps(vector)),
        )


def query_topk_episode_ids(
    db: sqlite3.Connection,
    query_vec: List[float],
    k: int = 5,
    allowed_ids: Optional[Sequence[int]] = None,
) -> List[Tuple[int, float]]:
    """
    Returns list of (episode_id, distance) sorted by nearest.
    For sqlite-vec, lower distance means closer.
    """
    return _query_topk_ids(
        db,
        table="episodes_vec",
        id_col="episode_id",
        query_vec=query_vec,
        k=k,
        allowed_ids=allowed_ids,
    )


def query_topk_item_ids(
    db: sqlite3.Connection,
    query_vec: List[float],
    k: int = 10,
    allowed_ids: Optional[Sequence[int]] = None,
) -> List[Tuple[int, float]]:
    return _query_topk_ids(
        db,
        table="memory_items_vec",
        id_col="item_id",
        query_vec=query_vec,
        k=k,
        allowed_ids=allowed_ids,
    )


def embed_query(embeddings: EmbeddingProvider, text: str) -> List[float]:
    emb = embeddings.embed([text])
    if not emb.vectors or not emb.vectors[0]:
        return []
    return list(emb.vectors[0])


def _query_topk_ids(
    db: sqlite3.Connection,
    *,
    table: str,
    id_col: str,
    query_vec: List[float],
    k: int,
    allowed_ids: Optional[Sequence[int]] = None,
) -> List[Tuple[int, float]]:
    import json

    if not query_vec:
        return []

    k = max(1, int(k))
    query_json = json.dumps(query_vec)

    if not allowed_ids:
        rows = db.execute(
            f"""
            SELECT {id_col} AS entity_id, distance
            FROM {table}
            WHERE embedding MATCH ?
            ORDER BY distance
            LIMIT ?
            """,
            (query_json, k),
        ).fetchall()
        return [(int(r["entity_id"]), float(r["distance"])) for r in rows]

    allowed = sorted({int(v) for v in allowed_ids})
    if not allowed:
        return []

    # Keep below SQLite's default parameter limit.
    chunk_size = 900
    out: List[Tuple[int, float]] = []
    for i in range(0, len(allowed), chunk_size):
        chunk = allowed[i : i + chunk_size]
        placeholders = ",".join("?" for _ in chunk)
        rows = db.execute(
            f"""
            SELECT {id_col} AS entity_id, distance
            FROM {table}
            WHERE embedding MATCH ?
              AND {id_col} IN ({placeholders})
            ORDER BY distance
            LIMIT ?
            """,
            (query_json, *chunk, k),
        ).fetchall()
        out.extend((int(r["entity_id"]), float(r["distance"])) for r in rows)

    out.sort(key=lambda t: t[1])
    return out[:k]
