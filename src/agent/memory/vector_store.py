from __future__ import annotations

import sqlite3
from typing import Any, Dict, List, Optional, Tuple

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
) -> List[Tuple[int, float]]:
    """
    Returns list of (episode_id, distance) sorted by nearest.
    For sqlite-vec, lower distance means closer.
    """
    import json
    rows = db.execute(
        """
        SELECT episode_id, distance
        FROM episodes_vec
        WHERE embedding MATCH ?
        ORDER BY distance
        LIMIT ?
        """,
        (json.dumps(query_vec), int(k)),
    ).fetchall()
    return [(int(r["episode_id"]), float(r["distance"])) for r in rows]


def query_topk_item_ids(
    db: sqlite3.Connection,
    query_vec: List[float],
    k: int = 10,
) -> List[Tuple[int, float]]:
    import json
    rows = db.execute(
        """
        SELECT item_id, distance
        FROM memory_items_vec
        WHERE embedding MATCH ?
        ORDER BY distance
        LIMIT ?
        """,
        (json.dumps(query_vec), int(k)),
    ).fetchall()
    return [(int(r["item_id"]), float(r["distance"])) for r in rows]


def embed_query(embeddings: EmbeddingProvider, text: str) -> List[float]:
    emb = embeddings.embed([text])
    if not emb.vectors or not emb.vectors[0]:
        return []
    return list(emb.vectors[0])