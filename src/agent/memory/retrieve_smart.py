from __future__ import annotations

import sqlite3
import time
from typing import Any, Dict, List, Tuple

from agent.providers.embeddings.base import EmbeddingProvider
from agent.memory.vector_store import embed_query, query_topk_episode_ids, query_topk_item_ids


def _recency_score(ts: int, now: int) -> float:
    # 0..1 with a soft decay (~7 days half-life-ish)
    age_s = max(0, now - int(ts))
    return 1.0 / (1.0 + (age_s / (7 * 24 * 3600)))


def retrieve_mt_smart(
    db: sqlite3.Connection,
    embeddings: EmbeddingProvider,
    *,
    user_id: str,
    session_id: str,
    user_message: str,
    limit: int = 3,
) -> List[Dict[str, Any]]:
    qv = embed_query(embeddings, user_message)
    if not qv:
        return []

    now = int(time.time())
    # vector candidates
    cand = query_topk_episode_ids(db, qv, k=8)
    if not cand:
        return []

    # join metadata
    ids = [c[0] for c in cand]
    rows = db.execute(
        f"""
        SELECT id, summary, importance, confidence, ts
        FROM episodes
        WHERE user_id = ? AND session_id = ?
          AND id IN ({",".join("?" for _ in ids)})
        """,
        (user_id, session_id, *ids),
    ).fetchall()

    by_id = {int(r["id"]): r for r in rows}

    # rerank: convert distance -> similarity proxy (simple)
    scored = []
    for eid, dist in cand:
        r = by_id.get(eid)
        if not r:
            continue
        sim = 1.0 / (1.0 + float(dist))  # crude but stable
        imp = float(r["importance"])
        rec = _recency_score(int(r["ts"]), now)
        score = 0.75 * sim + 0.15 * imp + 0.10 * rec
        scored.append((score, eid))

    scored.sort(reverse=True)
    top_ids = [eid for _, eid in scored[:limit]]

    out = []
    for eid in top_ids:
        r = by_id[eid]
        out.append(
            {
                "id": int(r["id"]),
                "summary": r["summary"],
                "importance": float(r["importance"]),
                "confidence": float(r["confidence"]),
                "ts": int(r["ts"]),
            }
        )
    return out


def retrieve_lt_smart(
    db: sqlite3.Connection,
    embeddings: EmbeddingProvider,
    *,
    user_id: str,
    user_message: str,
    limit: int = 6,
) -> List[Dict[str, Any]]:
    qv = embed_query(embeddings, user_message)
    if not qv:
        return []

    now = int(time.time())
    cand = query_topk_item_ids(db, qv, k=14)
    if not cand:
        return []

    ids = [c[0] for c in cand]
    rows = db.execute(
        f"""
        SELECT id, kind, mem_key, value, confidence, importance, ts_updated
        FROM memory_items
        WHERE user_id = ?
          AND id IN ({",".join("?" for _ in ids)})
        """,
        (user_id, *ids),
    ).fetchall()

    by_id = {int(r["id"]): r for r in rows}

    scored = []
    for mid, dist in cand:
        r = by_id.get(mid)
        if not r:
            continue
        sim = 1.0 / (1.0 + float(dist))
        imp = float(r["importance"])
        rec = _recency_score(int(r["ts_updated"]), now)
        score = 0.75 * sim + 0.20 * imp + 0.05 * rec
        scored.append((score, mid))

    scored.sort(reverse=True)
    top_ids = [mid for _, mid in scored[:limit]]

    out = []
    for mid in top_ids:
        r = by_id[mid]
        out.append(
            {
                "id": int(r["id"]),
                "kind": r["kind"],
                "mem_key": r["mem_key"],
                "value": r["value"],
                "confidence": float(r["confidence"]),
                "importance": float(r["importance"]),
                "ts_updated": int(r["ts_updated"]),
            }
        )
    return out


def render_mt(episodes: List[Dict[str, Any]]) -> str:
    return "\n".join([f"- {e['summary']}" for e in episodes]) if episodes else ""


def render_lt(items: List[Dict[str, Any]]) -> str:
    if not items:
        return ""
    lines = []
    for it in items:
        key = f" ({it['mem_key']})" if it.get("mem_key") else ""
        lines.append(f"- [{it['kind']}{key}] {it['value']}")
    return "\n".join(lines)