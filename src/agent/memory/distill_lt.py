from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import sqlite3

from agent.core.tracing import JSONTracer
from agent.memory.store_lt import upsert_memory_item
from agent.memory.vector_store import upsert_item_vec
from agent.providers.embeddings.base import EmbeddingProvider
from agent.providers.embeddings.utils import pack_f32
from agent.providers.llm.base import LLMProvider


@dataclass(frozen=True)
class LTConfig:
    max_items: int = 12
    min_importance: float = 0.45


def maybe_distill_profile_from_episode(
    *,
    db: sqlite3.Connection,
    llm: LLMProvider,
    embeddings: EmbeddingProvider,
    tracer: JSONTracer,
    correlation_id: str,
    user_id: str,
    session_id: str,
    episode_id: int,
    cfg: Optional[LTConfig] = None,
) -> int:
    """
    Extract LT memory items from an episode summary and upsert them.
    Returns number of upserted items.
    """
    cfg = cfg or LTConfig()

    ep = db.execute(
        "SELECT summary, topics_json, facts_json, open_tasks_json FROM episodes WHERE id = ?",
        (episode_id,),
    ).fetchone()
    if not ep:
        return 0

    episode_payload = {
        "summary": ep["summary"],
        "topics": _safe_load_json(ep["topics_json"]),
        "facts": _safe_load_json(ep["facts_json"]),
        "open_tasks": _safe_load_json(ep["open_tasks_json"]),
    }

    tracer.emit(
        event="lt.distill.start",
        level="info",
        correlation_id=correlation_id,
        user_id=user_id,
        session_id=session_id,
        payload={"episode_id": episode_id},
    )

    res = llm.generate(
        messages=_lt_messages(episode_payload, max_items=cfg.max_items),
        tools=None,
        tool_choice="none",
    )

    items = _parse_items_json(res.text)

    upserted = 0
    for it in items[: cfg.max_items]:
        kind = str(it.get("kind") or "").strip()
        mem_key = it.get("key")
        mem_key = str(mem_key).strip() if mem_key is not None else None
        value = str(it.get("value") or "").strip()
        confidence = _clamp01(_to_float(it.get("confidence"), 0.6))
        importance = _clamp01(_to_float(it.get("importance"), 0.5))

        if not kind or not value:
            continue
        if importance < cfg.min_importance:
            continue

        vec: List[float] = []
        emb_blob = None
        emb_model = None
        emb_dims = None
        try:
            emb = embeddings.embed([f"{kind}: {mem_key or ''} {value}".strip()])
            vec = emb.vectors[0] if emb.vectors else []
            emb_model = emb.model
            emb_dims = emb.dimensions
            emb_blob = pack_f32(vec) if vec else None
        except Exception as e:
            tracer.emit(
                event="lt.embed.error",
                level="warning",
                correlation_id=correlation_id,
                user_id=user_id,
                session_id=session_id,
                payload={"error": f"{type(e).__name__}: {e}"},
            )

        item_id = upsert_memory_item(
            db,
            user_id=user_id,
            kind=kind,
            mem_key=mem_key,
            value=value,
            confidence=confidence,
            importance=importance,
            source_session_id=session_id,
            source_episode_id=episode_id,
            source_note="distilled_from_episode",
            embedding_model=emb_model,
            embedding_dims=emb_dims,
            embedding_blob=emb_blob,
        )
        try:
            upsert_item_vec(db, item_id, vec)
        except Exception as e:
            tracer.emit(
                event="lt.vec.upsert.error",
                level="warning",
                correlation_id=correlation_id,
                user_id=user_id,
                session_id=session_id,
                payload={"item_id": item_id, "error": f"{type(e).__name__}: {e}"},
            )
        upserted += 1

    tracer.emit(
        event="lt.distill.end",
        level="info",
        correlation_id=correlation_id,
        user_id=user_id,
        session_id=session_id,
        payload={"episode_id": episode_id, "upserted": upserted},
    )

    return upserted


def _lt_messages(episode_payload: Dict[str, Any], max_items: int) -> List[Dict[str, Any]]:
    system = (
        "You extract LONG-TERM memory items for an agent.\n"
        "Return STRICT JSON only. No markdown.\n"
        "Return an array of items with schema:\n"
        "{\n"
        '  "kind": "identity"|"preference"|"constraint"|"goal"|"procedure"|"project",\n'
        '  "key": string|null,   // stable identifier for upsert when possible\n'
        '  "value": string,      // canonical memory text\n'
        '  "confidence": number, // 0..1\n'
        '  "importance": number  // 0..1\n'
        "}\n"
        f"Max items: {max_items}.\n"
        "Rules:\n"
        "- Only durable facts/preferences/constraints/procedures.\n"
        "- Avoid ephemeral details.\n"
        "- If unsure, lower confidence.\n"
    )
    user = "Episode:\n" + json.dumps(episode_payload, ensure_ascii=False)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _parse_items_json(text: str) -> List[Dict[str, Any]]:
    if not text:
        return []
    s = text.strip()
    start = s.find("[")
    end = s.rfind("]")
    if start != -1 and end != -1 and end > start:
        s = s[start : end + 1]
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, list) else []
    except Exception:
        return []


def _safe_load_json(s: Optional[str]) -> Any:
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        return None


def _to_float(v: Any, default: float) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x
