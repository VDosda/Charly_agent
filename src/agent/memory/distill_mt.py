from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import sqlite3

from agent.core.tracing import JSONTracer
from agent.memory import store
from agent.providers.embeddings.base import EmbeddingProvider
from agent.providers.embeddings.utils import pack_f32
from agent.providers.llm.base import LLMProvider


@dataclass(frozen=True)
class EpisodeConfig:
    """
    MT distillation config.
    """

    # Create an episode when there are at least this many new turns since last episode
    min_new_turns: int = 20

    # Summarize at most this many turns in one episode (avoid huge prompts)
    max_turns_per_episode: int = 30

    # Keep the last N turns unsummarized (avoid summarizing the very latest context)
    tail_keep_turns: int = 6

    # For distillation prompt size safety
    max_chars_per_turn: int = 2000


def maybe_create_episode(
    *,
    db: sqlite3.Connection,
    llm: LLMProvider,
    embeddings: EmbeddingProvider,
    tracer: JSONTracer,
    correlation_id: str,
    user_id: str,
    session_id: str,
    cfg: Optional[EpisodeConfig] = None,
) -> Optional[int]:
    """
    Create a MT episode if enough new ST turns accumulated.
    Returns episode_id if created, else None.
    """
    cfg = cfg or EpisodeConfig()

    last_end = store.get_last_episode_end_turn_id(db, session_id)
    new_turns = store.count_turns_after(db, session_id, last_end)

    tracer.emit(
        event="mt.check",
        level="debug",
        correlation_id=correlation_id,
        user_id=user_id,
        session_id=session_id,
        payload={
            "last_end_turn": last_end,
            "new_turns": new_turns,
            "min_new_turns": cfg.min_new_turns,
        },
    )

    if new_turns < cfg.min_new_turns:
        return None

    max_turn_id = store.get_max_turn_id(db, session_id)

    # Do not summarize the tail (keep latest turns for ST context freshness)
    effective_end = max_turn_id - cfg.tail_keep_turns
    if effective_end <= last_end:
        return None

    start_turn = last_end + 1
    end_turn = min(effective_end, start_turn + cfg.max_turns_per_episode - 1)

    turns = store.read_turns_range(db, session_id, start_turn, end_turn)
    if not turns:
        return None

    tracer.emit(
        event="mt.distill.start",
        level="info",
        correlation_id=correlation_id,
        user_id=user_id,
        session_id=session_id,
        payload={"start_turn": start_turn, "end_turn": end_turn, "turns": len(turns)},
    )

    # Build a compact transcript for the summarizer
    transcript = _render_transcript(turns, max_chars_per_turn=cfg.max_chars_per_turn)

    # Ask LLM for a structured episode (JSON) to avoid brittle parsing later
    result = llm.generate(
        messages=_distill_messages(transcript),
        tools=None,
        tool_choice="none",
    )

    episode = _parse_episode_json(result.text)

    # Fallback: if JSON parsing failed, store plain summary text
    summary = (episode.get("summary") or "").strip()
    if not summary:
        summary = (result.text or "").strip() or "(empty summary)"

    topics_json = _safe_json_dump(episode.get("topics"))
    facts_json = _safe_json_dump(episode.get("facts"))
    open_tasks_json = _safe_json_dump(episode.get("open_tasks"))

    importance = _clamp01(_to_float(episode.get("importance"), default=0.5))
    confidence = _clamp01(_to_float(episode.get("confidence"), default=0.6))

    # Embed summary (for later vector retrieval)
    emb_blob = None
    emb_model = None
    emb_dims = None
    try:
        emb = embeddings.embed([summary])
        vec = emb.vectors[0] if emb.vectors else []
        emb_model = emb.model
        emb_dims = emb.dimensions
        emb_blob = pack_f32(vec) if vec else None
    except Exception as e:
        tracer.emit(
            event="mt.embed.error",
            level="warning",
            correlation_id=correlation_id,
            user_id=user_id,
            session_id=session_id,
            payload={"error": f"{type(e).__name__}: {e}"},
        )

    episode_id = store.insert_episode(
        db,
        user_id=user_id,
        session_id=session_id,
        start_turn_id=start_turn,
        end_turn_id=end_turn,
        ts=int(time.time()),
        summary=summary,
        topics_json=topics_json,
        facts_json=facts_json,
        open_tasks_json=open_tasks_json,
        importance=importance,
        confidence=confidence,
        embedding_model=emb_model,
        embedding_dims=emb_dims,
        embedding_blob=emb_blob,
        source_turn_ids=[t["turn_id"] for t in turns],
    )

    tracer.emit(
        event="mt.distill.end",
        level="info",
        correlation_id=correlation_id,
        user_id=user_id,
        session_id=session_id,
        payload={
            "episode_id": episode_id,
            "start_turn": start_turn,
            "end_turn": end_turn,
        },
    )

    return episode_id


# -------------------------
# Helpers
# -------------------------


def _distill_messages(transcript: str) -> List[Dict[str, Any]]:
    """
    Distillation prompt: ask for STRICT JSON only.
    """
    system = (
        "You are a summarization engine for an agent memory system.\n"
        "Return STRICT JSON only. No markdown, no extra text.\n"
        "Schema:\n"
        "{\n"
        '  "summary": string,                 // 5-10 lines max\n'
        '  "topics": [string],                // up to 6\n'
        '  "facts": [{"k": string, "v": string}],       // stable decisions/facts\n'
        '  "open_tasks": [{"task": string, "status": "open"|"done"}],\n'
        '  "importance": number,              // 0..1\n'
        '  "confidence": number               // 0..1\n'
        "}\n"
        "Rules:\n"
        "- Be concise.\n"
        "- Put only actionable, durable info in facts.\n"
        "- If nothing important: keep facts/open_tasks empty and set importance low.\n"
    )

    user = "Transcript:\n" + transcript
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _render_transcript(turns: List[Dict[str, Any]], max_chars_per_turn: int) -> str:
    lines: List[str] = []
    for t in turns:
        role = t["role"]
        content = (t["content"] or "").strip()

        if len(content) > max_chars_per_turn:
            content = content[:max_chars_per_turn] + "…"

        if role == "tool":
            tool_name = t.get("tool_name") or "tool"
            lines.append(f"{t['turn_id']:04d} TOOL {tool_name}: {content}")
        else:
            lines.append(f"{t['turn_id']:04d} {role.upper()}: {content}")
    return "\n".join(lines)


def _parse_episode_json(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    s = text.strip()
    # best-effort: if model returns extra text, try to extract a JSON object
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        s = s[start : end + 1]
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _safe_json_dump(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return None


def _to_float(v: Any, default: float) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x
