from __future__ import annotations

import json
import re
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Set

from agent.memory.scoring import distance_to_similarity
from agent.memory.store_lt import archive_memory_item_by_id, upsert_memory_item
from agent.memory.vector_store import query_topk_item_ids, upsert_item_vec
from agent.providers.embeddings.base import EmbeddingProvider
from agent.providers.embeddings.utils import pack_f32


@dataclass(frozen=True)
class MemoryManagerConfig:
    min_importance: float = 0.50
    min_confidence: float = 0.60
    semantic_dedupe_top_k: int = 6
    semantic_dedupe_min_similarity: float = 0.92


@dataclass(frozen=True)
class MemoryIntent:
    """
    Structured intent extracted by the LT LLM prompt.

    Invariant: the LLM emits *intent semantics* only. Database semantics are
    resolved by MemoryManager, not by prompt-level free text.
    """

    action: str
    entity_type: str
    entity_name: str = ""
    description: str = ""
    attributes: Dict[str, Any] = field(default_factory=dict)
    evidence_span: str = ""
    source_turn_ids: List[int] = field(default_factory=list)
    confidence: float = 0.6
    importance: float = 0.5


@dataclass(frozen=True)
class MemoryActionResult:
    applied: bool
    action: str
    item_id: Optional[int] = None
    skip_reason: Optional[str] = None
    kind: Optional[str] = None
    mem_key: Optional[str] = None
    embedding_status: Optional[str] = None
    transition: Optional[str] = None


@dataclass(frozen=True)
class _ResolvedTarget:
    item_id: Optional[int]
    mem_key: Optional[str]
    kind: Optional[str] = None
    similarity: Optional[float] = None
    distance: Optional[float] = None


class MemoryManager:
    """
    Memory Manager: robust LT orchestration between LLM intents and DB writes.

    Engineering invariants:
    - No phrase matching over user text to trigger memory actions.
    - Persisted LT representation must be canonical and English.
    - Entity resolution happens before write to prevent duplicates.
    - State transitions (for example project abandonment) are explicit.
    """

    _ALLOWED_ACTIONS = {"create", "update", "archive", "none"}
    _KIND_MAP = {
        "identity": "identity",
        "preference": "preference",
        "constraint": "constraint",
        "goal": "goal",
        "procedure": "procedure",
        "assistant_behavior": "procedure",
        "project": "project",
        "other": "other",
    }
    _PROJECT_TERMINAL_STATUSES = {"abandoned", "cancelled", "dropped", "on_hold"}
    _NON_CANONICAL_TOKENS = {
        "utilisateur",
        "projet",
        "souvenir",
        "rappelle",
        "nom",
        "bonjour",
        "salut",
    }

    def __init__(
        self,
        *,
        db: sqlite3.Connection,
        embeddings: EmbeddingProvider,
        tracer: Any,
        cfg: MemoryManagerConfig,
        semantic_query_fn: Callable[..., List[tuple[int, float]]] = query_topk_item_ids,
    ) -> None:
        self.db = db
        self.embeddings = embeddings
        self.tracer = tracer
        self.cfg = cfg
        self.semantic_query_fn = semantic_query_fn
        self._existing_index: Dict[str, Any] = {
            "by_id": {},
            "ids_by_kind": {},
            "id_by_kind_key": {},
            "ids_by_entity": {},
        }
        self._existing_index_user_id: Optional[str] = None

    def process_intent(
        self,
        *,
        correlation_id: str,
        user_id: str,
        session_id: str,
        intent: MemoryIntent,
        st_turn_ids: Set[int],
        source_episode_id: Optional[int],
        source_note: str,
        mode: str,
        episode_id_for_trace: Optional[int],
    ) -> MemoryActionResult:
        self._ensure_index(user_id=user_id)

        kind = self._normalize_kind(intent.entity_type)
        normalized = self.normalize_candidate(kind=kind, intent=intent)
        entity_slug = normalized["entity_slug"]
        canonical_value = normalized["canonical_value"]
        canonical_status = normalized["status"]
        archive_reason = normalized["archive_reason"]
        valid_source_turn_ids = [tid for tid in intent.source_turn_ids if tid in st_turn_ids]
        transition = self.decide_memory_transition(
            action=intent.action,
            kind=kind,
            status=canonical_status,
        )
        mem_key_candidate = self.normalize_memory_key(
            kind=kind,
            entity_slug=entity_slug,
            status=canonical_status,
            resolved_target=None,
        )

        skip_reason = self.validate_memory_action(
            action=intent.action,
            transition=transition,
            kind=kind,
            value=canonical_value,
            mem_key=mem_key_candidate,
            evidence_span=intent.evidence_span,
            source_turn_ids=valid_source_turn_ids,
            confidence=float(intent.confidence),
            importance=float(intent.importance),
        )
        if skip_reason:
            return MemoryActionResult(
                applied=False,
                action=intent.action,
                skip_reason=skip_reason,
                kind=kind,
                transition=transition,
            )

        resolved_target = self.resolve_target_memory(
            user_id=user_id,
            kind=kind,
            value=canonical_value,
            entity_slug=entity_slug,
            mem_key_candidate=mem_key_candidate,
            transition=transition,
            correlation_id=correlation_id,
            session_id=session_id,
            mode=mode,
            episode_id_for_trace=episode_id_for_trace,
        )
        mem_key = self.normalize_memory_key(
            kind=kind,
            entity_slug=entity_slug,
            status=canonical_status,
            resolved_target=resolved_target,
        )

        if transition == "archive":
            return self.apply_memory_action(
                correlation_id=correlation_id,
                user_id=user_id,
                session_id=session_id,
                source_episode_id=source_episode_id,
                source_note=source_note,
                intent=intent,
                kind=kind,
                value=canonical_value,
                entity_slug=entity_slug,
                status=canonical_status,
                mem_key=mem_key,
                resolved_target=resolved_target,
                source_turn_ids=valid_source_turn_ids,
                archive_reason=archive_reason,
                transition=transition,
            )

        return self.apply_memory_action(
            correlation_id=correlation_id,
            user_id=user_id,
            session_id=session_id,
            source_episode_id=source_episode_id,
            source_note=source_note,
            intent=intent,
            kind=kind,
            value=canonical_value,
            entity_slug=entity_slug,
            status=canonical_status,
            mem_key=mem_key,
            resolved_target=resolved_target,
            source_turn_ids=valid_source_turn_ids,
            archive_reason=archive_reason,
            transition=transition,
        )

    def normalize_candidate(self, *, kind: str, intent: MemoryIntent) -> Dict[str, str]:
        attrs = intent.attributes or {}
        raw_entity_name = (
            str(intent.entity_name or "").strip()
            or str(attrs.get("entity_name") or "").strip()
            or str(attrs.get("name") or "").strip()
            or str(attrs.get("target") or "").strip()
        )
        raw_value = ""
        for k in ("canonical_value_en", "canonical_value", "normalized_value", "value"):
            if attrs.get(k) is not None and str(attrs.get(k)).strip():
                raw_value = str(attrs.get(k)).strip()
                break
        if not raw_value:
            raw_value = str(intent.description or "").strip()

        raw_status = str(attrs.get("status") or attrs.get("state") or attrs.get("lifecycle") or "").strip()
        status = self._normalize_status(raw_status)
        entity_slug = self._canonical_entity_slug(
            kind=kind,
            raw_entity_name=raw_entity_name,
            raw_value=raw_value,
            attrs=attrs,
        )
        canonical_value = self._canonical_value(
            kind=kind,
            entity_slug=entity_slug,
            status=status,
            raw_value=raw_value,
        )
        archive_reason = str(attrs.get("archive_reason") or attrs.get("reason") or "").strip()
        if not archive_reason:
            archive_reason = "state_transition_archive"
        archive_reason = self._sanitize_english_text(archive_reason, fallback="state_transition_archive")

        return {
            "entity_slug": entity_slug,
            "canonical_value": canonical_value,
            "status": status,
            "archive_reason": archive_reason,
        }

    def resolve_target_memory(
        self,
        *,
        user_id: str,
        kind: str,
        value: str,
        entity_slug: str,
        mem_key_candidate: str,
        transition: str,
        correlation_id: str,
        session_id: str,
        mode: str,
        episode_id_for_trace: Optional[int],
    ) -> _ResolvedTarget:
        exact_id = self._existing_index["id_by_kind_key"].get((kind, mem_key_candidate))
        if exact_id is not None:
            row = self._existing_index["by_id"].get(int(exact_id)) or {}
            return _ResolvedTarget(
                item_id=int(exact_id),
                mem_key=row.get("mem_key"),
                kind=row.get("kind"),
            )

        entity_ids = list(self._existing_index.get("ids_by_entity", {}).get(entity_slug, []))
        if entity_ids:
            row = self._best_entity_match(entity_ids=entity_ids, preferred_kind=kind)
            if row:
                return _ResolvedTarget(
                    item_id=int(row["id"]),
                    mem_key=row.get("mem_key"),
                    kind=row.get("kind"),
                )

        semantic_match = self._semantic_match(kind=kind, value=value)
        if semantic_match is not None:
            if semantic_match.mem_key and semantic_match.mem_key != mem_key_candidate:
                self.tracer.emit(
                    event="lt.item.semantic_merge",
                    level="debug",
                    correlation_id=correlation_id,
                    user_id=user_id,
                    session_id=session_id,
                    payload={
                        "mode": mode,
                        "episode_id": episode_id_for_trace,
                        "match_item_id": semantic_match.item_id,
                        "match_mem_key": semantic_match.mem_key,
                        "match_kind": semantic_match.kind,
                        "match_similarity": semantic_match.similarity,
                        "match_distance": semantic_match.distance,
                        "candidate_mem_key": mem_key_candidate,
                        "resolved_mem_key": semantic_match.mem_key,
                        "transition": transition,
                    },
                )
            return semantic_match

        return _ResolvedTarget(item_id=None, mem_key=None, kind=None)

    def normalize_memory_key(
        self,
        *,
        kind: str,
        entity_slug: str,
        status: str,
        resolved_target: Optional[_ResolvedTarget],
    ) -> str:
        if resolved_target and resolved_target.kind == kind and resolved_target.mem_key:
            if self._is_canonical_key(str(resolved_target.mem_key), expected_kind=kind):
                return str(resolved_target.mem_key)

        if kind == "identity":
            if entity_slug in {"name", "user_name", "username", "user"}:
                return "identity.user_name"
            return f"identity.{entity_slug}"
        if kind == "project":
            if status:
                return f"project.{entity_slug}.status"
            return f"project.{entity_slug}"
        if kind == "preference":
            return f"preference.{entity_slug}"
        if kind == "constraint":
            return f"constraint.{entity_slug}"
        if kind == "goal":
            return f"goal.{entity_slug}"
        if kind == "procedure":
            return f"procedure.{entity_slug}"
        return f"other.{entity_slug}"

    def decide_memory_transition(self, *, action: str, kind: str, status: str) -> str:
        normalized_action = (action or "").strip().lower()
        if normalized_action == "archive":
            return "archive"
        if normalized_action in {"create", "update"}:
            if kind == "project" and status in self._PROJECT_TERMINAL_STATUSES:
                return "archive"
            return normalized_action
        return "none"

    def apply_memory_action(
        self,
        *,
        correlation_id: str,
        user_id: str,
        session_id: str,
        source_episode_id: Optional[int],
        source_note: str,
        intent: MemoryIntent,
        kind: str,
        value: str,
        entity_slug: str,
        status: str,
        mem_key: str,
        resolved_target: _ResolvedTarget,
        source_turn_ids: Sequence[int],
        archive_reason: str,
        transition: str,
    ) -> MemoryActionResult:
        if transition == "archive":
            return self._apply_archive_transition(
                correlation_id=correlation_id,
                user_id=user_id,
                session_id=session_id,
                source_episode_id=source_episode_id,
                source_note=source_note,
                intent=intent,
                kind=kind,
                entity_slug=entity_slug,
                mem_key=mem_key,
                source_turn_ids=source_turn_ids,
                archive_reason=archive_reason,
                resolved_target=resolved_target,
            )

        target_item_id = None
        if (
            resolved_target.item_id is not None
            and resolved_target.kind == kind
            and str(resolved_target.mem_key or "") == mem_key
        ):
            target_item_id = int(resolved_target.item_id)

        write = self._upsert_canonical_memory(
            user_id=user_id,
            session_id=session_id,
            source_episode_id=source_episode_id,
            source_note=source_note,
            kind=kind,
            mem_key=mem_key,
            value=value,
            confidence=float(intent.confidence),
            importance=float(intent.importance),
            evidence_span=intent.evidence_span,
            source_turn_ids=source_turn_ids,
            target_item_id=target_item_id,
        )

        # If an old row represented the same fact but with a non-canonical key,
        # archive it after canonical write so duplicates do not keep polluting retrieval.
        if (
            resolved_target.item_id is not None
            and target_item_id is None
            and resolved_target.kind == kind
            and str(resolved_target.mem_key or "") != mem_key
        ):
            archived = archive_memory_item_by_id(
                self.db,
                user_id=user_id,
                item_id=int(resolved_target.item_id),
                reason="superseded_by_canonical_key",
                source_session_id=session_id,
                source_episode_id=source_episode_id,
                source_note=source_note,
            )
            if archived.archived:
                self._remove_from_index(item_id=int(resolved_target.item_id))

        self._register_in_index(item_id=int(write["item_id"]), kind=kind, mem_key=mem_key)
        return MemoryActionResult(
            applied=True,
            action=str(write["action"]),
            item_id=int(write["item_id"]),
            kind=kind,
            mem_key=mem_key,
            embedding_status=str(write["embedding_status"]),
            transition=transition,
        )

    def validate_memory_action(
        self,
        *,
        action: str,
        transition: str,
        kind: str,
        value: str,
        mem_key: str,
        evidence_span: str,
        source_turn_ids: Sequence[int],
        confidence: float,
        importance: float,
    ) -> Optional[str]:
        normalized_action = (action or "").strip().lower()
        if normalized_action not in self._ALLOWED_ACTIONS:
            return "unsupported_action"
        if transition == "none":
            return "no_action"
        if not evidence_span:
            return "missing_evidence_span"
        if not source_turn_ids:
            return "missing_source_turn_ids"
        if confidence < float(self.cfg.min_confidence):
            return "confidence_below_threshold"
        if importance < float(self.cfg.min_importance):
            return "importance_below_threshold"
        if not kind:
            return "missing_kind"
        if transition != "archive" and not value:
            return "missing_value"
        if transition != "archive" and not self._is_probably_english(value):
            return "non_english_canonical_value"
        if not mem_key or not self._is_canonical_key(mem_key, expected_kind=kind):
            return "invalid_canonical_mem_key"
        return None

    def _apply_archive_transition(
        self,
        *,
        correlation_id: str,
        user_id: str,
        session_id: str,
        source_episode_id: Optional[int],
        source_note: str,
        intent: MemoryIntent,
        kind: str,
        entity_slug: str,
        mem_key: str,
        source_turn_ids: Sequence[int],
        archive_reason: str,
        resolved_target: _ResolvedTarget,
    ) -> MemoryActionResult:
        target_ids: Set[int] = set()
        if resolved_target.item_id is not None:
            target_ids.add(int(resolved_target.item_id))
        for item_id in self._existing_index.get("ids_by_entity", {}).get(entity_slug, []):
            row = self._existing_index["by_id"].get(int(item_id))
            if not row:
                continue
            if row.get("kind") in {"project", "constraint", "goal"}:
                target_ids.add(int(item_id))

        if not target_ids:
            return MemoryActionResult(
                applied=False,
                action="archive",
                skip_reason="archive_target_not_found",
                kind=kind,
                mem_key=mem_key,
                transition="archive",
            )

        status_key = f"project.{entity_slug}.status" if kind == "project" else ""
        status_item_id: Optional[int] = None
        if kind == "project":
            # Keep one active status fact in canonical English while archiving
            # outdated project facts. This preserves lifecycle information and
            # prevents abandoned project details from leaking into active context.
            status_write = self._upsert_canonical_memory(
                user_id=user_id,
                session_id=session_id,
                source_episode_id=source_episode_id,
                source_note=source_note,
                kind="project",
                mem_key=status_key,
                value="Project status is abandoned",
                confidence=float(intent.confidence),
                importance=max(float(intent.importance), 0.8),
                evidence_span=intent.evidence_span,
                source_turn_ids=source_turn_ids,
                target_item_id=None,
            )
            status_item_id = int(status_write["item_id"])
            self._register_in_index(item_id=status_item_id, kind="project", mem_key=status_key)

        archived_any = False
        for item_id in sorted(target_ids):
            row = self._existing_index["by_id"].get(int(item_id))
            if not row:
                continue
            if status_item_id is not None and int(item_id) == int(status_item_id):
                continue
            archived = archive_memory_item_by_id(
                self.db,
                user_id=user_id,
                item_id=int(item_id),
                reason=archive_reason,
                source_session_id=session_id,
                source_episode_id=source_episode_id,
                source_note=source_note,
            )
            if archived.archived:
                archived_any = True
                self._remove_from_index(item_id=int(item_id))

        if not archived_any:
            return MemoryActionResult(
                applied=False,
                action="archive",
                skip_reason="already_archived_or_missing",
                kind=kind,
                mem_key=mem_key,
                transition="archive",
            )

        self.tracer.emit(
            event="lt.item.archive.transition",
            level="info",
            correlation_id=correlation_id,
            user_id=user_id,
            session_id=session_id,
            payload={
                "entity_slug": entity_slug,
                "archived_count": len(target_ids),
                "status_item_id": status_item_id,
            },
        )
        return MemoryActionResult(
            applied=True,
            action="archived",
            item_id=status_item_id if status_item_id is not None else next(iter(target_ids)),
            kind=kind,
            mem_key=status_key or mem_key,
            transition="archive",
        )

    def _upsert_canonical_memory(
        self,
        *,
        user_id: str,
        session_id: str,
        source_episode_id: Optional[int],
        source_note: str,
        kind: str,
        mem_key: str,
        value: str,
        confidence: float,
        importance: float,
        evidence_span: str,
        source_turn_ids: Sequence[int],
        target_item_id: Optional[int],
    ) -> Dict[str, Any]:
        source_turn_ids_json = self._json_int_list(source_turn_ids)
        embedding_payload = self._build_embedding_payload(
            kind=kind,
            mem_key_hint=mem_key,
            value=value,
            evidence_span=evidence_span,
        )

        upsert_result = upsert_memory_item(
            self.db,
            user_id=user_id,
            kind=kind,
            mem_key=mem_key,
            value=value,
            confidence=self._clamp01(float(confidence)),
            importance=self._clamp01(float(importance)),
            source_session_id=session_id,
            source_episode_id=source_episode_id,
            source_note=source_note,
            evidence_span=evidence_span,
            source_turn_ids_json=source_turn_ids_json,
            embedding_model=embedding_payload["embedding_model"],
            embedding_dims=embedding_payload["embedding_dims"],
            embedding_blob=embedding_payload["embedding_blob"],
            embedding_status=embedding_payload["embedding_status"],
            last_embedding_error=embedding_payload["last_embedding_error"],
            embedding_retry_count=embedding_payload["embedding_retry_count"],
            embedding_last_attempt_ts=embedding_payload["embedding_last_attempt_ts"],
            embedding_next_retry_ts=embedding_payload["embedding_next_retry_ts"],
            target_item_id=target_item_id,
        )

        if embedding_payload["vec"]:
            try:
                upsert_item_vec(self.db, int(upsert_result.item_id), embedding_payload["vec"])
            except Exception:
                pass

        return {
            "item_id": int(upsert_result.item_id),
            "action": str(upsert_result.action),
            "embedding_status": str(embedding_payload["embedding_status"]),
        }

    def _ensure_index(self, *, user_id: str) -> None:
        if self._existing_index_user_id == user_id:
            return
        self._existing_index = self._read_existing_memory_index(user_id=user_id)
        self._existing_index_user_id = user_id

    def _read_existing_memory_index(self, *, user_id: str) -> Dict[str, Any]:
        rows = self.db.execute(
            """
            SELECT id, kind, mem_key, value
            FROM memory_items
            WHERE user_id = ? AND archived = 0
            """,
            (user_id,),
        ).fetchall()

        by_id: Dict[int, Dict[str, Any]] = {}
        ids_by_kind: Dict[str, List[int]] = {}
        id_by_kind_key: Dict[tuple[str, str], int] = {}
        ids_by_entity: Dict[str, List[int]] = {}

        for r in rows:
            item_id = int(r["id"])
            kind = str(r["kind"] or "").strip()
            mem_key = str(r["mem_key"] or "").strip() or None
            value = str(r["value"] or "").strip()

            by_id[item_id] = {"id": item_id, "kind": kind, "mem_key": mem_key, "value": value}
            ids_by_kind.setdefault(kind, []).append(item_id)
            if mem_key is not None:
                id_by_kind_key[(kind, mem_key)] = item_id

            for token in self._entity_tokens_from_row(kind=kind, mem_key=mem_key, value=value):
                ids_by_entity.setdefault(token, []).append(item_id)

        return {
            "by_id": by_id,
            "ids_by_kind": ids_by_kind,
            "id_by_kind_key": id_by_kind_key,
            "ids_by_entity": ids_by_entity,
        }

    def _register_in_index(self, *, item_id: int, kind: str, mem_key: str) -> None:
        row = self._existing_index["by_id"].get(int(item_id), {})
        value = str(row.get("value") or "")
        self._existing_index["by_id"][int(item_id)] = {
            "id": int(item_id),
            "kind": kind,
            "mem_key": mem_key,
            "value": value,
        }
        ids = self._existing_index["ids_by_kind"].setdefault(kind, [])
        if int(item_id) not in ids:
            ids.append(int(item_id))
        self._existing_index["id_by_kind_key"][(kind, mem_key)] = int(item_id)

        for token in self._entity_tokens_from_row(kind=kind, mem_key=mem_key, value=value):
            entity_ids = self._existing_index["ids_by_entity"].setdefault(token, [])
            if int(item_id) not in entity_ids:
                entity_ids.append(int(item_id))

    def _remove_from_index(self, *, item_id: int) -> None:
        row = self._existing_index["by_id"].pop(int(item_id), None)
        if not row:
            return
        kind = str(row.get("kind") or "")
        mem_key = row.get("mem_key")
        kind_ids = self._existing_index["ids_by_kind"].get(kind, [])
        self._existing_index["ids_by_kind"][kind] = [x for x in kind_ids if int(x) != int(item_id)]
        if mem_key is not None:
            self._existing_index["id_by_kind_key"].pop((kind, mem_key), None)
        for token, ids in list(self._existing_index.get("ids_by_entity", {}).items()):
            filtered = [x for x in ids if int(x) != int(item_id)]
            if filtered:
                self._existing_index["ids_by_entity"][token] = filtered
            else:
                self._existing_index["ids_by_entity"].pop(token, None)

    def _best_entity_match(self, *, entity_ids: Sequence[int], preferred_kind: str) -> Optional[Dict[str, Any]]:
        best: Optional[Dict[str, Any]] = None
        best_rank = 10
        for item_id in entity_ids:
            row = self._existing_index["by_id"].get(int(item_id))
            if not row:
                continue
            rank = 1 if row.get("kind") == preferred_kind else 2
            if best is None or rank < best_rank:
                best = row
                best_rank = rank
        return best

    def _semantic_match(self, *, kind: str, value: str) -> Optional[_ResolvedTarget]:
        allowed_kinds = self._semantic_kinds(kind)
        allowed_ids: List[int] = []
        for k in allowed_kinds:
            allowed_ids.extend(self._existing_index.get("ids_by_kind", {}).get(k, []))
        allowed_ids = sorted({int(x) for x in allowed_ids})
        if not allowed_ids:
            return None

        vec = self._embed_value(value)
        if not vec:
            return None

        try:
            neighbors = self.semantic_query_fn(
                self.db,
                list(vec),
                k=max(1, int(self.cfg.semantic_dedupe_top_k)),
                allowed_ids=allowed_ids,
            )
        except Exception:
            return None

        by_id = self._existing_index.get("by_id", {})
        for item_id, distance in neighbors:
            row = by_id.get(int(item_id))
            if row is None:
                continue
            similarity = distance_to_similarity(float(distance))
            if similarity < float(self.cfg.semantic_dedupe_min_similarity):
                continue
            return _ResolvedTarget(
                item_id=int(item_id),
                mem_key=row.get("mem_key"),
                kind=row.get("kind"),
                similarity=float(similarity),
                distance=float(distance),
            )
        return None

    def _semantic_kinds(self, kind: str) -> Set[str]:
        if kind == "project":
            return {"project", "constraint", "goal", "other"}
        if kind == "identity":
            return {"identity", "preference", "constraint"}
        return {kind}

    def _entity_tokens_from_row(self, *, kind: str, mem_key: Optional[str], value: str) -> List[str]:
        out: List[str] = []
        if mem_key:
            key_parts = [p for p in re.split(r"[._]", mem_key.lower()) if p]
            for p in key_parts:
                if len(p) >= 4 and p not in {"project", "status", "identity", "preference", "constraint"}:
                    out.append(p)
            if mem_key.startswith("project."):
                parts = mem_key.split(".")
                if len(parts) >= 2 and parts[1]:
                    out.append(parts[1].lower())

        for token in re.findall(r"[A-Za-z0-9]{4,}", value):
            lowered = token.lower()
            if lowered not in self._NON_CANONICAL_TOKENS:
                out.append(lowered)

        deduped: List[str] = []
        seen = set()
        for token in out:
            if token in seen:
                continue
            seen.add(token)
            deduped.append(token)
        return deduped

    def _build_embedding_payload(
        self,
        *,
        kind: str,
        mem_key_hint: str,
        value: str,
        evidence_span: str,
    ) -> Dict[str, Any]:
        emb_status = "ready"
        emb_error = None
        emb_retry_count = 0
        emb_last_attempt_ts = int(time.time())
        emb_next_retry_ts = None
        vec: List[float] = []
        emb_blob = None
        emb_model = None
        emb_dims = None
        text = f"{kind}: {mem_key_hint} {value}\nEvidence: {evidence_span}".strip()
        try:
            emb = self.embeddings.embed([text])
            vec = emb.vectors[0] if emb.vectors else []
            emb_model = emb.model
            emb_dims = emb.dimensions
            emb_blob = pack_f32(vec) if vec else None
            if not vec:
                emb_status = "pending"
                emb_error = "Embedding provider returned an empty vector"
                emb_next_retry_ts = emb_last_attempt_ts + self._retry_delay_seconds(0)
        except Exception as e:
            emb_status = "pending"
            emb_error = f"{type(e).__name__}: {e}"
            emb_next_retry_ts = emb_last_attempt_ts + self._retry_delay_seconds(0)

        return {
            "vec": vec,
            "embedding_model": emb_model,
            "embedding_dims": emb_dims,
            "embedding_blob": emb_blob,
            "embedding_status": emb_status,
            "last_embedding_error": emb_error,
            "embedding_retry_count": emb_retry_count,
            "embedding_last_attempt_ts": emb_last_attempt_ts,
            "embedding_next_retry_ts": emb_next_retry_ts,
        }

    def _embed_value(self, value: str) -> List[float]:
        try:
            emb = self.embeddings.embed([value])
            return list(emb.vectors[0]) if emb.vectors else []
        except Exception:
            return []

    def _normalize_kind(self, entity_type: str) -> str:
        key = (entity_type or "").strip().lower()
        return self._KIND_MAP.get(key, "other")

    def _canonical_entity_slug(
        self,
        *,
        kind: str,
        raw_entity_name: str,
        raw_value: str,
        attrs: Dict[str, Any],
    ) -> str:
        base = raw_entity_name or str(attrs.get("key_hint") or "").strip() or str(attrs.get("topic") or "").strip()
        if not base:
            base = raw_value
        slug = self._slugify(base)
        if kind == "identity" and slug in {"name", "user", "username", "user_name", "identity"}:
            return "user_name"
        if kind == "identity" and "name" in slug:
            return "user_name"
        if not slug:
            return "general"
        return slug

    def _canonical_value(
        self,
        *,
        kind: str,
        entity_slug: str,
        status: str,
        raw_value: str,
    ) -> str:
        if kind == "identity" and entity_slug == "user_name":
            # Keep identity name value concise and canonical.
            name_candidate = raw_value.strip()
            if name_candidate.lower().startswith("user name is "):
                name_candidate = name_candidate[len("user name is ") :].strip()
            if re.fullmatch(r"[A-Za-z][A-Za-z0-9 _\\-]{0,40}", name_candidate):
                return f"User name is {name_candidate.strip()}"
            return "User name is set"
        if kind == "project":
            if status:
                return f"Project status is {status}"
            if entity_slug != "general":
                return f"Project {entity_slug}"
        if kind == "preference" and entity_slug == "language":
            lower = raw_value.strip().lower()
            if lower in {"fr", "french"}:
                return "Preferred language is French"
            if lower in {"en", "english"}:
                return "Preferred language is English"
        sanitized = self._sanitize_english_text(raw_value, fallback=f"{kind} {entity_slug}")
        return sanitized

    def _normalize_status(self, raw_status: str) -> str:
        normalized = self._slugify(raw_status).replace("_", " ")
        if not normalized:
            return ""
        status_map = {
            "abandon": "abandoned",
            "abandoned": "abandoned",
            "cancelled": "cancelled",
            "canceled": "cancelled",
            "dropped": "dropped",
            "on hold": "on_hold",
            "paused": "on_hold",
            "inactive": "on_hold",
            "active": "active",
            "in progress": "active",
            "planned": "planned",
            "done": "completed",
            "completed": "completed",
        }
        return status_map.get(normalized, normalized.replace(" ", "_"))

    def _sanitize_english_text(self, text: str, *, fallback: str) -> str:
        cleaned = " ".join(str(text or "").strip().split())
        if not cleaned:
            cleaned = fallback
        cleaned = cleaned.encode("ascii", errors="ignore").decode("ascii")
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if not cleaned:
            cleaned = fallback
        return cleaned

    def _slugify(self, text: str) -> str:
        lowered = str(text or "").strip().lower()
        lowered = lowered.encode("ascii", errors="ignore").decode("ascii")
        token = re.sub(r"[^a-z0-9]+", "_", lowered)
        token = re.sub(r"_+", "_", token).strip("_")
        return token

    def _is_probably_english(self, text: str) -> bool:
        cleaned = str(text or "").strip()
        if not cleaned:
            return False
        if cleaned.encode("ascii", errors="ignore").decode("ascii") != cleaned:
            return False
        tokens = [t.lower() for t in re.findall(r"[A-Za-z]+", cleaned)]
        for token in tokens:
            if token in self._NON_CANONICAL_TOKENS:
                return False
        return True

    def _is_canonical_key(self, key: str, *, expected_kind: str) -> bool:
        normalized = str(key or "").strip().lower()
        if not normalized:
            return False
        if not normalized.startswith(f"{expected_kind}."):
            return False
        if not re.fullmatch(r"[a-z]+(?:\.[a-z0-9_]+)+", normalized):
            return False
        for token in re.findall(r"[a-z]+", normalized):
            if token in self._NON_CANONICAL_TOKENS:
                return False
        return True

    def _json_int_list(self, values: Sequence[int]) -> str:
        out: List[int] = []
        seen = set()
        for v in values:
            iv = int(v)
            if iv in seen:
                continue
            seen.add(iv)
            out.append(iv)
        return json.dumps(out, ensure_ascii=False)

    def _clamp01(self, x: float) -> float:
        return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x

    def _retry_delay_seconds(self, retry_count: int) -> int:
        base = 30
        capped = min(3600, base * (2 ** max(0, int(retry_count))))
        return int(capped)
