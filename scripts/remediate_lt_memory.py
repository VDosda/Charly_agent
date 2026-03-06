#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sqlite3
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from agent.memory.store_lt import archive_memory_item_by_id, upsert_memory_item


@dataclass(frozen=True)
class MemoryRow:
    id: int
    user_id: str
    kind: str
    mem_key: Optional[str]
    value: str
    confidence: float
    importance: float
    ts_updated: int
    archived: int


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Remediate LT memory data by consolidating semantic duplicates into canonical "
            "English keys and archiving superseded rows."
        )
    )
    parser.add_argument("--db", required=True, help="Path to SQLite database.")
    parser.add_argument("--user-id", default=None, help="Optional user_id scope.")
    parser.add_argument("--apply", action="store_true", help="Apply changes. Default is dry-run.")
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    dry_run = not bool(args.apply)

    rows = load_rows(conn, user_id=args.user_id)
    by_user: Dict[str, List[MemoryRow]] = {}
    for row in rows:
        by_user.setdefault(row.user_id, []).append(row)

    print(f"[info] users={len(by_user)} rows={len(rows)} dry_run={dry_run}")
    for user_id, user_rows in by_user.items():
        print(f"[info] user={user_id} rows={len(user_rows)}")
        remediate_user(conn, user_id=user_id, rows=user_rows, dry_run=dry_run)

    if dry_run:
        print("[info] dry-run completed, no writes committed.")
    else:
        conn.commit()
        print("[info] remediation applied and committed.")


def load_rows(conn: sqlite3.Connection, *, user_id: Optional[str]) -> List[MemoryRow]:
    where = ["archived = 0"]
    params: List[object] = []
    if user_id:
        where.append("user_id = ?")
        params.append(user_id)
    where_sql = " AND ".join(where)

    db_rows = conn.execute(
        f"""
        SELECT id, user_id, kind, mem_key, value, confidence, importance, ts_updated, archived
        FROM memory_items
        WHERE {where_sql}
        ORDER BY ts_updated DESC, id DESC
        """
        ,
        tuple(params),
    ).fetchall()
    out: List[MemoryRow] = []
    for r in db_rows:
        out.append(
            MemoryRow(
                id=int(r["id"]),
                user_id=str(r["user_id"]),
                kind=str(r["kind"] or ""),
                mem_key=str(r["mem_key"]) if r["mem_key"] is not None else None,
                value=str(r["value"] or ""),
                confidence=float(r["confidence"] or 0.6),
                importance=float(r["importance"] or 0.5),
                ts_updated=int(r["ts_updated"] or 0),
                archived=int(r["archived"] or 0),
            )
        )
    return out


def remediate_user(conn: sqlite3.Connection, *, user_id: str, rows: Sequence[MemoryRow], dry_run: bool) -> None:
    identity_rows = [r for r in rows if r.kind == "identity"]
    if identity_rows:
        consolidate_group(
            conn,
            user_id=user_id,
            canonical_kind="identity",
            canonical_key="identity.user_name",
            canonical_value=_best_identity_value(identity_rows),
            rows=identity_rows,
            archive_reason="deduplicated_identity",
            dry_run=dry_run,
        )

    for project_slug, group in group_projects(rows).items():
        project_rows = [r for r in group if r.kind == "project"]
        if project_rows:
            consolidate_group(
                conn,
                user_id=user_id,
                canonical_kind="project",
                canonical_key=f"project.{project_slug}",
                canonical_value=f"Project {project_slug}",
                rows=project_rows,
                archive_reason="deduplicated_project",
                dry_run=dry_run,
            )

        if should_archive_project(group):
            apply_project_abandoned_transition(
                conn,
                user_id=user_id,
                project_slug=project_slug,
                rows=group,
                dry_run=dry_run,
            )


def consolidate_group(
    conn: sqlite3.Connection,
    *,
    user_id: str,
    canonical_kind: str,
    canonical_key: str,
    canonical_value: str,
    rows: Sequence[MemoryRow],
    archive_reason: str,
    dry_run: bool,
) -> None:
    keep = pick_primary(rows)
    print(
        f"[plan] consolidate kind={canonical_kind} key={canonical_key} "
        f"rows={len(rows)} keep_id={keep.id if keep else 'none'}"
    )
    if dry_run:
        return

    upserted = upsert_memory_item(
        conn,
        user_id=user_id,
        kind=canonical_kind,
        mem_key=canonical_key,
        value=canonical_value,
        confidence=max([r.confidence for r in rows] + [0.6]),
        importance=max([r.importance for r in rows] + [0.5]),
        source_note="remediation_canonical_merge",
        evidence_span="Remediation consolidation",
        source_turn_ids_json="[]",
        embedding_status="pending",
    )
    canonical_id = int(upserted.item_id)
    for row in rows:
        if int(row.id) == canonical_id:
            continue
        archived = archive_memory_item_by_id(
            conn,
            user_id=user_id,
            item_id=int(row.id),
            reason=archive_reason,
            source_note="remediation_archive_superseded",
        )
        print(
            f"[apply] archive id={row.id} reason={archive_reason} archived={archived.archived}"
        )


def group_projects(rows: Sequence[MemoryRow]) -> Dict[str, List[MemoryRow]]:
    out: Dict[str, List[MemoryRow]] = {}
    for row in rows:
        if row.kind not in {"project", "constraint", "goal"}:
            continue
        token = project_token(row)
        if not token:
            continue
        out.setdefault(token, []).append(row)
    return out


def project_token(row: MemoryRow) -> Optional[str]:
    candidates: List[str] = []
    stop = {
        "project",
        "status",
        "constraint",
        "identity",
        "assistant",
        "user",
        "name",
        "parking",
        "space",
        "management",
        "named",
        "utilisateur",
    }
    if row.mem_key:
        key = row.mem_key.lower()
        if key.startswith("project."):
            parts = key.split(".")
            if len(parts) >= 2 and parts[1]:
                candidates.append(parts[1])

    # Prefer proper nouns from value (for example Parktage) to avoid
    # generic words like "project" or "user".
    proper_nouns = [m.group(0).lower() for m in re.finditer(r"\b[A-Z][A-Za-z0-9]{3,}\b", row.value)]
    candidates.extend([p for p in proper_nouns if p not in stop])

    if row.mem_key and row.kind == "project" and not proper_nouns:
        key = row.mem_key.lower()
        for p in re_split_tokens(key):
            lowered = p.lower()
            if len(lowered) >= 4 and lowered not in stop:
                candidates.append(lowered)

    value_lower = row.value.lower()
    if row.kind in {"constraint", "goal"} and "project" in value_lower:
        for p in re_split_tokens(value_lower):
            if len(p) >= 4 and p not in stop:
                candidates.append(p)

    for c in candidates:
        if c in stop:
            continue
        return c
    return None


def should_archive_project(rows: Sequence[MemoryRow]) -> bool:
    status_rows = [r for r in rows if (r.mem_key or "").lower().endswith(".status") and "abandoned" in r.value.lower()]
    if status_rows:
        return True
    for r in rows:
        key = (r.mem_key or "").lower()
        value = r.value.lower()
        if "no_time" in key or "abandoned" in key or "cancelled" in key:
            return True
        if "status is abandoned" in value:
            return True
    return False


def apply_project_abandoned_transition(
    conn: sqlite3.Connection,
    *,
    user_id: str,
    project_slug: str,
    rows: Sequence[MemoryRow],
    dry_run: bool,
) -> None:
    status_key = f"project.{project_slug}.status"
    print(f"[plan] project_abandoned slug={project_slug} rows={len(rows)}")
    if dry_run:
        return

    upsert_memory_item(
        conn,
        user_id=user_id,
        kind="project",
        mem_key=status_key,
        value="Project status is abandoned",
        confidence=0.9,
        importance=0.9,
        source_note="remediation_project_status",
        evidence_span="Remediation inferred abandoned status",
        source_turn_ids_json="[]",
        embedding_status="pending",
    )
    for row in rows:
        if (row.mem_key or "").lower() == status_key:
            continue
        if row.kind not in {"project", "constraint", "goal"}:
            continue
        archived = archive_memory_item_by_id(
            conn,
            user_id=user_id,
            item_id=int(row.id),
            reason="project_abandoned",
            source_note="remediation_project_transition",
        )
        print(f"[apply] archive id={row.id} reason=project_abandoned archived={archived.archived}")


def pick_primary(rows: Sequence[MemoryRow]) -> Optional[MemoryRow]:
    if not rows:
        return None
    return sorted(
        rows,
        key=lambda r: (r.importance, r.confidence, r.ts_updated, r.id),
        reverse=True,
    )[0]


def _best_identity_value(rows: Sequence[MemoryRow]) -> str:
    # Keep stable canonical English representation.
    names: List[str] = []
    for row in rows:
        for tok in re_split_tokens(row.value):
            if tok and tok[0].isalpha() and tok[0].isupper() and len(tok) >= 3:
                names.append(tok)
    if names:
        return f"User name is {names[0]}"
    return "User name is set"


def re_split_tokens(text: str) -> List[str]:
    return [t for t in re.split(r"[^A-Za-z0-9]+", text or "") if t]


if __name__ == "__main__":
    main()
