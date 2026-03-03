# src/agent/db/migrate.py

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable, List, Tuple


MIGRATIONS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS schema_migrations (
  version INTEGER PRIMARY KEY,
  filename TEXT NOT NULL,
  applied_ts INTEGER NOT NULL DEFAULT (strftime('%s','now'))
);
"""


def migrate(conn: sqlite3.Connection, migrations_dir: str | Path = None) -> None:
    """
    Apply SQL migrations in a deterministic order.

    - Migrations are .sql files in db/migrations/
    - Filenames must start with an integer version, e.g.:
        0001_init.sql
        0002_fts.sql
        0003_vec.sql

    Robustness goals:
    - Idempotent: already-applied migrations are skipped.
    - Transactional: each migration runs inside a transaction.
    - Deterministic ordering.
    """
    if migrations_dir is None:
        # Default: src/agent/db/migrations relative to this file
        migrations_dir = Path(__file__).resolve().parent / "migrations"
    else:
        migrations_dir = Path(migrations_dir)

    if not migrations_dir.exists():
        raise FileNotFoundError(f"Migrations directory not found: {migrations_dir}")

    _ensure_migrations_table(conn)

    applied = _get_applied_versions(conn)
    available = _list_migration_files(migrations_dir)

    to_apply = [(v, p) for (v, p) in available if v not in applied]
    if not to_apply:
        return

    for version, path in to_apply:
        sql = path.read_text(encoding="utf-8")
        _apply_one(conn, version=version, filename=path.name, sql=sql)


def _ensure_migrations_table(conn: sqlite3.Connection) -> None:
    with conn:
        conn.execute(MIGRATIONS_TABLE_SQL)


def _get_applied_versions(conn: sqlite3.Connection) -> set[int]:
    rows = conn.execute("SELECT version FROM schema_migrations").fetchall()
    return {int(r[0]) for r in rows}


def _list_migration_files(migrations_dir: Path) -> List[Tuple[int, Path]]:
    """
    Return sorted list of (version, path) for *.sql migrations.
    """
    files = sorted(migrations_dir.glob("*.sql"))
    migrations: List[Tuple[int, Path]] = []

    for p in files:
        version = _parse_version(p.name)
        migrations.append((version, p))

    # Ensure deterministic ordering by version
    migrations.sort(key=lambda x: x[0])

    # Detect duplicates (two files with same numeric version)
    seen = set()
    for v, p in migrations:
        if v in seen:
            raise ValueError(f"Duplicate migration version {v} detected in {migrations_dir}")
        seen.add(v)

    return migrations


def _parse_version(filename: str) -> int:
    """
    Parse the numeric prefix from a migration filename.

    Example: '0003_vec.sql' -> 3
    """
    prefix = filename.split("_", 1)[0]
    if not prefix.isdigit():
        raise ValueError(
            f"Invalid migration filename '{filename}'. "
            "Expected a numeric prefix like '0001_init.sql'."
        )
    return int(prefix)


def _apply_one(conn: sqlite3.Connection, version: int, filename: str, sql: str) -> None:
    """
    Apply a single migration inside its own transaction.

    Note:
    - We use executescript() to allow multiple statements.
    - If it fails, transaction is rolled back automatically by the context manager.
    """
    try:
        with conn:
            conn.executescript(sql)
            conn.execute(
                "INSERT INTO schema_migrations(version, filename) VALUES(?, ?)",
                (version, filename),
            )
    except sqlite3.Error as e:
        raise RuntimeError(
            f"Failed applying migration v{version:04d} ({filename}): {e}"
        ) from e