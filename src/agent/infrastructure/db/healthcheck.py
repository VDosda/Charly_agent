import sqlite3
from typing import Optional

from agent.bootstrap.settings import VecExtensionType


def healthcheck_db(conn: sqlite3.Connection, vec_extension: Optional[VecExtensionType] = None):
    """
    Perform a database health check.

    This function verifies:
    - SQLite connection is alive
    - required tables exist
    - PRAGMA configuration is correct
    - vector extension is available if configured

    Raises RuntimeError if something critical is misconfigured.
    """

    _check_connection(conn)
    _check_pragmas(conn)
    _check_migrations_table(conn)

    if vec_extension and vec_extension != "none":
        _check_vector_extension(conn, vec_extension)


def _check_connection(conn: sqlite3.Connection):
    """
    Ensure SQLite connection works.
    """

    try:
        conn.execute("SELECT 1")
    except sqlite3.Error as e:
        raise RuntimeError("Database connection failed") from e


def _check_pragmas(conn: sqlite3.Connection):
    """
    Validate important SQLite PRAGMA settings.
    """

    cursor = conn.cursor()

    journal = cursor.execute("PRAGMA journal_mode").fetchone()[0]

    if journal.lower() != "wal":
        raise RuntimeError(
            f"SQLite journal_mode should be WAL for production, got '{journal}'"
        )

    foreign_keys = cursor.execute("PRAGMA foreign_keys").fetchone()[0]

    if foreign_keys != 1:
        raise RuntimeError("SQLite foreign_keys must be enabled")

    cursor.close()


def _check_migrations_table(conn: sqlite3.Connection):
    """
    Ensure migration tracking table exists.
    """

    result = conn.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type='table'
        AND name='schema_migrations'
        """
    ).fetchone()

    if not result:
        raise RuntimeError(
            "schema_migrations table not found. "
            "Database migrations were likely not applied."
        )


def _check_vector_extension(conn: sqlite3.Connection, extension: VecExtensionType):
    """
    Verify that the vector extension is usable.

    We run a minimal query that should exist
    if the extension is properly loaded.
    """

    cursor = conn.cursor()

    try:
        if extension == "sqlite_vec":
            # sqlite-vec exposes vec_version()
            cursor.execute("SELECT vec_version()")

        elif extension == "sqlite_vss":
            # sqlite-vss exposes vss_version()
            cursor.execute("SELECT vss_version()")

        else:
            raise RuntimeError(f"Unknown vector extension: {extension}")

    except sqlite3.Error as e:
        raise RuntimeError(
            f"Vector extension '{extension}' appears not to be loaded correctly."
        ) from e

    finally:
        cursor.close()