import sqlite3
from functools import lru_cache
from pathlib import Path
from typing import Iterator, Optional

from agent.config.settings import load_settings
from agent.config.settings import VecExtensionType


def get_connection(
    db_path: str,
    vec_extension: VecExtensionType = "none",
) -> sqlite3.Connection:
    """
    Create and configure a SQLite connection for the agent.

    Responsibilities:
    - Ensure database file exists
    - Enable important PRAGMA settings
    - Enable extension loading
    - Load vector extension (sqlite-vec or sqlite-vss)
    """

    # Ensure database directory exists
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    # Create SQLite connection
    conn = sqlite3.connect(
        db_path,
        check_same_thread=False,  # required if used across threads
    )

    # Return rows as dictionaries instead of tuples
    conn.row_factory = sqlite3.Row

    # Enable extension loading
    conn.enable_load_extension(True)

    # Configure SQLite for better performance and reliability
    _configure_pragmas(conn)

    # Load vector extension if requested
    if vec_extension != "none":
        _load_vector_extension(conn, vec_extension)

    return conn


def _configure_pragmas(conn: sqlite3.Connection):
    """
    Configure SQLite PRAGMA settings.

    These settings improve durability and concurrency for
    long-running applications such as agents.
    """

    cursor = conn.cursor()

    # Write-Ahead Logging improves concurrent reads/writes
    cursor.execute("PRAGMA journal_mode=WAL;")

    # Enforce foreign key constraints
    cursor.execute("PRAGMA foreign_keys=ON;")

    # Better sync balance between durability and speed
    cursor.execute("PRAGMA synchronous=NORMAL;")

    # Allow SQLite to use temporary memory instead of disk
    cursor.execute("PRAGMA temp_store=MEMORY;")

    # Increase cache size (negative value = KB)
    cursor.execute("PRAGMA cache_size=-20000;")  # ~20MB

    cursor.close()


def _load_vector_extension(
    conn: sqlite3.Connection,
    extension_type: VecExtensionType,
):
    """
    Load vector search extension.

    Supported extensions:
    - sqlite_vec
    - sqlite_vss

    The extension shared library must be available
    in the system library path or provided via absolute path.
    """

    if extension_type == "sqlite_vec":
        import sqlite_vec
        sqlite_vec.load(conn)

    elif extension_type == "sqlite_vss":
        conn.execute("SELECT load_extension('vss0')")

    else:
        raise ValueError(f"Unsupported vector extension: {extension_type}")


@lru_cache(maxsize=1)
def _cached_connection() -> sqlite3.Connection:
    settings = load_settings()
    return get_connection(
        db_path=settings.db.path,
        vec_extension=settings.db.vec_extension,
    )


def get_db() -> Iterator[sqlite3.Connection]:
    """
    FastAPI dependency that yields a shared SQLite connection.
    """
    yield _cached_connection()
