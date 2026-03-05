-- ==========================================
-- Long-Term Memory (LT): memory_items
-- ==========================================
-- Stores stable, reusable memory facts/preferences/procedures.
-- Each item is atomic, scored, and can be embedded for retrieval.
-- ==========================================

CREATE TABLE IF NOT EXISTS memory_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    user_id TEXT NOT NULL,

    -- Type of memory item (preference, identity, constraint, goal, procedure, etc.)
    kind TEXT NOT NULL,

    -- Optional key for upsert semantics (e.g. "preferred_language", "goal_primary")
    mem_key TEXT,

    -- Human-readable memory content (canonical)
    value TEXT NOT NULL,

    -- Provenance
    source_session_id TEXT,
    source_episode_id INTEGER,
    source_note TEXT,

    -- Timestamps
    ts_created INTEGER NOT NULL,
    ts_updated INTEGER NOT NULL,
    last_seen_ts INTEGER NOT NULL,

    -- Scoring
    confidence REAL NOT NULL DEFAULT 0.6,  -- [0..1]
    importance REAL NOT NULL DEFAULT 0.5,  -- [0..1]

    -- Embedding storage (packed float32)
    embedding_model TEXT,
    embedding_dims INTEGER,
    embedding_blob BLOB,
    embedding_status TEXT NOT NULL DEFAULT 'pending',
    last_embedding_error TEXT,
    embedding_retry_count INTEGER NOT NULL DEFAULT 0,
    embedding_last_attempt_ts INTEGER,
    embedding_next_retry_ts INTEGER,

    -- Evidence / provenance to transcript
    evidence_span TEXT,
    source_turn_ids_json TEXT,

    FOREIGN KEY (source_episode_id) REFERENCES episodes(id) ON DELETE SET NULL
);

-- Unique upsert when mem_key is provided (per user/kind/key)
CREATE UNIQUE INDEX IF NOT EXISTS idx_memory_items_unique_key
ON memory_items(user_id, kind, mem_key)
WHERE mem_key IS NOT NULL;

-- Retrieval helpers
CREATE INDEX IF NOT EXISTS idx_memory_items_user_updated
ON memory_items(user_id, ts_updated DESC);

CREATE INDEX IF NOT EXISTS idx_memory_items_user_importance
ON memory_items(user_id, importance DESC, ts_updated DESC);

CREATE INDEX IF NOT EXISTS idx_memory_items_user_kind
ON memory_items(user_id, kind);

CREATE INDEX IF NOT EXISTS idx_memory_items_embedding_retry
ON memory_items(user_id, embedding_status, embedding_next_retry_ts);

-- ==========================================
-- LT memory versioning: snapshots before UPDATE
-- ==========================================

CREATE TABLE IF NOT EXISTS memory_item_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    memory_item_id INTEGER NOT NULL,
    user_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    mem_key TEXT,
    value TEXT NOT NULL,

    confidence REAL,
    importance REAL,

    source_session_id TEXT,
    source_episode_id INTEGER,
    source_note TEXT,
    evidence_span TEXT,
    source_turn_ids_json TEXT,

    embedding_model TEXT,
    embedding_dims INTEGER,
    embedding_status TEXT,
    last_embedding_error TEXT,
    embedding_retry_count INTEGER,
    embedding_last_attempt_ts INTEGER,
    embedding_next_retry_ts INTEGER,

    ts_created INTEGER,
    ts_updated INTEGER,
    last_seen_ts INTEGER,

    versioned_ts INTEGER NOT NULL,
    change_reason TEXT,

    FOREIGN KEY (memory_item_id) REFERENCES memory_items(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_memory_item_versions_item_ts
ON memory_item_versions(memory_item_id, versioned_ts DESC);

CREATE INDEX IF NOT EXISTS idx_memory_item_versions_user_ts
ON memory_item_versions(user_id, versioned_ts DESC);
