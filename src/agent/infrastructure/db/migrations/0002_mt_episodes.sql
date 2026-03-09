-- ==========================================
-- Medium-Term Memory (MT): Episodes
-- ==========================================
-- Creates:
--   - episodes: one row per summarized chunk of ST turns
--   - episode_sources: mapping from episodes to chat_history turns
--
-- Notes:
-- - We store embeddings as BLOB (float32 packed) for future ANN indexing
--   (sqlite-vec/sqlite-vss index will be added in a later migration).
-- ==========================================

-- ------------------------------------------------
-- episodes
-- ------------------------------------------------
CREATE TABLE IF NOT EXISTS episodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    user_id TEXT NOT NULL,
    session_id TEXT NOT NULL,

    -- Range of ST turns summarized
    start_turn_id INTEGER NOT NULL,
    end_turn_id INTEGER NOT NULL,

    -- Unix timestamp when episode was created
    ts INTEGER NOT NULL,

    -- The summarized content
    summary TEXT NOT NULL,

    -- Optional structured metadata
    topics_json TEXT,          -- e.g. ["networking","security"]
    facts_json TEXT,           -- e.g. [{"k":"goal","v":"..."}]
    open_tasks_json TEXT,      -- e.g. [{"task":"...","status":"open"}]

    -- Scoring
    importance REAL NOT NULL DEFAULT 0.5,  -- [0..1]
    confidence REAL NOT NULL DEFAULT 0.6,  -- [0..1]

    -- Embedding storage (packed float32)
    embedding_model TEXT,
    embedding_dims INTEGER,
    embedding_blob BLOB
);

-- Uniqueness: one episode per exact turn range per session
CREATE UNIQUE INDEX IF NOT EXISTS idx_episodes_unique_range
ON episodes(session_id, start_turn_id, end_turn_id);

-- Fast retrieval by recency per user/session
CREATE INDEX IF NOT EXISTS idx_episodes_user_ts
ON episodes(user_id, ts DESC);

CREATE INDEX IF NOT EXISTS idx_episodes_session_endturn
ON episodes(session_id, end_turn_id DESC);

-- Importance sorting helper
CREATE INDEX IF NOT EXISTS idx_episodes_user_importance
ON episodes(user_id, importance DESC, ts DESC);


-- ------------------------------------------------
-- episode_sources
-- ------------------------------------------------
-- Tracks which ST turns were summarized into a given episode.
-- Useful for:
-- - audit/debug
-- - preventing double-summarization
-- - later compression flags (optional)
-- ------------------------------------------------
CREATE TABLE IF NOT EXISTS episode_sources (
    episode_id INTEGER NOT NULL,
    session_id TEXT NOT NULL,
    turn_id INTEGER NOT NULL,

    PRIMARY KEY (episode_id, turn_id),

    FOREIGN KEY (episode_id) REFERENCES episodes(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_episode_sources_session_turn
ON episode_sources(session_id, turn_id);

CREATE INDEX IF NOT EXISTS idx_episode_sources_episode
ON episode_sources(episode_id);