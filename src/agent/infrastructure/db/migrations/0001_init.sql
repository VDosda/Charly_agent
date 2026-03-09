-- ==========================================
-- Agent database initialization
-- ==========================================
-- Creates the base tables required by the agent runtime.
-- This migration initializes:
--   - Short-term memory (chat_history)
-- ==========================================


-- ------------------------------------------------
-- chat_history
-- ------------------------------------------------
-- Stores all conversation turns for a session.
-- This table represents Short-Term Memory (ST).
-- ------------------------------------------------

CREATE TABLE IF NOT EXISTS chat_history (

    id INTEGER PRIMARY KEY AUTOINCREMENT,

    -- user identifier
    user_id TEXT NOT NULL,

    -- session identifier
    session_id TEXT NOT NULL,

    -- incremental turn number within a session
    turn_id INTEGER NOT NULL,

    -- unix timestamp
    ts INTEGER NOT NULL,

    -- role of the message
    -- user | assistant | tool
    role TEXT NOT NULL,

    -- main text content
    content TEXT NOT NULL,

    -- tool metadata (nullable)
    tool_name TEXT,
    tool_args_json TEXT,
    tool_result_json TEXT,

    -- ST archival flag
    -- 0 = active turns injected in prompt
    -- 1 = archived turns (cleanup/summarized)
    archived INTEGER NOT NULL DEFAULT 0
);


-- ------------------------------------------------
-- Indexes
-- ------------------------------------------------

-- Fast retrieval of session history
CREATE INDEX IF NOT EXISTS idx_chat_session_turn
ON chat_history(session_id, turn_id);

-- Fast filtering by user
CREATE INDEX IF NOT EXISTS idx_chat_user
ON chat_history(user_id);

-- Fast time based queries
CREATE INDEX IF NOT EXISTS idx_chat_timestamp
ON chat_history(ts);


-- ------------------------------------------------
-- Constraints
-- ------------------------------------------------

-- Ensure unique turn order within a session
CREATE UNIQUE INDEX IF NOT EXISTS idx_chat_session_turn_unique
ON chat_history(session_id, turn_id);

-- Fast retrieval of active (non-archived) ST turns
CREATE INDEX IF NOT EXISTS idx_chat_history_archived
ON chat_history(session_id, archived, turn_id);
