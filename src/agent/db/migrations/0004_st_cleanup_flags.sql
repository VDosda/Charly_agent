-- ==========================================
-- Migration 0004 : Short-Term Memory cleanup
-- ==========================================
--
-- Goal
-- ----
-- Prevent the `chat_history` table (Short-Term memory) from growing
-- indefinitely while preserving recent conversational context.
--
-- Instead of immediately deleting old turns, we introduce a soft
-- archival mechanism using a boolean flag (`archived`).
--
-- This allows the agent to:
--   • keep recent turns available for context injection
--   • safely summarize older turns into MT episodes
--   • optionally delete archived turns later
--
-- This approach avoids losing raw data prematurely and provides
-- better traceability for debugging and monitoring.
--
-- Memory architecture reminder
-- ----------------------------
-- ST (Short-Term)  : chat_history (raw conversation turns)
-- MT (Medium-Term) : episodes (summarized conversation chunks)
-- LT (Long-Term)   : memory_items (stable facts / preferences)
--
-- When ST turns become old:
--   1. they are summarized into an MT episode
--   2. they are marked `archived = 1`
--   3. optionally deleted later if configured
--
-- ==========================================


-- ------------------------------------------------
-- Add archival flag to chat_history
-- ------------------------------------------------
-- archived = 0 → active ST memory (used in prompt)
-- archived = 1 → old turns that have been summarized
--
-- SQLite allows adding columns without rewriting the table.
-- Existing rows will default to archived = 0.
-- ------------------------------------------------

ALTER TABLE chat_history
ADD COLUMN archived INTEGER NOT NULL DEFAULT 0;


-- ------------------------------------------------
-- Index to speed up ST cleanup and retrieval
-- ------------------------------------------------
-- We frequently query:
--   WHERE session_id = ?
--     AND archived = 0
--     ORDER BY turn_id
--
-- This composite index ensures cleanup operations and
-- retrieval of active turns remain efficient even when
-- the table grows large.
-- ------------------------------------------------

CREATE INDEX IF NOT EXISTS idx_chat_history_archived
ON chat_history(session_id, archived, turn_id);