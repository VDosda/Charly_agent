-- ==========================================
-- LT archive state
-- ==========================================
-- We archive LT memories instead of deleting them so we can:
-- - preserve audit/history,
-- - avoid destructive loss,
-- - keep default retrieval focused on active memories only.

ALTER TABLE memory_items ADD COLUMN archived INTEGER NOT NULL DEFAULT 0;
ALTER TABLE memory_items ADD COLUMN archived_at INTEGER;
ALTER TABLE memory_items ADD COLUMN archived_reason TEXT;

CREATE INDEX IF NOT EXISTS idx_memory_items_user_archived
ON memory_items(user_id, archived, ts_updated DESC);

ALTER TABLE memory_item_versions ADD COLUMN archived INTEGER;
ALTER TABLE memory_item_versions ADD COLUMN archived_at INTEGER;
ALTER TABLE memory_item_versions ADD COLUMN archived_reason TEXT;
