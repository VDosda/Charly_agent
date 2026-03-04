-- ==========================================
-- Migration 0005 : Vector search tables
-- ==========================================

-- Vector table for MT episodes
CREATE VIRTUAL TABLE IF NOT EXISTS episodes_vec USING vec0(
    episode_id INTEGER PRIMARY KEY,
    embedding FLOAT[768]
);

-- Vector table for LT memory items
CREATE VIRTUAL TABLE IF NOT EXISTS memory_items_vec USING vec0(
    item_id INTEGER PRIMARY KEY,
    embedding FLOAT[768]
);