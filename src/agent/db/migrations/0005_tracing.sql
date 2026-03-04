-- ==========================================
-- Tracing storage: trace_events
-- ==========================================
-- Stores structured trace events for monitoring UI.
-- Compatible with JSONTracer events.
-- ==========================================

CREATE TABLE IF NOT EXISTS trace_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts_ms INTEGER NOT NULL,

  level TEXT NOT NULL,
  event TEXT NOT NULL,

  correlation_id TEXT NOT NULL,
  user_id TEXT,
  session_id TEXT,

  payload_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_trace_events_corr_ts
ON trace_events(correlation_id, ts_ms);

CREATE INDEX IF NOT EXISTS idx_trace_events_user_ts
ON trace_events(user_id, ts_ms DESC);

CREATE INDEX IF NOT EXISTS idx_trace_events_session_ts
ON trace_events(session_id, ts_ms DESC);

CREATE INDEX IF NOT EXISTS idx_trace_events_event_ts
ON trace_events(event, ts_ms DESC);