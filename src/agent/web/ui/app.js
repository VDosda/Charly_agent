async function apiGet(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return await res.json();
}

function fmtMs(ts) {
  const d = new Date(ts);
  return d.toISOString();
}

function buildQuery() {
  const userId = document.getElementById("userId").value.trim();
  const sessionId = document.getElementById("sessionId").value.trim();
  const correlationId = document.getElementById("correlationId").value.trim();
  const limit = Number.parseInt(document.getElementById("limit").value, 10) || 200;

  const params = new URLSearchParams();
  params.set("limit", String(Math.min(Math.max(limit, 1), 1000)));
  if (userId) params.set("user_id", userId);
  if (sessionId) params.set("session_id", sessionId);
  if (correlationId) params.set("correlation_id", correlationId);

  return params;
}

function formatLogLine(ev) {
  const level = (ev.level || "info").toUpperCase();
  const payload = JSON.stringify(ev.payload || {});
  return `${fmtMs(ev.ts_ms)} | ${level} | ${ev.event} | corr=${ev.correlation_id || "-"} | user=${ev.user_id || "-"} | session=${ev.session_id || "-"} | ${payload}`;
}

function renderLogs(items) {
  const logs = document.getElementById("logs");
  logs.innerHTML = "";

  if (!items.length) {
    const empty = document.createElement("div");
    empty.className = "logLine muted";
    empty.textContent = "Aucun log trouve pour ces filtres.";
    logs.appendChild(empty);
    return;
  }

  for (const ev of items) {
    const line = document.createElement("div");
    line.className = `logLine level-${(ev.level || "info").toLowerCase()}`;
    line.textContent = formatLogLine(ev);
    logs.appendChild(line);
  }
}

async function loadLogs() {
  const meta = document.getElementById("logsMeta");

  try {
    const params = buildQuery();
    const data = await apiGet(`/api/logs?${params.toString()}`);
    renderLogs(data.items || []);
    meta.textContent = `${(data.items || []).length} logs affiches`;
  } catch (err) {
    console.error(err);
    if (String(err.message).includes("HTTP 404")) {
      meta.textContent = "Erreur: /api/logs introuvable. Redemarre le monitor avec le code actuel.";
    } else {
      meta.textContent = `Erreur: ${err.message}`;
    }
  }
}

let timer = null;

function setAutoRefresh(enabled) {
  if (timer) {
    clearInterval(timer);
    timer = null;
  }
  if (enabled) {
    timer = setInterval(loadLogs, 2000);
  }
}

document.getElementById("refreshBtn").addEventListener("click", loadLogs);
document.getElementById("autoRefresh").addEventListener("change", (e) => {
  setAutoRefresh(Boolean(e.target.checked));
});

loadLogs();
