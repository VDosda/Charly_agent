async function apiGet(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return await res.json();
}

function fmtMs(ts) {
  const d = new Date(ts);
  return d.toISOString();
}

async function loadCorrelations() {
  const userId = document.getElementById("userId").value.trim();
  const sessionId = document.getElementById("sessionId").value.trim();

  const params = new URLSearchParams();
  params.set("limit", "80");
  if (userId) params.set("user_id", userId);
  if (sessionId) params.set("session_id", sessionId);

  const data = await apiGet(`/api/correlations?${params.toString()}`);
  const list = document.getElementById("corrList");
  list.innerHTML = "";

  for (const item of data.items) {
    const li = document.createElement("li");
    li.className = "corr";
    li.innerHTML = `
      <button class="corrBtn" data-id="${item.correlation_id}">
        ${item.correlation_id}
      </button>
      <div class="muted">
        ${fmtMs(item.ts_start)} → ${fmtMs(item.ts_end)} • events=${item.events}
      </div>
    `;
    list.appendChild(li);
  }

  document.querySelectorAll(".corrBtn").forEach(btn => {
    btn.addEventListener("click", () => loadTimeline(btn.dataset.id));
  });
}

async function loadTimeline(correlationId) {
  const data = await apiGet(`/api/correlation/${encodeURIComponent(correlationId)}`);
  const timeline = document.getElementById("timeline");
  const meta = document.getElementById("timelineMeta");

  meta.textContent = `correlation_id=${correlationId} • events=${data.items.length}`;

  const lines = data.items.map(ev => {
    const p = JSON.stringify(ev.payload);
    return `${fmtMs(ev.ts_ms)} [${ev.level}] ${ev.event}  ${p}`;
  });

  timeline.textContent = lines.join("\n");
}

document.getElementById("refreshBtn").addEventListener("click", loadCorrelations);
loadCorrelations().catch(err => console.error(err));