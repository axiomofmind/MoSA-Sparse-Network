import { escapeHtml, formatDate } from "./model.js";

export const TITLES = {
  home: "Home", work: "My work", workspace: "Task", settings: "Settings",
  overview: "Overview", fleet: "Models & Hardware", requests: "Activity", vision: "Visual evidence",
  escalation: "Verifier", evaluation: "Evaluation", configuration: "Settings",
};

export function routeFromHash(hash) {
  const [view, id] = String(hash || "").replace(/^#/, "").split("/");
  return { view: view in TITLES ? view : "home", id: id || null };
}

export function renderShellChrome({ documentRoot = document, state, status }) {
  documentRoot.querySelector("#view-title").textContent = TITLES[state.view];
  documentRoot.querySelectorAll(".nav-item").forEach((button) => button.classList.toggle("is-active", button.dataset.view === state.view));
  const chip = documentRoot.querySelector("#connection-chip");
  chip.className = `connection-chip ${status.tone === "good" ? "is-online" : status.tone === "warn" ? "is-stale" : "is-offline"}`;
  chip.innerHTML = `<span></span><strong>${escapeHtml(status.label)}</strong><small>${escapeHtml(status.detail)}</small>`;
  documentRoot.querySelector("#session-button").textContent = state.token ? "Session" : "Connect";
  documentRoot.querySelector("#snapshot-time").textContent = state.snapshot ? `Snapshot ${formatDate(state.snapshot.generated_at)}` : "No snapshot";
  documentRoot.querySelector("#event-position").textContent = `Event ${state.lastEventId || "—"}`;
  const rolePill = documentRoot.querySelector("#role-pill");
  const role = state.snapshot?.session?.role || "disconnected";
  rolePill.textContent = role.toUpperCase();
  rolePill.className = `readonly-pill ${state.snapshot?.session?.read_only ? "" : "is-operator"}`;
}
