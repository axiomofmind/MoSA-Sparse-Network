export function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

export function formatBytes(value) {
  const number = Number(value || 0);
  if (!Number.isFinite(number) || number <= 0) return "0 B";
  const units = ["B", "KiB", "MiB", "GiB", "TiB"];
  const index = Math.min(Math.floor(Math.log(number) / Math.log(1024)), units.length - 1);
  const scaled = number / (1024 ** index);
  return `${scaled >= 100 ? scaled.toFixed(0) : scaled.toFixed(1)} ${units[index]}`;
}

export function formatDuration(value) {
  const milliseconds = Number(value || 0);
  if (milliseconds < 1000) return `${Math.round(milliseconds)} ms`;
  if (milliseconds < 60000) return `${(milliseconds / 1000).toFixed(1)} s`;
  return `${(milliseconds / 60000).toFixed(1)} min`;
}

export function formatDate(value) {
  if (!value) return "—";
  const date = new Date(value);
  return Number.isNaN(date.valueOf()) ? "—" : date.toLocaleString();
}

export function toneForState(state) {
  const normalized = String(state || "unknown").toLowerCase();
  if (["ready", "completed", "accepted", "answer", "healthy"].includes(normalized)) return "good";
  if (["busy", "running", "loading", "restoring", "queued", "draining"].includes(normalized)) return "warn";
  if (["failed", "cancelled", "quarantined", "unavailable"].includes(normalized)) return "bad";
  return "neutral";
}

export function profileForSelection(catalog, selection) {
  const profiles = catalog?.profiles || [];
  if (selection === "auto") {
    const recommended = catalog?.recommended_selection;
    return profiles.find(
      (profile) => profile.selection_key === recommended && profile.active,
    )
      || profiles.find(
        (profile) => profile.selection_key === recommended && profile.variant === "standard",
      )
      || profiles.find((profile) => profile.selection_key === recommended)
      || profiles.find((profile) => profile.active)
      || profiles[0]
      || null;
  }
  if (selection === "custom") return catalog?.custom || null;
  const exact = profiles.find((profile) => profile.id === selection);
  if (exact) return exact;
  return profiles.find((profile) => profile.selection_key === selection && profile.active)
    || profiles.find((profile) => profile.selection_key === selection)
    || null;
}

export function groupSwapEvents(events) {
  const groups = new Map();
  for (const event of events || []) {
    if (event.event !== "swap_phase" && event.event !== "swap_recovered") continue;
    const id = event.execution_id || "unassigned";
    if (!groups.has(id)) groups.set(id, { id, endpoint: event.endpoint, events: [] });
    groups.get(id).events.push(event);
  }
  return [...groups.values()]
    .map((group) => ({
      ...group,
      events: group.events.sort((a, b) => Number(a.event_id) - Number(b.event_id)),
    }))
    .sort((a, b) => Number(b.events.at(-1)?.event_id || 0) - Number(a.events.at(-1)?.event_id || 0));
}

export function graphColumns(graph) {
  const nodes = graph?.nodes || [];
  const levels = new Map();
  const byId = new Map(nodes.map((node) => [node.id, node]));
  const levelOf = (node, seen = new Set()) => {
    if (levels.has(node.id)) return levels.get(node.id);
    if (seen.has(node.id)) return 0;
    seen.add(node.id);
    const dependencies = node.depends_on || [];
    const level = dependencies.length
      ? 1 + Math.max(...dependencies.map((id) => byId.has(id) ? levelOf(byId.get(id), seen) : 0))
      : 0;
    levels.set(node.id, level);
    return level;
  };
  nodes.forEach((node) => levelOf(node));
  const columns = [];
  for (const node of nodes) {
    const level = levels.get(node.id) || 0;
    if (!columns[level]) columns[level] = [];
    columns[level].push(node);
  }
  return columns;
}

export function percent(used, total) {
  if (!Number(total)) return 0;
  return Math.max(0, Math.min(100, (Number(used) / Number(total)) * 100));
}
