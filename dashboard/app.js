import { ControllerApi, ApiError } from "/dashboard/api-client.js";
import {
  escapeHtml,
  formatBytes,
  formatDate,
  formatDuration,
  graphColumns,
  groupSwapEvents,
  percent,
  profileForSelection,
  toneForState,
} from "/dashboard/model.js";
import {
  buildTaskPayload,
  buildFollowUpPayload,
  classifyUpload,
  createTaskDraft,
  suggestTask,
  taskCopy,
  taskPlan,
} from "/dashboard/task-workspace.js";
import {
  clearDrafts,
  createDraftId,
  deleteDraft,
  listDrafts,
  saveDraft,
} from "/dashboard/draft-store.js";
import { taskState } from "/dashboard/task-state.js";
import { renderShellChrome, routeFromHash } from "/dashboard/shell.js";
import { reconcileStatusElements, replaceRegionPreservingContext, shouldRefreshAfterEvent } from "/dashboard/event-reconciliation.js";
import { hasTaskIssues, issuesFromError, mergeTaskIssues, validateTaskDraft } from "/dashboard/task-errors.js";
import { renderHomeView, STARTER_EXAMPLES } from "/dashboard/views/home.js";
import { renderMyWorkView } from "/dashboard/views/my-work.js";
import { renderSettingsView } from "/dashboard/views/settings.js";
import { renderWorkspaceView } from "/dashboard/views/workspace.js";
import { renderDeliverableView, renderWorkspaceThread } from "/dashboard/views/results.js";

const initialRoute = routeFromHash(location.hash);
const storedDrafts = listDrafts(localStorage);
const storedById = Object.fromEntries(storedDrafts.map((draft) => [draft.id, draft]));

const state = {
  view: initialRoute.view,
  token: sessionStorage.getItem("sparse-dashboard-token") || "",
  snapshot: null,
  selection: "auto",
  lastEventId: 0,
  connected: false,
  stale: false,
  lastSuccess: 0,
  polling: false,
  actionPending: false,
  useCaseTemplate: null,
  taskIntent: "",
  taskDrafts: storedById,
  activeDraftId: initialRoute.view === "workspace" ? initialRoute.id : null,
  activeRunId: null,
  homeFiles: [],
  runDetails: {},
  taskSuggestion: null,
  readiness: null,
  taskIssues: {},
};

const api = new ControllerApi(state.token);
const main = document.querySelector("#main-content");
const authDialog = document.querySelector("#auth-dialog");
const detailDialog = document.querySelector("#detail-dialog");
const confirmDialog = document.querySelector("#confirm-dialog");
let confirmationResolver = null;

function can(permission) {
  return state.snapshot?.session?.permissions?.includes(permission) || false;
}

function actionButton(label, attributes, tone = "secondary") {
  return `<button class="button ${tone} small" ${attributes} ${state.actionPending ? "disabled" : ""}>${escapeHtml(label)}</button>`;
}

function badge(value, tone = toneForState(value)) {
  return `<span class="badge ${tone}"><i></i>${escapeHtml(value || "unknown")}</span>`;
}

function metric(label, value, note = "") {
  return `<article class="metric-card"><p>${escapeHtml(label)}</p><strong>${escapeHtml(value)}</strong><small>${escapeHtml(note)}</small></article>`;
}

function emptyState(title, text) {
  return `<div class="empty-state"><span>◇</span><h3>${escapeHtml(title)}</h3><p>${escapeHtml(text)}</p></div>`;
}

function sectionHeading(kicker, title, aside = "") {
  return `<div class="section-heading"><div><p class="eyebrow">${escapeHtml(kicker)}</p><h2>${escapeHtml(title)}</h2></div>${aside}</div>`;
}

function modelFor(id) {
  return state.snapshot?.models?.find((model) => model.id === id) || { id, display_name: id, modalities: [], capabilities: [] };
}

function activeProfile() {
  return profileForSelection(state.snapshot?.profiles, state.selection);
}

function systemStatus() {
  if (!state.connected || state.stale) {
    return { label: "Connection lost", detail: "Sparse cannot currently confirm local task state.", tone: "bad" };
  }
  if (state.snapshot?.operations?.profile_switch_in_progress) {
    return { label: "Preparing models", detail: "Sparse is applying an authorized local model configuration.", tone: "warn" };
  }
  const entries = Object.values(state.snapshot?.fleet?.endpoints || {});
  if (entries.some((entry) => ["loading", "draining"].includes(entry.state))) {
    return { label: "Preparing models", detail: "Installed models are being prepared for local work.", tone: "warn" };
  }
  if (!entries.length || entries.every((entry) => !["ready", "busy"].includes(entry.state))) {
    return { label: "Needs setup", detail: "No task model is ready. Open Settings for permitted recovery actions.", tone: "warn" };
  }
  return { label: "Ready", detail: "Sparse is connected and ready for supported local tasks.", tone: "good" };
}

function renderOverview() {
  const fleet = state.snapshot.fleet;
  const endpoints = Object.entries(fleet.endpoints || {});
  const ready = endpoints.filter(([, value]) => value.state === "ready").length;
  const active = state.snapshot.profiles.profiles.find((profile) => profile.active);
  const used = fleet.current_vram_bytes;
  const total = fleet.total_vram_bytes;
  const recent = [...state.snapshot.events].reverse().slice(0, 8);
  return `
    <section class="hero-grid">
      <div class="hero-copy">
        <p class="eyebrow">SYSTEM POSTURE</p>
        <h2>${ready === endpoints.length ? "Fleet is nominal." : "Fleet needs attention."}</h2>
        <p>${ready} of ${endpoints.length} resident endpoints ready. The dashboard is observing <strong>${escapeHtml(active?.id || fleet.roster)}</strong> without control privileges.</p>
      </div>
      <div class="memory-dial"><div><strong>${total ? percent(used, total).toFixed(0) : "—"}%</strong><span>accelerator</span></div></div>
    </section>
    <section class="metric-grid">
      ${metric("Resident health", `${ready}/${endpoints.length}`, "ready endpoints")}
      ${metric("Accelerator memory", formatBytes(used), total ? `${formatBytes(total)} capacity` : "telemetry unavailable")}
      ${metric("Queue", String(fleet.queue_depth), `${fleet.active_requests} active`)}
      ${metric("Event position", String(state.lastEventId || "—"), "durable replay cursor")}
    </section>
    <div class="two-column">
      <section class="panel">
        ${sectionHeading("FLEET MAP", "Resident endpoints", `<a href="#fleet" class="text-link">Inspect fleet →</a>`)}
        <div class="endpoint-strip">${endpoints.map(([id, value]) => `<div><span class="state-dot ${toneForState(value.state)}"></span><strong>${escapeHtml(modelFor(id).display_name)}</strong><small>${escapeHtml(value.state)}</small></div>`).join("")}</div>
      </section>
      <section class="panel">
        ${sectionHeading("EVENT JOURNAL", "Latest controller activity", `<span class="mono">LIVE / ${state.lastEventId}</span>`)}
        <ol class="event-list">${recent.map((event) => `<li><span>${escapeHtml(event.event)}</span><strong>${escapeHtml(event.endpoint)}</strong><time>${formatDate(event.timestamp)}</time></li>`).join("") || `<li>No retained events</li>`}</ol>
      </section>
    </div>`;
}

function profilePreview() {
  const profile = activeProfile();
  if (!profile) return emptyState("No matching profile", "Choose another preview tier.");
  const apply = can("profile:apply") && !profile.active
    ? actionButton("Plan profile switch", `data-profile-plan="${escapeHtml(profile.id)}"`, "primary")
    : `<span class="readonly-pill">${profile.active ? "ACTIVE" : "PREVIEW ONLY"}</span>`;
  return `<div class="profile-preview">
    <div><p class="eyebrow">PREVIEW / ${escapeHtml(profile.hardware_profile)}</p><h3>${escapeHtml(profile.display_name || profile.id)}</h3><p class="mono">${escapeHtml(profile.id)}</p><p>${badge(profile.validation_state, profile.validation_state.includes("provisional") ? "warn" : "good")} ${profile.active ? badge("active", "good") : badge("not applied", "neutral")}</p>${apply}</div>
    <dl><div><dt>Residents</dt><dd>${profile.resident.length}</dd></div><div><dt>Proposed VRAM</dt><dd>${formatBytes(profile.proposed_vram_bytes)}</dd></div><div><dt>Proposed RAM</dt><dd>${formatBytes(profile.proposed_ram_bytes)}</dd></div><div><dt>KV cache</dt><dd>${formatBytes(profile.proposed_kv_cache_bytes)}</dd></div><div><dt>Workspace</dt><dd>${formatBytes(profile.proposed_workspace_bytes)}</dd></div><div><dt>Reserve</dt><dd>${escapeHtml(profile.budgets.minimum_vram_reserve_gb ?? "—")} GiB</dd></div><div><dt>Parallel limit</dt><dd>${profile.scheduler.maximum_parallel_generations ?? "—"}</dd></div></dl>
    <div class="roster-pills">${profile.resident.map((id) => `<span>${escapeHtml(id)}</span>`).join("")}</div>
    <p class="profile-note">Large tier: ${escapeHtml(profile.exclusive_swap.join(", ") || "none")} · System RAM: ${escapeHtml(profile.requirements.recommended_system_ram_gb ?? "—")} GiB recommended</p>
    ${profile.missing_or_invalid.length ? `<div class="notice warn">Missing or invalid: ${escapeHtml(profile.missing_or_invalid.join(", "))}</div>` : `<div class="notice good">All declared resident and swap artifacts validate.</div>`}
  </div>`;
}

function renderFleet() {
  const fleet = state.snapshot.fleet;
  const rows = Object.entries(fleet.endpoints || {}).map(([id, value]) => {
    const model = modelFor(id);
    const actions = can("fleet:operate") ? `<div class="row-actions">${actionButton("Smoke", `data-fleet-op="smoke" data-endpoint="${escapeHtml(id)}"`)}${value.state === "ready" ? actionButton("Drain", `data-fleet-op="drain" data-endpoint="${escapeHtml(id)}"`) + actionButton("Unload", `data-fleet-op="unload" data-endpoint="${escapeHtml(id)}"`) + actionButton("Quarantine", `data-fleet-op="quarantine" data-endpoint="${escapeHtml(id)}"`, "danger") : ""}</div>` : "—";
    return `<tr><td><a href="#model:${escapeHtml(id)}" data-detail-model="${escapeHtml(id)}"><strong>${escapeHtml(model.display_name)}</strong><small>${escapeHtml(id)}</small></a></td><td>${badge(value.state)}<small>${escapeHtml(model.admission?.state || "unreviewed")}</small></td><td>${escapeHtml(model.runtime?.adapter || "—")}<small>${escapeHtml(model.runtime?.artifact_class || "")}</small></td><td>${escapeHtml(model.source?.revision || "builtin")}</td><td>${formatBytes(value.loaded_process_ram_bytes)}<small>peak req ${formatBytes(value.peak_request_ram_bytes)}</small></td><td>${formatBytes(value.incremental_vram_bytes)}<small>peak req ${formatBytes(value.peak_request_vram_bytes)}</small></td><td>${formatBytes(value.kv_cache_bytes)}</td><td>${model.context_size ?? "—"}</td><td>${formatDuration(value.load_ms)}<small>last req ${formatDuration(value.last_request_elapsed_ms)}</small></td><td>${value.queue_depth}</td><td>${actions}</td></tr>`;
  }).join("");
  const fleetActions = can("fleet:operate") ? `<div class="toolbar">${actionButton("Start fleet", `data-fleet-op="start"`, "primary")}${actionButton("Reload fleet", `data-fleet-op="reload"`)}${actionButton("Unload all", `data-fleet-op="unload_all"`, "danger")}</div>` : `<span class="readonly-pill">VIEWER</span>`;
  return `${sectionHeading("RESIDENT PLANE", "Fleet state", `<span>${badge(state.snapshot.session.role, "neutral")}</span>`)}
    <section class="panel profile-panel">${sectionHeading("HARDWARE TIERS", "Profile preview", fleetActions)}${profilePreview()}</section>
    <section class="panel table-panel"><div class="table-scroll"><table><thead><tr><th>Endpoint</th><th>State</th><th>Runtime</th><th>Revision</th><th>RAM</th><th>VRAM Δ</th><th>KV</th><th>Context</th><th>Load / request</th><th>Queue</th><th>Controls</th></tr></thead><tbody>${rows}</tbody></table></div></section>
    <section class="metric-grid compact">${metric("Current", formatBytes(fleet.current_vram_bytes), "accelerator memory")}${metric("Peak", formatBytes(fleet.peak_vram_bytes), "since controller start")}${metric("Reserve", formatBytes(fleet.minimum_vram_reserve_bytes), "untouched target")}${metric("Headroom", formatBytes(fleet.headroom_bytes), "reported capacity")}</section>`;
}

function renderGraph(graph) {
  const columns = graphColumns(graph);
  if (!columns.length) return "";
  return `<div class="dag">${columns.map((column) => `<div class="dag-column">${column.map((node) => `<div class="dag-node ${toneForState(node.state)}"><small>${escapeHtml(node.kind)}</small><strong>${escapeHtml(node.id)}</strong><span>${escapeHtml(node.endpoint || node.state)}</span></div>`).join("")}</div>`).join("")}</div>`;
}

function renderRequests() {
  const requests = [...state.snapshot.requests].reverse();
  const traces = state.snapshot.execution_traces || [];
  const requestRows = requests.map((request) => {
    const cancellable = can("request:cancel") && ["queued", "running"].includes(request.state);
    return `<tr><td><button class="link-button" data-detail-request="${escapeHtml(request.id)}">${escapeHtml(request.id)}</button></td><td>${escapeHtml(request.endpoint)}</td><td>${badge(request.state)}</td><td>${formatDate(request.created_at)}</td><td>${escapeHtml(request.result?.envelope?.model_revision || "—")}</td><td>${formatDuration(request.result?.envelope?.resource_usage?.elapsed_ms)}</td><td>${cancellable ? actionButton("Cancel", `data-cancel-request="${escapeHtml(request.id)}"`, "danger") : "—"}</td></tr>`;
  }).join("");
  return `${sectionHeading("REQUEST PLANE", "Requests and bounded graphs", `<span class="readonly-pill">PROMPTS REDACTED</span>`)}
    <section class="panel table-panel">${requestRows ? `<div class="table-scroll"><table><thead><tr><th>Request</th><th>Endpoint</th><th>Status</th><th>Created</th><th>Revision</th><th>Latency</th><th>Controls</th></tr></thead><tbody>${requestRows}</tbody></table></div>` : emptyState("No service requests", "Submit requests through the CLI or controller API; they will appear here.")}</section>
    <section class="trace-grid">${traces.map((trace) => `<article class="panel trace-card"><div class="trace-meta"><div><p class="eyebrow">${escapeHtml(trace.graph?.mode || trace.mode || "EXECUTION")}</p><h3>${escapeHtml(trace.id)}</h3></div>${badge(trace.status || "recorded")}</div><p>${escapeHtml(trace.decision?.reason || trace.error || "Controller-owned execution graph")}</p>${renderGraph(trace.graph)}</article>`).join("") || emptyState("No execution graphs", "Top-1, top-2, and MoSA traces will render as finite DAGs.")}</section>`;
}

function renderVision() {
  const visual = state.snapshot.artifacts.filter((artifact) => String(artifact.media_type).startsWith("image/"));
  const events = state.snapshot.events.filter((event) => event.event.includes("vision") || event.event.includes("ocr"));
  return `${sectionHeading("PERCEPTION PLANE", "Vision evidence", `<span class="readonly-pill">SOURCE ≠ DERIVATION</span>`)}
    <div class="vision-grid">${visual.map((artifact) => `<article class="vision-card"><div class="vision-preview" data-artifact-preview="${escapeHtml(artifact.id)}" data-artifact-kind="${escapeHtml(artifact.kind)}"><div class="spinner" aria-hidden="true"></div></div><div><p class="eyebrow">${escapeHtml(artifact.kind)}</p><h3>${escapeHtml(artifact.id)}</h3><p>${formatBytes(artifact.size_bytes)} · ${formatDate(artifact.created_at)}</p><button class="button secondary small" data-detail-artifact="${escapeHtml(artifact.id)}">Open artifact</button></div></article>`).join("") || emptyState("No visual artifacts", "Images produced or attached to tasks will appear here.")}</div>
    <section class="panel">${sectionHeading("VISION EVENTS", "Model and projector activity")}<ol class="event-list">${events.slice(-20).reverse().map((event) => `<li><span>${escapeHtml(event.event)}</span><strong>${escapeHtml(event.endpoint)}</strong><time>${formatDate(event.timestamp)}</time></li>`).join("") || `<li>No vision events retained.</li>`}</ol></section>`;
}

function renderEscalation() {
  const swaps = groupSwapEvents(state.snapshot.events);
  const activeEscalation = state.snapshot.fleet.active_escalation_endpoint;
  return `${sectionHeading("LARGE MODEL TIER", "Escalation timeline", `<span class="readonly-pill">CONTROLLER OWNED</span>`)}
    <section class="metric-grid compact">${metric("Recorded swaps", String(swaps.length), "retained event window")}${metric("Active queue", String(state.snapshot.fleet.queue_depth), "all fleet requests")}${metric("Active verifier", activeEscalation ? modelFor(activeEscalation).display_name : "none", activeEscalation || "roster has no swap tier")}${metric("Verifier artifact", activeEscalation ? (modelFor(activeEscalation).artifact_state || "unknown") : "not configured", "roster-bound selection")}</section>
    <div class="swap-list">${swaps.map((swap) => `<article class="panel swap-card"><div class="trace-meta"><div><p class="eyebrow">${escapeHtml(swap.endpoint)}</p><h3>${escapeHtml(swap.id)}</h3></div>${badge(swap.events.at(-1)?.details?.phase || "recorded")}</div><div class="timeline">${swap.events.map((event) => `<div class="timeline-step ${toneForState(event.details?.phase)}"><i></i><div><strong>${escapeHtml(event.details?.phase || event.event)}</strong><span>${formatDuration(event.details?.elapsed_ms)} · event ${event.event_id}</span></div></div>`).join("")}</div>${swap.events.some((event) => event.details?.error) ? `<div class="notice bad">${escapeHtml(swap.events.find((event) => event.details?.error)?.details?.error)}</div>` : ""}</article>`).join("") || emptyState("No swap history", "Qwen drain, reclaim, load, solve, and restore phases will appear here.")}</div>`;
}

function renderEvaluation() {
  const anti = state.snapshot.antidoom;
  const policy = anti.runtime_detector || {};
  const runs = state.snapshot.runs || [];
  const persistedEvaluations = runs.filter((run) => run.summary?.baselines).map((run) => ({ id: run.id, suite: run.summary.suite, baselines: run.summary.baselines }));
  const evaluations = [...(state.snapshot.evaluations || []), ...persistedEvaluations.filter((run) => !(state.snapshot.evaluations || []).some((evaluation) => `${evaluation.id}.json` === run.id))];
  const comparisonCards = evaluations.flatMap((evaluation) => (evaluation.baselines || []).map((baseline) => {
    const manifestId = String(evaluation.id).endsWith(".json") ? evaluation.id : `${evaluation.id}.json`;
    return `<article class="panel comparison-card"><p class="eyebrow">${escapeHtml(evaluation.suite || "EVALUATION")}</p><h3>${escapeHtml(baseline.id)}</h3><div class="metric-bars">${Object.entries(baseline.metrics || {}).map(([name, value]) => `<label><span>${escapeHtml(name)} <strong>${escapeHtml(value)}</strong></span><progress max="100" value="${Math.max(0, Math.min(100, Number(value)))}"></progress></label>`).join("") || `<p>No numeric metrics recorded.</p>`}</div><button class="link-button" data-detail-run="${escapeHtml(manifestId)}">Open immutable manifest</button></article>`;
  }));
  const evaluationControl = can("evaluation:run") ? `<section class="panel"><form id="evaluation-form" class="control-form"><p class="eyebrow">CONTROLLED EVALUATION</p><h3>Record a versioned comparison</h3><label>Suite ID<input id="evaluation-suite" required value="dashboard-comparison"></label><label>Baselines JSON<textarea id="evaluation-baselines" rows="8" spellcheck="false">${escapeHtml(JSON.stringify([{ id: "top-1", metrics: { acceptance_rate: 82, escalation_rate: 18 } }, { id: "top-2", metrics: { acceptance_rate: 90, escalation_rate: 10 } }], null, 2))}</textarea></label><button class="button primary" type="submit">Create immutable evaluation</button></form></section>` : "";
  return `${sectionHeading("QUALITY PLANE", "Evaluation and AntiDoom", `<span>${badge("exact detector", "good")}</span>`)}
    <section class="metric-grid compact">${metric("Min repeats", String(policy.min_repeats ?? "—"), "exact sequence threshold")}${metric("Max period", String(policy.max_period ?? "—"), "characters")}${metric("Retry ceiling", String(policy.maximum_retries ?? "—"), "different-family only")}${metric("Training runs", String(anti.training_runs?.length || 0), "offline maintenance")}</section>
    ${evaluationControl}
    <section class="comparison-grid">${comparisonCards.join("") || emptyState("No baseline comparisons", "Evaluator and administrator sessions can create immutable comparison manifests.")}</section>
    <section class="panel table-panel">${sectionHeading("IMMUTABLE RUNS", "Recent evaluation manifests")} ${runs.length ? `<div class="table-scroll"><table><thead><tr><th>Manifest</th><th>Schema</th><th>Outcome</th><th>Updated</th><th></th></tr></thead><tbody>${runs.map((run) => `<tr><td><strong>${escapeHtml(run.id)}</strong></td><td>${escapeHtml(run.schema || "—")}</td><td>${badge(run.passed === true ? "passed" : run.status || "recorded", run.passed === true ? "good" : "neutral")}</td><td>${formatDate(run.updated_at)}</td><td><button class="link-button" data-detail-run="${escapeHtml(run.id)}">View manifest</button></td></tr>`).join("")}</tbody></table></div>` : emptyState("No evaluation manifests", "Frozen admission and comparison runs will appear here.")}</section>
    <section class="panel"><div class="notice neutral">${escapeHtml(anti.note)}</div></section>`;
}

function renderConfiguration() {
  const config = state.snapshot.configuration;
  const models = state.snapshot.models;
  const editor = can("configuration:write") ? `<section class="panel"><form id="config-form" class="control-form compact-form"><p class="eyebrow">ALLOWLISTED EDIT</p><h3>Controller runtime limits</h3><div class="form-grid"><label>Queue capacity<input id="config-queue" type="number" min="1" value="${escapeHtml(config.fleet?.queue_capacity ?? 32)}" required></label><label>Request timeout (seconds)<input id="config-timeout" type="number" min="0.1" step="0.1" value="${escapeHtml(config.controller?.request_timeout_seconds ?? 300)}" required></label><label>Telemetry interval (seconds)<input id="config-telemetry" type="number" min="0.01" step="0.01" value="${escapeHtml(config.controller?.telemetry_interval_seconds ?? 0.2)}" required></label></div><button class="button primary" type="submit">Validate change</button></form></section>` : "";
  return `${sectionHeading("POLICY PLANE", "Configuration and audit", `<span class="readonly-pill">REDACTED / ${escapeHtml(state.snapshot.session.role)}</span>`)}
    ${editor}
    <div class="two-column config-layout"><section class="panel"><p class="eyebrow">CONTROLLER CONFIG</p><pre class="json-view">${escapeHtml(JSON.stringify(config, null, 2))}</pre></section><section class="panel"><p class="eyebrow">MODEL REGISTRY</p><div class="registry-list">${models.map((model) => `<button data-detail-model="${escapeHtml(model.id)}"><span><strong>${escapeHtml(model.display_name)}</strong><small>${escapeHtml(model.role)}</small></span>${badge(model.artifact_state, model.artifact_state === "validated" || model.artifact_state === "not_required" ? "good" : "bad")}</button>`).join("")}</div></section></div>
    <section class="panel">${sectionHeading("AUDIT JOURNAL", "Immutable controller events")}<div class="audit-log">${[...state.snapshot.events].reverse().slice(0, 50).map((event) => `<div><code>${event.event_id}</code><time>${formatDate(event.timestamp)}</time><strong>${escapeHtml(event.event)}</strong><span>${escapeHtml(event.endpoint)}</span></div>`).join("")}</div></section>`;
}

function draftFor(template, { createNew = false } = {}) {
  const active = state.activeDraftId ? state.taskDrafts[state.activeDraftId] : null;
  if (!createNew && active?.templateId === template.id) return active;
  const existing = !createNew
    ? allDrafts().find((draft) => draft.templateId === template.id)
    : null;
  if (existing) return existing;
  const draft = createTaskDraft(template);
  draft.id = createDraftId();
  const saved = saveDraft(localStorage, draft);
  state.taskDrafts[saved.id] = saved;
  return saved;
}

function allDrafts() {
  return Object.values(state.taskDrafts)
    .filter((draft) => draft?.id)
    .sort((left, right) => String(right.updatedAt || "").localeCompare(String(left.updatedAt || "")));
}

function persistDraft(draft) {
  const saved = saveDraft(localStorage, draft);
  state.taskDrafts[saved.id] = { ...draft, ...saved, files: draft.files || [] };
  return state.taskDrafts[saved.id];
}

function draftById(id) {
  return allDrafts().find((draft) => draft.id === id) || null;
}

function openWorkspace(template, { prompt = "", files = [] } = {}) {
  const draft = draftFor(template, { createNew: true });
  if (prompt && !draft.prompt) draft.prompt = prompt;
  if (files.length) draft.files = [...(draft.files || []).filter((file) => !file.needsReselection), ...files];
  if (!draft.title) {
    const sourceName = files[0]?.name?.replace(/\.[^.]+$/, "");
    const promptTitle = prompt.split(/\s+/).slice(0, 7).join(" ");
    draft.title = sourceName
      ? `${taskCopy(template).action}: ${sourceName}`
      : promptTitle || taskCopy(template).action;
  }
  state.useCaseTemplate = template.id;
  state.activeDraftId = draft.id;
  state.readiness = null;
  persistDraft(draft);
  state.view = "workspace";
  location.hash = `workspace/${encodeURIComponent(draft.id)}`;
  render();
  return draft;
}

function humanize(value) {
  return String(value || "").replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function taskSourceFields(template, draft) {
  const copy = taskCopy(template);
  const issues = state.taskIssues[draft.id] || { fields: {}, sources: {} };
  const optionsFor = (selected) => (template.input_kinds || []).map((kind) => `<option value="${escapeHtml(kind)}" ${selected === kind ? "selected" : ""}>${escapeHtml(humanize(kind))}</option>`).join("");
  const files = (draft.files || []).map((file, index) => {
    const result = classifyUpload(template, file);
    const selected = draft.fileKinds?.[index] || result.kind || "";
    const sourceError = issues.sources?.[index];
    const status = sourceError || (file.needsReselection ? "Choose this file again" : result.reason);
    const needsChoice = ["unsupported", "ambiguous"].includes(result.status);
    const invalid = Boolean(sourceError || needsChoice || file.needsReselection);
    return `<li class="source-item ${invalid ? "has-error" : ""}" ${sourceError ? `aria-describedby="source-error-${index}"` : ""}><span class="source-icon" aria-hidden="true">${result.kind === "pdf" ? "PDF" : "FILE"}</span><span><strong>${escapeHtml(file.name)}</strong><small class="${sourceError ? "field-error" : ""}" ${sourceError ? `id="source-error-${index}"` : ""}>${escapeHtml(status)} · ${formatBytes(file.size)}</small></span>${needsChoice ? `<label>File type<select data-source-kind="${index}" ${sourceError ? 'aria-invalid="true"' : ""}><option value="">Choose type</option>${optionsFor(selected)}</select></label>` : ""}<button type="button" aria-label="Remove ${escapeHtml(file.name)}" data-remove-task-file="${index}">Remove</button></li>`;
  }).join("");
  const pasted = String(draft.pastedContent || "").trim() ? `<li class="source-item"><span class="source-icon" aria-hidden="true">TEXT</span><span><strong>Pasted text</strong><small>Ready to analyze</small></span><button type="button" data-clear-pasted>Remove</button></li>` : "";
  const local = String(draft.paths || "").trim().split(/\r?\n/).filter(Boolean).map((path) => `<li class="source-item"><span class="source-icon" aria-hidden="true">LOCAL</span><span><strong>${escapeHtml(path.split(/[\\/]/).pop())}</strong><small>Approved local source will be checked by Sparse</small></span></li>`).join("");
  return `<fieldset class="task-fieldset source-fieldset"><legend>Sources</legend>
    <label id="task-dropzone" class="task-dropzone">Add files
      <input id="task-files" type="file" multiple accept="${escapeHtml((copy.accepted_uploads || []).join(","))}">
      <span>Choose or drop files here.</span>
    </label>
    <ul id="task-file-list" class="source-list">${files}${pasted}${local || (!files && !pasted ? `<li class="source-empty">No sources added yet.</li>` : "")}</ul>
    <details class="source-additional"><summary>Add pasted text or an approved local source</summary>
      <label>${escapeHtml(copy.paste_label || "Pasted text")}<textarea rows="4" data-task-field="pastedContent" placeholder="Paste supporting text">${escapeHtml(draft.pastedContent)}</textarea></label>
      <label class="${issues.fields?.paths ? "has-error" : ""}">${escapeHtml(copy.source_label || "Approved local source")}<textarea rows="2" data-task-field="paths" ${issues.fields?.paths ? 'aria-invalid="true" aria-describedby="paths-error"' : ""} placeholder="${escapeHtml(copy.source_placeholder || "One approved path per line")}">${escapeHtml(draft.paths)}</textarea><small>Sparse checks every path against its configured allowed locations.</small>${issues.fields?.paths ? `<small class="field-error" id="paths-error">${escapeHtml(issues.fields.paths)}</small>` : ""}</label>
    </details>
  </fieldset>`;
}

function taskToolFields(template, draft) {
  if (!(template.tools || []).length) return "";
  return `<fieldset class="task-fieldset"><legend>Optional checks</legend><div class="choice-grid">${template.tools.map((tool) => {
    const permitted = tool.mode === "read_only" && can("tool:diagnose");
    const enabled = tool.available && permitted;
    const note = tool.mode === "read_only"
      ? `Read-only diagnostic${can("tool:diagnose") ? "" : " · your role cannot run tools"}`
      : "Remediation is available only through an administrator-confirmed advanced payload";
    return `<label class="choice-card ${tool.mode === "state_changing" ? "is-risky" : ""}"><input type="checkbox" data-task-tool="${escapeHtml(tool.id)}" ${draft.tools.includes(tool.id) && enabled ? "checked" : ""} ${enabled ? "" : "disabled"}><span><strong>${escapeHtml(tool.label)}</strong><small>${escapeHtml(note)}${tool.available ? "" : " · unavailable"}</small></span></label>`;
  }).join("")}</div></fieldset>`;
}

function taskSpecificFields(template, draft) {
  if (template.kind === "developer") {
    return `${taskToolFields(template, draft)}<fieldset class="task-fieldset"><legend>Review depth</legend><label class="check-row task-check"><input type="checkbox" data-task-flag="ambiguous" ${draft.flags.includes("ambiguous") ? "checked" : ""}>Add critic and reconciliation stages when available</label></fieldset>`;
  }
  if (["incident", "troubleshooting"].includes(template.kind)) {
    return `${taskToolFields(template, draft)}<fieldset class="task-fieldset"><legend>Safety boundary</legend><label class="check-row task-check"><input type="checkbox" data-task-field="highRisk" ${draft.highRisk ? "checked" : ""}>This involves a high-risk interpretation or change</label><label class="check-row task-check"><input type="checkbox" data-task-field="humanReview" ${draft.humanReview ? "checked" : ""}>Require human review before acceptance</label><p>Guided tasks request diagnostics only. State-changing remediation remains a separate administrator-confirmed operation.</p></fieldset>`;
  }
  if (template.kind === "document") {
    return `<fieldset class="task-fieldset"><legend>Review policy</legend><label class="check-row task-check"><input type="checkbox" data-task-field="highRisk" ${draft.highRisk ? "checked" : ""}>The interpretation may affect a high-stakes decision</label><label class="check-row task-check"><input type="checkbox" data-task-field="humanReview" ${draft.humanReview ? "checked" : ""}>Require human review for high-risk conclusions</label></fieldset>`;
  }
  if (template.kind === "research") {
    return `<fieldset class="task-fieldset"><legend>Evidence retrieval</legend><label class="check-row task-check"><input type="checkbox" data-task-field="indexSources" ${draft.indexSources ? "checked" : ""}>Index text sources and retrieve relevant passages before synthesis</label><label>Maximum retrieved passages<input type="number" min="1" max="20" data-task-field="topK" value="${escapeHtml(draft.topK)}"></label></fieldset>`;
  }
  if (template.kind === "batch") {
    const issues = state.taskIssues[draft.id]?.fields || {};
    return `<fieldset class="task-fieldset"><legend>Records and validation</legend><label class="${issues.recordsJson ? "has-error" : ""}">Records JSON<textarea rows="10" data-task-field="recordsJson" spellcheck="false" ${issues.recordsJson ? 'aria-invalid="true" aria-describedby="records-error"' : ""}>${escapeHtml(draft.recordsJson)}</textarea><small>A JSON array. Stable record IDs prevent duplicate accepted output.</small>${issues.recordsJson ? `<small class="field-error" id="records-error">${escapeHtml(issues.recordsJson)}</small>` : ""}</label><label class="${issues.schemaJson ? "has-error" : ""}">Record schema JSON<textarea rows="8" data-task-field="schemaJson" spellcheck="false" ${issues.schemaJson ? 'aria-invalid="true" aria-describedby="schema-error"' : ""}>${escapeHtml(draft.schemaJson)}</textarea>${issues.schemaJson ? `<small class="field-error" id="schema-error">${escapeHtml(issues.schemaJson)}</small>` : ""}</label><label>Maximum retries<input type="number" min="0" max="2" data-task-field="maximumRetries" value="${escapeHtml(draft.maximumRetries)}"></label></fieldset>`;
  }
  if (template.kind === "meeting") {
    return `<fieldset class="task-fieldset"><legend>Transcript preferences</legend><div class="form-grid"><label>Language<input data-task-field="language" value="${escapeHtml(draft.language)}" placeholder="auto or language code"></label><label>Names and hotwords<input data-task-field="hotwords" value="${escapeHtml(draft.hotwords)}" placeholder="Comma-separated"></label></div></fieldset>`;
  }
  if (template.kind === "experiment") {
    return `<fieldset class="task-fieldset"><legend>Frozen evaluation identity</legend><div class="form-grid"><label>Dataset revision<input data-task-field="datasetRevision" value="${escapeHtml(draft.datasetRevision)}" placeholder="sha256:…"></label><label>Prompt revision<input data-task-field="promptRevision" value="${escapeHtml(draft.promptRevision)}" placeholder="sha256:…"></label></div><label>Baseline metrics JSON<textarea rows="8" data-task-field="baselinesJson" spellcheck="false">${escapeHtml(draft.baselinesJson)}</textarea></label></fieldset>`;
  }
  return taskToolFields(template, draft);
}

function taskPlanMarkup(template, draft, maximumStages) {
  const plan = taskPlan(template, draft, maximumStages);
  return `<div class="task-plan-summary"><div><span>At most</span><strong>${plan.maximumModelCalls}</strong><small>model stages</small></div><div><span>Unavailable</span><strong>${plan.unavailable.length}</strong><small>planned capabilities</small></div></div><ol class="task-plan-steps">${plan.steps.map((step, index) => `<li class="${escapeHtml(step.state)}"><span>${index + 1}</span><div><strong>${escapeHtml(humanize(step.title))}</strong><p>${escapeHtml(step.detail)}</p>${step.endpoints?.length ? `<small>${escapeHtml(step.endpoints.join(", "))}</small>` : ""}</div></li>`).join("")}</ol>`;
}

function documentOutputFields(draft) {
  const issues = state.taskIssues[draft.id]?.fields || {};
  const fields = (draft.documentFields || []).map((field, index) => {
    const error = issues[`documentField:${index}`];
    return `<div class="document-field ${error ? "has-error" : ""}"><input aria-label="Field name" data-document-field="label" data-field-index="${index}" value="${escapeHtml(field.label)}" ${error ? `aria-invalid="true" aria-describedby="document-field-error-${index}"` : ""} placeholder="Field name"><select aria-label="Field type" data-document-field="type" data-field-index="${index}"><option value="string" ${field.type === "string" ? "selected" : ""}>Text</option><option value="number" ${field.type === "number" ? "selected" : ""}>Number</option><option value="date" ${field.type === "date" ? "selected" : ""}>Date</option><option value="boolean" ${field.type === "boolean" ? "selected" : ""}>Yes / no</option></select><label><input type="checkbox" data-document-field="required" data-field-index="${index}" ${field.required ? "checked" : ""}> Required</label><button type="button" data-remove-document-field="${index}">Remove</button>${error ? `<small class="field-error" id="document-field-error-${index}">${escapeHtml(error)}</small>` : ""}</div>`;
  }).join("");
  return `<fieldset class="task-fieldset ${issues.documentFields ? "has-error" : ""}"><legend>Result format</legend><div class="segmented-control"><label><input type="radio" name="output-intent" data-task-field="outputIntent" value="answer" ${draft.outputIntent === "answer" ? "checked" : ""}> Answer</label><label><input type="radio" name="output-intent" data-task-field="outputIntent" value="summary" ${draft.outputIntent === "summary" ? "checked" : ""}> Summary</label><label><input type="radio" name="output-intent" data-task-field="outputIntent" value="table" ${draft.outputIntent === "table" ? "checked" : ""}> Extracted table</label></div>${draft.outputIntent === "table" ? `<div class="document-fields">${fields}<button class="button secondary small" type="button" data-add-document-field>Add field</button>${issues.documentFields ? `<small class="field-error">${escapeHtml(issues.documentFields)}</small>` : ""}</div>` : ""}</fieldset>`;
}

function documentOcrFields(template, draft) {
  const modes = template.task_ui?.ocr_modes || [
    { id: "fast", label: "Fast", summary: "Quick text extraction for straightforward pages." },
    { id: "complex", label: "Complex", summary: "Document parsing for tables, formulas, and complex layouts." },
  ];
  return `<fieldset class="task-fieldset"><legend>OCR mode</legend><div class="ocr-mode-grid">${modes.map((mode) => `<label class="ocr-mode-card"><input type="radio" name="ocr-mode" data-task-field="ocrMode" value="${escapeHtml(mode.id)}" ${draft.ocrMode === mode.id ? "checked" : ""}><span><strong>${escapeHtml(mode.label)}</strong><small>${escapeHtml(mode.summary)}</small></span></label>`).join("")}</div></fieldset>`;
}

function taskEntries() {
  const useCases = state.snapshot?.use_cases || {};
  const jobIds = new Set((useCases.jobs || []).map((value) => value.id));
  return [...(useCases.jobs || []), ...(useCases.runs || []).filter((value) => !jobIds.has(value.id))];
}

function taskRecord(id) {
  const record = taskEntries().find((value) => value.id === id) || null;
  const detail = state.runDetails[id];
  if (!detail) return record;
  return record ? { ...record, result: detail } : detail;
}

async function hydrateActiveRun() {
  const draft = state.activeDraftId ? draftById(state.activeDraftId) : null;
  if (!draft?.runId) return;
  const runIds = Array.isArray(draft.runIds) && draft.runIds.length
    ? draft.runIds
    : [draft.runId];
  for (const runId of runIds) {
    const record = taskEntries().find((value) => value.id === runId);
    if (record && !record.result && ["queued", "preparing_models", "running", "cancelling"].includes(record.state)) continue;
    try {
      state.runDetails[runId] = await api.useCase(runId);
    } catch (error) {
      if (!(error instanceof ApiError) || error.status !== 404) throw error;
    }
  }
}

async function loadRunChain(runId, maximumTurns = 20) {
  const values = [];
  const seen = new Set();
  let currentId = runId;
  while (currentId && !seen.has(currentId) && values.length < maximumTurns) {
    seen.add(currentId);
    const run = await api.useCase(currentId);
    state.runDetails[currentId] = run;
    values.unshift(run);
    currentId = String(run.parent_run_id || "");
  }
  return values;
}

function renderDeliverable(run) {
  return renderDeliverableView(run, { editable: run?.kind === "document" && can("usecase:run") });
}

function workspaceResult(draft) {
  const runIds = Array.isArray(draft.runIds) && draft.runIds.length
    ? draft.runIds
    : draft.runId ? [draft.runId] : [];
  const records = runIds.map((runId) => taskRecord(runId)).filter(Boolean);
  const latest = records.at(-1);
  const run = latest?.result || latest;
  return renderWorkspaceThread({
    draft,
    records,
    editable: run?.kind === "document" && can("usecase:run"),
    canFollowUp: can("usecase:run"),
  });
}

function taskComposer(template, maximumStages) {
  const draft = draftFor(template);
  const copy = taskCopy(template);
  const readiness = state.readiness;
  return renderWorkspaceView({
    template,
    draft,
    copy,
    readiness,
    issues: state.taskIssues[draft.id],
    sourceFields: taskSourceFields(template, draft),
    documentFields: template.kind === "document" ? `${documentOcrFields(template, draft)}${documentOutputFields(draft)}` : "",
    specificFields: taskSpecificFields(template, draft),
    planMarkup: taskPlanMarkup(template, draft, maximumStages),
    resultMarkup: workspaceResult(draft),
    showRequestForm: !draft.runId,
  });
}

function renderHome() {
  const templates = state.snapshot.use_cases?.catalog?.templates || [];
  return renderHomeView({ state, templates, drafts: allDrafts() });
}

function renderMyWork() {
  const templates = state.snapshot.use_cases?.catalog?.templates || [];
  return renderMyWorkView({
    drafts: allDrafts(),
    entries: taskEntries(),
    templates,
    canCancel: can("request:cancel"),
    actionButton,
    emptyState,
  });
}

function renderWorkspace() {
  const template = selectedTaskTemplate();
  if (!template) return emptyState("Task not found", "Return Home and choose a task.");
  state.useCaseTemplate = template.id;
  const draft = draftFor(template);
  if (!state.activeDraftId) state.activeDraftId = draft.id;
  return taskComposer(template, state.snapshot.use_cases.catalog.maximum_model_stages);
}

function renderSettings() {
  return renderSettingsView({ snapshot: state.snapshot, selectedProfile: activeProfile(), status: systemStatus() });
}

const renderers = {
  home: renderHome,
  work: renderMyWork,
  workspace: renderWorkspace,
  settings: renderSettings,
  overview: renderOverview,
  fleet: renderFleet,
  requests: renderRequests,
  vision: renderVision,
  escalation: renderEscalation,
  evaluation: renderEvaluation,
  configuration: renderConfiguration,
};

function render() {
  renderShellChrome({ state, status: systemStatus() });
  if (!state.snapshot) return;
  main.innerHTML = renderers[state.view]();
  if (!main.contains(document.activeElement)) main.focus({ preventScroll: true });
  bindDetailActions();
  hydrateArtifactPreviews(main);
}

function updateChrome() {
  renderShellChrome({ state, status: systemStatus() });
}

function populateProfiles() {
  const select = document.querySelector("#profile-select");
  if (!select) return;
  const profiles = (state.snapshot?.profiles?.profiles || []).filter((profile) => ["16", "24", "32"].includes(profile.selection_key));
  select.innerHTML = `<option value="auto">Auto-detect</option>${profiles.map((profile) => `<option value="${escapeHtml(profile.id)}">${escapeHtml(profile.display_name || `${profile.selection_key} GB`)}</option>`).join("")}`;
  select.value = [...select.options].some((option) => option.value === state.selection) ? state.selection : "auto";
  state.selection = select.value;
  select.disabled = false;
}

function showBanner(message, tone = "warn") {
  const banner = document.querySelector("#system-banner");
  banner.textContent = message;
  banner.className = `system-banner ${tone}`;
}

function hideBanner() { document.querySelector("#system-banner").className = "system-banner is-hidden"; }

function reconcileVisibleTaskStatus() {
  reconcileStatusElements(main, taskRecord, taskState);
}

function reconcileWorkspaceResult() {
  reconcileVisibleTaskStatus();
  if (state.view !== "workspace") return false;
  const template = selectedTaskTemplate();
  if (!template) return false;
  const draft = draftFor(template);
  const current = main.querySelector("[data-workspace-result]");
  if (!current) return false;
  replaceRegionPreservingContext({
    current,
    markup: workspaceResult(draft),
    bind: bindWorkspaceResultActions,
  });
  return true;
}

async function loadSnapshot({ refreshView = true } = {}) {
  if (!state.token) {
    authDialog.showModal();
    throw new ApiError("Authentication required", 401);
  }
  const snapshot = await api.snapshot();
  state.snapshot = snapshot;
  state.lastEventId = Math.max(0, ...snapshot.events.map((event) => Number(event.event_id || 0)));
  state.connected = true;
  state.stale = false;
  state.lastSuccess = Date.now();
  if (state.view === "workspace") await hydrateActiveRun();
  populateProfiles();
  hideBanner();
  updateChrome();
  if (refreshView) render();
  else if (!reconcileWorkspaceResult()) reconcileVisibleTaskStatus();
}

async function pollEvents() {
  if (state.polling) return;
  state.polling = true;
  try {
    const events = await api.events(state.lastEventId);
    if (events.length && state.snapshot) {
      state.snapshot.events.push(...events);
      state.snapshot.events = state.snapshot.events.slice(-500);
      state.lastEventId = Math.max(state.lastEventId, ...events.map((event) => Number(event.event_id || 0)));
      await loadSnapshot({
        refreshView: shouldRefreshAfterEvent({ view: state.view, activeElement: document.activeElement, main }),
      });
    } else {
      state.connected = true;
      state.lastSuccess = Date.now();
      state.stale = false;
    }
  } catch (error) {
    state.connected = false;
    state.stale = true;
    showBanner(`Connection lost. The task may still be running. Sparse will reconnect to the same task automatically. ${error.message}`, "bad");
  } finally {
    state.polling = false;
    updateChrome();
  }
}

async function showDetail(title, payload) {
  document.querySelector("#detail-title").textContent = title;
  document.querySelector("#detail-body").innerHTML = `<pre class="json-view">${escapeHtml(JSON.stringify(payload, null, 2))}</pre>`;
  detailDialog.showModal();
}

function hydrateArtifactPreviews(root) {
  for (const preview of root.querySelectorAll("[data-artifact-preview]")) {
    const artifactId = preview.dataset.artifactPreview;
    const metadata = state.snapshot.artifacts.find((artifact) => artifact.id === artifactId);
    api.artifactContent(artifactId).then((content) => {
      if (!preview.isConnected || content?.encoding !== "base64") return;
      const mediaType = String(metadata?.media_type || "").split(";", 1)[0];
      if (!mediaType.startsWith("image/")) return;
      const image = document.createElement("img");
      image.alt = `${metadata?.kind || "Visual artifact"} preview`;
      image.loading = "lazy";
      image.src = `data:${mediaType};base64,${content.content}`;
      preview.replaceChildren(image);
    }).catch(() => {
      if (preview.isConnected) preview.textContent = "Preview unavailable";
    });
  }
}

async function showArtifactDetail(artifactId) {
  const metadata = await api.artifact(artifactId);
  let content = null;
  try { content = await api.artifactContent(artifactId); } catch { /* Metadata remains viewable for read-only roles. */ }
  const mediaType = String(metadata.media_type || "application/octet-stream").split(";", 1)[0];
  let viewer = "";
  if (content?.encoding === "utf-8") viewer = `<pre class="json-view artifact-content">${escapeHtml(content.content)}</pre>`;
  if (content?.encoding === "base64" && mediaType.startsWith("image/")) viewer = `<img class="artifact-image" alt="Persisted visual artifact" src="data:${escapeHtml(mediaType)};base64,${content.content}">`;
  if (content?.encoding === "base64" && mediaType.startsWith("audio/")) viewer = `<audio class="artifact-media" controls src="data:${escapeHtml(mediaType)};base64,${content.content}"></audio>`;
  if (content?.encoding === "base64" && mediaType.startsWith("video/")) viewer = `<video class="artifact-media" controls src="data:${escapeHtml(mediaType)};base64,${content.content}"></video>`;
  document.querySelector("#detail-title").textContent = "Artifact evidence";
  document.querySelector("#detail-body").innerHTML = `${viewer}<pre class="json-view">${escapeHtml(JSON.stringify(metadata, null, 2))}</pre>`;
  detailDialog.showModal();
}

function requestConfirmation({ title, phrase, summary, licenseEndpoints = [] }) {
  document.querySelector("#confirm-title").textContent = title;
  document.querySelector("#confirm-summary").innerHTML = summary;
  document.querySelector("#confirm-phrase").textContent = phrase;
  const input = document.querySelector("#confirm-input");
  input.value = "";
  input.setCustomValidity("");
  const licenseRow = document.querySelector("#license-confirm-row");
  const license = document.querySelector("#license-confirm");
  license.checked = false;
  license.setCustomValidity("");
  licenseRow.classList.toggle("is-hidden", licenseEndpoints.length === 0);
  if (licenseEndpoints.length) licenseRow.lastChild.textContent = ` I reviewed the licenses for: ${licenseEndpoints.join(", ")}.`;
  confirmDialog.showModal();
  input.focus();
  return new Promise((resolve) => { confirmationResolver = { resolve, phrase, needsLicense: licenseEndpoints.length > 0 }; });
}

async function runAction(label, callback) {
  state.actionPending = true;
  showBanner(`${label} in progress…`, "warn");
  render();
  try {
    await callback();
    await loadSnapshot();
    showBanner(`${label} completed.`, "good");
  } catch (error) {
    showBanner(`${label} failed: ${error.message}`, "bad");
  } finally {
    state.actionPending = false;
    render();
  }
}

async function planAndApplyProfile(profileId, customProfile = null) {
  await runAction("Profile validation", async () => {
    const plan = await api.planProfile(profileId, customProfile);
    if (!plan.valid) {
      await showDetail("Profile validation failed", plan);
      throw new ApiError("The profile did not pass controller validation", 409);
    }
    const resource = plan.resources;
    const answer = await requestConfirmation({
      title: `Apply ${plan.profile_id}`,
      phrase: plan.confirmation_phrase,
      licenseEndpoints: plan.validation.license_review_endpoints,
      summary: `<div class="confirmation-summary"><p>${escapeHtml(plan.current_profile)} → <strong>${escapeHtml(plan.profile_id)}</strong></p><dl><div><dt>Add</dt><dd>${escapeHtml(plan.additions.join(", ") || "none")}</dd></div><div><dt>Remove</dt><dd>${escapeHtml(plan.removals.join(", ") || "none")}</dd></div><div><dt>Qwen tier</dt><dd>${escapeHtml(plan.qwen_quant_change.from.join(", ") || "none")} → ${escapeHtml(plan.qwen_quant_change.to.join(", ") || "none")}</dd></div><div><dt>VRAM + reserve</dt><dd>${formatBytes(resource.proposed_vram_bytes)} + ${formatBytes(resource.required_reserve_bytes)}</dd></div><div><dt>Queued / active</dt><dd>${plan.queue_impact.queued} / ${plan.queue_impact.active}</dd></div></dl></div>`,
    });
    if (!answer) throw new ApiError("Profile switch cancelled", 0);
    await api.applyProfile(plan.id, answer.phrase, answer.licenseAcknowledged);
  });
}

function selectedTaskTemplate() {
  const templates = state.snapshot?.use_cases?.catalog?.templates || [];
  const activeDraft = state.activeDraftId ? draftById(state.activeDraftId) : null;
  if (activeDraft) {
    return templates.find((value) => value.id === activeDraft.templateId) || null;
  }
  return templates.find((value) => value.id === state.useCaseTemplate) || templates[0] || null;
}

function updateTaskPlanPreview(template, draft) {
  const destination = document.querySelector("#task-plan-body");
  if (destination) destination.innerHTML = taskPlanMarkup(
    template,
    draft,
    state.snapshot?.use_cases?.catalog?.maximum_model_stages || 3,
  );
}

function storeTaskField(target, draft) {
  const name = target.dataset.taskField;
  if (!name) return;
  if (target.type === "checkbox") draft[name] = target.checked;
  else if (target.type === "number") draft[name] = Number(target.value);
  else draft[name] = target.value;
}

function bytesToBase64(buffer) {
  const bytes = new Uint8Array(buffer);
  const parts = [];
  const chunkSize = 0x8000;
  for (let offset = 0; offset < bytes.length; offset += chunkSize) {
    parts.push(String.fromCharCode(...bytes.subarray(offset, offset + chunkSize)));
  }
  return btoa(parts.join(""));
}

async function taskFileInput(template, file, kindOverride = null) {
  const classification = classifyUpload(template, file);
  const kind = kindOverride || classification.kind;
  if (!kind || !template.input_kinds.includes(kind)) {
    throw new Error(`${file.name}: ${classification.reason}. Choose a supported source type.`);
  }
  if (file.needsReselection || typeof file.arrayBuffer !== "function") {
    throw new Error(`${file.name} must be selected again before this task can start.`);
  }
  const lower = file.name.toLowerCase();
  const textual = String(file.type || "").startsWith("text/")
    || /\.(txt|md|log|py|js|ts|tsx|jsx|json|jsonl|csv|ya?ml|toml|ini|patch|diff)$/.test(lower);
  if (textual) {
    return { kind, content: await file.text(), metadata: { source_name: file.name } };
  }
  const suffixMatch = lower.match(/(\.[a-z0-9]{1,8})$/);
  return {
    kind,
    content_base64: bytesToBase64(await file.arrayBuffer()),
    media_type: file.type || "application/octet-stream",
    suffix: suffixMatch?.[1] || ".bin",
    render_pages: kind === "pdf",
    metadata: { source_name: file.name },
  };
}

async function startTaskExecution(template, draft, { automatic = false } = {}) {
  const localIssues = validateTaskDraft(template, draft);
  if (hasTaskIssues(localIssues)) {
    state.taskIssues[draft.id] = localIssues;
    render();
    focusFirstTaskIssue();
    if (automatic) showBanner("Sparse needs the highlighted information before it can start.", "warn");
    return false;
  }
  delete state.taskIssues[draft.id];
  try {
    const inputs = await Promise.all((draft.files || []).map((file, index) => taskFileInput(template, file, draft.fileKinds?.[index])));
    const payload = buildTaskPayload(template, draft, inputs, false);
    state.readiness = await api.useCaseReadiness(payload);
    if (state.readiness.state === "blocked") {
      throw new Error(state.readiness.blocking_reasons.join(" ") || "A required capability is unavailable.");
    }
    if (state.readiness.state === "needs_preparation") {
      render();
      if (!state.readiness.preparation?.permitted) {
        showBanner("The required installed models must be prepared by an authorized role.", "warn");
        return false;
      }
      if (automatic) {
        showBanner("Review the model preparation step, then choose Prepare models and start.", "warn");
        return false;
      }
    }
    state.actionPending = true;
    const job = await api.createUseCase(payload);
    draft.runId = job.id;
    draft.runIds = [...new Set([...(draft.runIds || []), job.id])];
    draft.turns = [...(draft.turns || []), { runId: job.id, prompt: draft.prompt }];
    state.activeRunId = job.id;
    persistDraft(draft);
    delete state.taskIssues[draft.id];
    await loadSnapshot();
    showBanner("Task started. You can leave this page and return from My work.", "good");
    return true;
  } catch (error) {
    state.taskIssues[draft.id] = mergeTaskIssues(
      state.taskIssues[draft.id],
      issuesFromError(error, draft),
    );
    render();
    focusFirstTaskIssue();
    showBanner(`Task could not start: ${error.message}`, "bad");
    return false;
  } finally {
    state.actionPending = false;
  }
}

async function submitFollowUp(template, draft, { preparationConfirmed = false } = {}) {
  const prompt = String(draft.followUp || "").trim();
  try {
    const payload = buildFollowUpPayload(template, draft, prompt, draft.runId);
    state.readiness = await api.useCaseReadiness(payload);
    if (state.readiness.state === "blocked") {
      throw new Error(state.readiness.blocking_reasons.join(" ") || "This follow-up cannot run with the available capabilities.");
    }
    if (state.readiness.state === "needs_preparation") {
      if (!state.readiness.preparation?.permitted) {
        throw new Error("The models needed for this follow-up must be prepared by an authorized role.");
      }
      if (!preparationConfirmed) {
        draft.followUpNeedsPreparation = true;
        persistDraft(draft);
        render();
        showBanner("Review the model preparation step, then choose Prepare models and send.", "warn");
        return;
      }
    }
    state.actionPending = true;
    const job = await api.createUseCase(payload);
    draft.runId = job.id;
    draft.runIds = [...new Set([...(draft.runIds || []), job.id])];
    draft.turns = [...(draft.turns || []), { runId: job.id, prompt }];
    draft.followUp = "";
    draft.followUpNeedsPreparation = false;
    state.activeRunId = job.id;
    persistDraft(draft);
    await loadSnapshot();
    showBanner("Follow-up started with the previous result and sources attached.", "good");
  } catch (error) {
    showBanner(`Follow-up could not start: ${error.message}`, "bad");
  } finally {
    state.actionPending = false;
  }
}

function bindResultActions(root = main) {
  root.querySelectorAll("[data-save-corrections]").forEach((button) => button.addEventListener("click", async () => {
    const changes = [];
    const textInput = root.querySelector("[data-result-text]");
    if (textInput && textInput.value !== textInput.dataset.original) {
      changes.push({ field_id: "$text", value: textInput.value, source_reference: null });
    }
    root.querySelectorAll("[data-result-cell]").forEach((input) => {
      if (input.value === input.dataset.original) return;
      changes.push({
        field_id: input.dataset.fieldId,
        row_index: Number(input.dataset.rowIndex),
        value: input.value,
        source_reference: null,
      });
    });
    if (!changes.length) {
      showBanner("No result changes to save.", "warn");
      return;
    }
    await runAction("Reviewed version", () => api.correctUseCase(button.dataset.saveCorrections, {
      version_id: button.dataset.versionId,
      changes,
    }));
  }));
  root.querySelectorAll("[data-export-result]").forEach((button) => button.addEventListener("click", async () => {
    try {
      const exported = await api.exportUseCase(button.dataset.exportResult, {
        version_id: button.dataset.versionId,
        format: button.dataset.exportFormat,
      });
      await showArtifactDetail(exported.artifact.id);
      showBanner(`${button.dataset.exportFormat.toUpperCase()} export created.`, "good");
    } catch (error) {
      showBanner(`Export failed: ${error.message}`, "bad");
    }
  }));
  root.querySelectorAll("[data-detail-artifact]").forEach((button) => button.addEventListener("click", async () => {
    try { await showArtifactDetail(button.dataset.detailArtifact); }
    catch (error) { showBanner(error.message, "bad"); }
  }));
  root.querySelectorAll("[data-focus-result-field]").forEach((button) => button.addEventListener("click", () => {
    const field = button.dataset.focusResultField;
    const row = button.dataset.rowIndex;
    const input = root.querySelector(`[data-result-cell][data-row-index="${CSS.escape(row)}"][data-field-id="${CSS.escape(field)}"]`)
      || [...root.querySelectorAll("[data-result-cell]")].find((value) => value.dataset.rowIndex === row && value.dataset.fieldId.toLowerCase() === field.toLowerCase());
    input?.scrollIntoView({ behavior: "smooth", block: "center" });
    input?.focus();
  }));
  root.querySelectorAll("[data-focus-batch-record]").forEach((button) => button.addEventListener("click", () => {
    const record = root.querySelector(`[data-batch-record="${CSS.escape(button.dataset.focusBatchRecord)}"]`);
    record?.scrollIntoView({ behavior: "smooth", block: "center" });
    record?.focus();
  }));
  bindReplayActions(root);
}

function bindWorkspaceResultActions(root = main) {
  bindResultActions(root);
  const form = root.querySelector("#task-followup-form");
  const field = root.querySelector("#task-followup");
  const template = selectedTaskTemplate();
  const draft = template ? draftFor(template) : null;
  if (!form || !field || !template || !draft) return;
  field.addEventListener("input", () => {
    draft.followUp = field.value;
    draft.followUpNeedsPreparation = false;
    persistDraft(draft);
  });
  field.addEventListener("keydown", (event) => {
    if (event.key !== "Enter" || event.shiftKey || event.isComposing) return;
    event.preventDefault();
    form.requestSubmit();
  });
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    draft.followUp = field.value;
    persistDraft(draft);
    await submitFollowUp(template, draft, {
      preparationConfirmed: Boolean(draft.followUpNeedsPreparation),
    });
  });
}

function bindReplayActions(root = main) {
  root.querySelectorAll("[data-replay-usecase]").forEach((button) => {
    if (button.dataset.boundReplay) return;
    button.dataset.boundReplay = "true";
    button.addEventListener("click", async () => {
      const runId = button.dataset.replayUsecase;
      const phrase = `REPLAY ${runId}`;
      const answer = await requestConfirmation({ title: "Replay dead-letter records", phrase, summary: `<p>Only failed record identifiers are replayed; accepted outputs are not duplicated.</p>` });
      if (answer) await runAction("Batch replay", () => api.replayUseCase(runId, { confirmation: answer.phrase, record_schema: {} }));
    });
  });
}

function focusFirstTaskIssue() {
  const target = main.querySelector('[aria-invalid="true"], .task-error-summary');
  target?.scrollIntoView({ behavior: "smooth", block: "center" });
  if (typeof target?.focus === "function") target.focus({ preventScroll: true });
}

function bindDetailActions() {
  const templates = state.snapshot?.use_cases?.catalog?.templates || [];
  bindWorkspaceResultActions();
  main.querySelectorAll("[data-home-template]").forEach((button) => button.addEventListener("click", () => {
    const template = templates.find((value) => value.id === button.dataset.homeTemplate);
    if (!template) return;
    const example = STARTER_EXAMPLES[template.kind];
    openWorkspace(template, {
      prompt: state.taskIntent || example?.request || "",
      files: state.homeFiles,
    });
  }));
  const homeForm = main.querySelector("#home-task-form");
  if (homeForm) homeForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    state.taskIntent = main.querySelector("#home-intent").value.trim();
    state.taskSuggestion = suggestTask(
      state.taskIntent,
      state.homeFiles.map((file) => file.name),
      templates,
    );
    if (state.taskSuggestion.status === "matched") {
      const template = templates.find((value) => value.id === state.taskSuggestion.templateId);
      if (template) {
        const draft = openWorkspace(template, { prompt: state.taskIntent, files: state.homeFiles });
        state.taskIntent = "";
        state.homeFiles = [];
        await startTaskExecution(template, draft, { automatic: true });
      }
      return;
    }
    render();
  });
  const homeIntent = main.querySelector("#home-intent");
  if (homeIntent) homeIntent.addEventListener("input", () => { state.taskIntent = homeIntent.value; });
  const homeFiles = main.querySelector("#home-files");
  if (homeFiles) homeFiles.addEventListener("change", () => {
    state.homeFiles.push(...homeFiles.files);
    render();
  });
  main.querySelectorAll("[data-remove-home-file]").forEach((button) => button.addEventListener("click", () => {
    state.homeFiles = state.homeFiles.filter((file) => file.name !== button.dataset.removeHomeFile);
    render();
  }));
  main.querySelectorAll("[data-resume-draft]").forEach((button) => button.addEventListener("click", () => {
    const draft = draftById(button.dataset.resumeDraft);
    if (!draft) return;
    state.activeDraftId = draft.id;
    state.useCaseTemplate = draft.templateId;
    state.view = "workspace";
    location.hash = `workspace/${encodeURIComponent(draft.id)}`;
    render();
  }));
  main.querySelectorAll("[data-delete-draft]").forEach((button) => button.addEventListener("click", () => {
    const draft = draftById(button.dataset.deleteDraft);
    if (!draft) return;
    deleteDraft(localStorage, draft.id);
    delete state.taskDrafts[draft.id];
    render();
  }));
  main.querySelectorAll("[data-clear-drafts]").forEach((button) => button.addEventListener("click", () => {
    clearDrafts(localStorage);
    state.taskDrafts = {};
    state.activeDraftId = null;
    render();
    showBanner("Local drafts cleared. Completed controller results were not deleted.", "good");
  }));
  main.querySelectorAll("[data-open-run]").forEach((button) => button.addEventListener("click", async () => {
    const record = taskRecord(button.dataset.openRun);
    const run = record?.result || record;
    const template = templates.find((value) => value.id === (run?.template_id || record?.template_id));
    if (!record || !template) return;
    const draft = allDrafts().find((value) => value.runId === record.id || value.runIds?.includes(record.id))
      || draftFor(template, { createNew: true });
    draft.runId = record.id;
    draft.runIds = [...new Set([...(draft.runIds || []), record.id])];
    if (!(draft.turns || []).some((value) => value.runId === record.id)) {
      draft.turns = [...(draft.turns || []), { runId: record.id, prompt: run.prompt || draft.prompt || "Open task" }];
    }
    draft.title = run.title || record.title || draft.title;
    persistDraft(draft);
    state.activeDraftId = draft.id;
    state.activeRunId = record.id;
    state.view = "workspace";
    location.hash = `workspace/${encodeURIComponent(draft.id)}`;
    try {
      const chain = await loadRunChain(record.id);
      if (chain.length) {
        draft.runIds = chain.map((value) => value.id);
        draft.runId = chain.at(-1).id;
        draft.prompt = chain[0].prompt || draft.prompt;
        draft.turns = chain.map((value, index) => ({
          runId: value.id,
          prompt: value.prompt || (index === 0 ? draft.prompt : "Follow-up request"),
        }));
        persistDraft(draft);
      }
    }
    catch (error) { showBanner(`Task details could not be loaded: ${error.message}`, "bad"); }
    render();
  }));
  main.querySelectorAll("[data-go-home], [data-change-task]").forEach((button) => button.addEventListener("click", () => {
    state.view = "home";
    location.hash = "home";
    render();
  }));
  main.querySelectorAll("[data-advanced-view]").forEach((button) => button.addEventListener("click", () => {
    state.view = button.dataset.advancedView;
    location.hash = state.view;
    render();
  }));
  const settingsProfile = main.querySelector("#profile-select");
  if (settingsProfile) settingsProfile.addEventListener("change", () => {
    state.selection = settingsProfile.value;
    state.view = "fleet";
    location.hash = "fleet";
    render();
  });
  main.querySelectorAll("[data-detail-model]").forEach((button) => button.addEventListener("click", (event) => { event.preventDefault(); showDetail("Model definition", modelFor(button.dataset.detailModel)); }));
  main.querySelectorAll("[data-detail-request]").forEach((button) => button.addEventListener("click", () => showDetail("Redacted request", state.snapshot.requests.find((item) => item.id === button.dataset.detailRequest))));
  main.querySelectorAll("[data-detail-run]").forEach((button) => button.addEventListener("click", async () => { try { await showDetail("Run manifest", await api.run(button.dataset.detailRun)); } catch (error) { showBanner(error.message, "bad"); } }));
  main.querySelectorAll("[data-detail-usecase]").forEach((button) => button.addEventListener("click", async () => { try { await showDetail("Use-case run", await api.useCase(button.dataset.detailUsecase)); } catch (error) { showBanner(error.message, "bad"); } }));
  main.querySelectorAll("[data-profile-plan]").forEach((button) => button.addEventListener("click", () => planAndApplyProfile(button.dataset.profilePlan)));
  main.querySelectorAll("[data-fleet-op]").forEach((button) => button.addEventListener("click", async () => {
    const operation = button.dataset.fleetOp;
    const endpoint = button.dataset.endpoint || null;
    if (operation === "smoke") {
      await runAction(`Smoke ${endpoint}`, () => api.fleetOperation(operation, endpoint));
      return;
    }
    const phrase = `CONFIRM ${operation} ${endpoint || "fleet"}`;
    const answer = await requestConfirmation({ title: `${operation} ${endpoint || "fleet"}`, phrase, summary: `<p>The controller will validate lifecycle and resource state before performing this operation.</p>` });
    if (answer) await runAction(`${operation} ${endpoint || "fleet"}`, () => api.fleetOperation(operation, endpoint, answer.phrase));
  }));
  main.querySelectorAll("[data-cancel-request]").forEach((button) => button.addEventListener("click", async () => {
    const requestId = button.dataset.cancelRequest;
    const phrase = `CANCEL ${requestId}`;
    const answer = await requestConfirmation({ title: "Cancel request", phrase, summary: `<p>Cancellation is cooperative. The trace will distinguish it from timeout, policy rejection, and failure.</p>` });
    if (answer) await runAction("Request cancellation", () => api.cancelRequest(requestId, answer.phrase));
  }));
  main.querySelectorAll("[data-cancel-usecase]").forEach((button) => button.addEventListener("click", async () => {
    const runId = button.dataset.cancelUsecase;
    const phrase = `CANCEL ${runId}`;
    const answer = await requestConfirmation({ title: "Cancel use-case run", phrase, summary: `<p>Cancellation is cooperative and remains visible in the controller audit timeline.</p>` });
    if (answer) await runAction("Use-case cancellation", () => api.cancelUseCase(runId, answer.phrase));
  }));
  main.querySelectorAll("[data-rerun-usecase]").forEach((button) => button.addEventListener("click", async () => {
    const runId = button.dataset.rerunUsecase;
    const entries = [...(state.snapshot.use_cases?.jobs || []), ...(state.snapshot.use_cases?.runs || [])];
    const entry = entries.find((value) => value.id === runId);
    const run = entry?.result || entry;
    if (!run) return;
    const inputs = (run.inputs || []).filter((value) => value.artifact_ids?.length).map((value) => ({ kind: value.kind, artifact_id: value.artifact_ids[0], metadata: value.metadata || {} }));
    await runAction("Use-case rerun", () => api.createUseCase({ template_id: run.template_id, title: `Rerun of ${runId}`, prompt: "Rerun with the preserved evidence graph", inputs, claims: [], plan_only: false }));
  }));
  main.querySelectorAll("[data-admit-usecase]").forEach((button) => button.addEventListener("click", async () => {
    const runId = button.dataset.admitUsecase;
    const phrase = `ADMIT ${runId}`;
    const answer = await requestConfirmation({ title: "Record admission decision", phrase, summary: `<p>This records an explicit decision manifest. It does not silently alter production routing.</p>` });
    if (answer) await runAction("Admission decision", () => api.decideUseCaseAdmission(runId, { decision: "admit", confirmation: answer.phrase, reason: "Frozen evaluation reviewed in dashboard" }));
  }));
  const configForm = main.querySelector("#config-form");
  if (configForm) configForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const changes = {
      "fleet.queue_capacity": Number(document.querySelector("#config-queue").value),
      "controller.request_timeout_seconds": Number(document.querySelector("#config-timeout").value),
      "controller.telemetry_interval_seconds": Number(document.querySelector("#config-telemetry").value),
    };
    const answer = await requestConfirmation({ title: "Apply configuration", phrase: "APPLY CONFIG", summary: `<pre class="json-view">${escapeHtml(JSON.stringify(changes, null, 2))}</pre>` });
    if (answer) await runAction("Configuration update", () => api.updateConfig(changes, answer.phrase));
  });
  const evaluationForm = main.querySelector("#evaluation-form");
  if (evaluationForm) evaluationForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    try {
      const suite = document.querySelector("#evaluation-suite").value.trim();
      const baselines = JSON.parse(document.querySelector("#evaluation-baselines").value);
      await runAction("Evaluation manifest", () => api.createEvaluation(suite, baselines));
    } catch (error) { showBanner(`Evaluation error: ${error.message}`, "bad"); }
  });
  const taskForm = main.querySelector("#task-workspace-form");
  const template = selectedTaskTemplate();
  const draft = template ? draftFor(template) : null;
  if (taskForm && template && draft) {
    taskForm.addEventListener("input", (event) => {
      const target = event.target;
      storeTaskField(target, draft);
      if (state.taskIssues[draft.id]) {
        delete state.taskIssues[draft.id];
        target.removeAttribute("aria-invalid");
        target.closest(".has-error")?.classList.remove("has-error");
      }
      if (target.name === "output-intent") {
        persistDraft(draft);
        render();
        return;
      }
      if (target.dataset.taskTool) {
        draft.tools = [...taskForm.querySelectorAll("[data-task-tool]:checked")].map((value) => value.dataset.taskTool);
      }
      if (target.dataset.taskFlag) {
        draft.flags = [...taskForm.querySelectorAll("[data-task-flag]:checked")].map((value) => value.dataset.taskFlag);
      }
      persistDraft(draft);
      updateTaskPlanPreview(template, draft);
    });
    taskForm.addEventListener("change", (event) => {
      const target = event.target;
      storeTaskField(target, draft);
      if (target.id === "task-files") {
        draft.files = [...(draft.files || []).filter((file) => !file.needsReselection), ...target.files];
        persistDraft(draft);
        render();
        return;
      }
      persistDraft(draft);
      updateTaskPlanPreview(template, draft);
    });
    const dropzone = taskForm.querySelector("#task-dropzone");
    if (dropzone) {
      dropzone.addEventListener("dragover", (event) => {
        event.preventDefault();
        dropzone.classList.add("is-dragging");
      });
      dropzone.addEventListener("dragleave", () => dropzone.classList.remove("is-dragging"));
      dropzone.addEventListener("drop", (event) => {
        event.preventDefault();
        dropzone.classList.remove("is-dragging");
        draft.files = [...(draft.files || []).filter((file) => !file.needsReselection), ...event.dataTransfer.files];
        persistDraft(draft);
        render();
      });
    }
    taskForm.addEventListener("submit", async (event) => {
      event.preventDefault();
      await startTaskExecution(template, draft);
    });
  }
  main.querySelectorAll("[data-remove-task-file]").forEach((button) => button.addEventListener("click", () => {
    const active = selectedTaskTemplate();
    if (!active) return;
    draftFor(active).files.splice(Number(button.dataset.removeTaskFile), 1);
    persistDraft(draftFor(active));
    render();
  }));
  main.querySelectorAll("[data-source-kind]").forEach((select) => select.addEventListener("change", () => {
    if (!draft) return;
    draft.fileKinds[Number(select.dataset.sourceKind)] = select.value;
    persistDraft(draft);
    render();
  }));
  main.querySelectorAll("[data-clear-pasted]").forEach((button) => button.addEventListener("click", () => {
    if (!draft) return;
    draft.pastedContent = "";
    persistDraft(draft);
    render();
  }));
  main.querySelectorAll("[data-add-document-field]").forEach((button) => button.addEventListener("click", () => {
    if (!draft) return;
    draft.documentFields.push({ id: `field-${draft.documentFields.length + 1}`, label: "", type: "string", required: false });
    persistDraft(draft);
    render();
  }));
  main.querySelectorAll("[data-remove-document-field]").forEach((button) => button.addEventListener("click", () => {
    if (!draft) return;
    draft.documentFields.splice(Number(button.dataset.removeDocumentField), 1);
    persistDraft(draft);
    render();
  }));
  main.querySelectorAll("[data-document-field]").forEach((input) => input.addEventListener("input", () => {
    if (!draft) return;
    const field = draft.documentFields[Number(input.dataset.fieldIndex)];
    if (!field) return;
    field[input.dataset.documentField] = input.type === "checkbox" ? input.checked : input.value;
    persistDraft(draft);
  }));
  const workspaceTitle = main.querySelector(".workspace-title");
  if (workspaceTitle && draft) workspaceTitle.addEventListener("input", () => {
    draft.title = workspaceTitle.value;
    persistDraft(draft);
  });
}

document.querySelector("#primary-nav").addEventListener("click", (event) => {
  const button = event.target.closest("[data-view]");
  if (!button) return;
  state.view = button.dataset.view;
  location.hash = state.view;
  document.body.classList.remove("nav-open");
  render();
});
document.querySelector("#connection-chip").addEventListener("click", () => {
  state.view = "settings";
  location.hash = "settings";
  render();
});
document.querySelector("#menu-button").addEventListener("click", () => document.body.classList.toggle("nav-open"));
document.querySelector("#session-button").addEventListener("click", () => { document.querySelector("#token-input").value = state.token; authDialog.showModal(); });
document.querySelector("#auth-cancel").addEventListener("click", () => authDialog.close());
document.querySelector("#auth-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const token = document.querySelector("#token-input").value.trim();
  if (token.length < 16) return;
  state.token = token;
  api.setToken(token);
  sessionStorage.setItem("sparse-dashboard-token", token);
  authDialog.close();
  main.innerHTML = `<section class="loading-state"><div class="spinner"></div><h2>Validating session</h2></section>`;
  try { await loadSnapshot(); } catch (error) { state.token = ""; sessionStorage.removeItem("sparse-dashboard-token"); showBanner(error.message, "bad"); authDialog.showModal(); }
});
document.querySelector("#confirm-cancel").addEventListener("click", () => {
  confirmDialog.close();
  confirmationResolver?.resolve(null);
  confirmationResolver = null;
});
confirmDialog.addEventListener("cancel", () => {
  confirmationResolver?.resolve(null);
  confirmationResolver = null;
});
document.querySelector("#confirm-form").addEventListener("submit", (event) => {
  event.preventDefault();
  if (!confirmationResolver) return;
  const input = document.querySelector("#confirm-input");
  const license = document.querySelector("#license-confirm");
  if (input.value !== confirmationResolver.phrase) {
    input.setCustomValidity("The confirmation phrase does not match.");
    input.reportValidity();
    return;
  }
  if (confirmationResolver.needsLicense && !license.checked) {
    license.setCustomValidity("License acknowledgement is required.");
    license.reportValidity();
    return;
  }
  const resolver = confirmationResolver.resolve;
  confirmationResolver = null;
  confirmDialog.close();
  resolver({ phrase: input.value, licenseAcknowledged: license.checked });
});
window.addEventListener("hashchange", () => {
  const route = routeFromHash(location.hash);
  state.view = route.view;
  if (route.view === "workspace") state.activeDraftId = route.id;
  render();
});

setInterval(pollEvents, 1500);
setInterval(() => { if (state.connected && Date.now() - state.lastSuccess > 10000) { state.stale = true; updateChrome(); } }, 1000);

updateChrome();
loadSnapshot().catch((error) => { if (error.status !== 401) showBanner(error.message, "bad"); });
