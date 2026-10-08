import { escapeHtml } from "../model.js";
import { taskState } from "../task-state.js";

export function terminalTaskMessage(job) {
  if (job.state === "cancelled") {
    return {
      title: "Task cancelled",
      detail: "Your request and sources are still here. Start again when you are ready.",
    };
  }
  const resourceFailure = job.error_type === "ResourceAdmissionError";
  return {
    title: resourceFailure ? "Not enough resources to prepare the models" : "Task could not finish",
    detail: resourceFailure
      ? "Close other local model work or choose an appropriate hardware profile in Settings, then start this task again."
      : job.error || "Review the highlighted request fields and sources, then try again. Technical details retain the recorded error type.",
  };
}

function failedCheckTarget(check, citations) {
  const failures = Array.isArray(check.failures) ? check.failures : [];
  const fieldMatch = failures.map(String).join(" ").match(/row\s+(\d+)\s*:\s*([^,;]+)/i);
  if (fieldMatch) return { kind: "field", row: Number(fieldMatch[1]) - 1, field: fieldMatch[2].trim() };
  const artifactId = check.artifact_id || check.source_reference || citations[0]?.artifact_id;
  return artifactId ? { kind: "artifact", artifactId } : null;
}

export function renderDeliverableView(run, { editable = false } = {}) {
  const presentation = run?.presentation;
  if (!presentation) return `<div class="result-empty"><h3>No result yet</h3><p>Describe the outcome and add any sources, then start the task.</p></div>`;
  const deliverable = presentation.deliverable || {};
  let content = editable
    ? `<label class="reviewed-text-label">Reviewed result<textarea class="reviewed-text" data-result-text data-original="${escapeHtml(deliverable.text || "")}">${escapeHtml(deliverable.text || "")}</textarea></label>`
    : `<div class="result-text">${escapeHtml(deliverable.text || "No written result was recorded.").replaceAll("\n", "<br>")}</div>`;
  if (deliverable.type === "table") {
    const columns = deliverable.columns || [];
    const rows = deliverable.rows || [];
    content = rows.length ? `<div class="table-scroll"><table class="result-table"><thead><tr>${columns.map((column) => `<th>${escapeHtml(column.label)}</th>`).join("")}</tr></thead><tbody>${rows.map((row, rowIndex) => `<tr>${columns.map((column) => {
      const value = row[column.id] ?? "";
      return `<td>${editable ? `<input data-result-cell data-row-index="${rowIndex}" data-field-id="${escapeHtml(column.id)}" data-original="${escapeHtml(value)}" value="${escapeHtml(value)}" aria-label="${escapeHtml(column.label)}, row ${rowIndex + 1}">` : escapeHtml(value || "—")}</td>`;
    }).join("")}</tr>`).join("")}</tbody></table></div>` : `${content}<p class="result-note">No structured rows were returned.</p>`;
  }
  const cited = presentation.citations || [];
  const references = cited.length ? cited : (presentation.sources || []);
  const citations = references.map((citation) => {
    const lineLocation = citation.line_start ? ` · lines ${escapeHtml(citation.line_start)}${citation.line_end ? `–${escapeHtml(citation.line_end)}` : ""}` : "";
    return `<button class="citation-link" data-detail-artifact="${escapeHtml(citation.artifact_id)}">${escapeHtml(citation.label || citation.name || "Saved source")}${citation.page ? ` · page ${escapeHtml(citation.page)}` : ""}${lineLocation}</button>`;
  }).join("");
  const checks = presentation.checks || [];
  const stale = checks.filter((check) => check.status === "stale");
  const failed = checks.filter((check) => check.status !== "stale" && !check.passed);
  const checkCopy = stale.length
    ? `${stale.length} check${stale.length === 1 ? " is" : "s are"} stale after editing. Rerun checks are listed in details.`
    : failed.length ? `${failed.length} recorded check${failed.length === 1 ? " needs" : "s need"} attention.`
      : checks.length ? "Checks passed." : "";
  const failedLinks = failed.map((check) => {
    const target = failedCheckTarget(check, cited);
    const label = `Review ${String(check.name || "failed check").replaceAll("_", " ")}`;
    if (target?.kind === "field") return `<button type="button" class="text-button" data-focus-result-field="${escapeHtml(target.field)}" data-row-index="${target.row}">${escapeHtml(label)}</button>`;
    if (target?.kind === "artifact") return `<button type="button" class="text-button" data-detail-artifact="${escapeHtml(target.artifactId)}">${escapeHtml(label)} evidence</button>`;
    return `<span>${escapeHtml(label)}</span>`;
  }).join("");
  const batchTraces = run.kind === "batch" ? (run.record_traces || []) : [];
  const failedRecords = batchTraces.filter((record) => record.status === "dead_letter" || (record.errors || []).length);
  const completed = batchTraces.length - failedRecords.length;
  const batchSummary = batchTraces.length ? `<section class="batch-attention ${failedRecords.length ? "has-errors" : ""}"><strong>${completed} completed · ${failedRecords.length} need attention</strong>${failedRecords.length ? `<div class="batch-failure-links">${failedRecords.map((record) => `<button type="button" class="text-button" data-focus-batch-record="${escapeHtml(record.id)}">${escapeHtml(record.id)}: ${escapeHtml((record.errors || []).join(", ") || "Review required")}</button>`).join("")}<button type="button" class="button secondary small" data-replay-usecase="${escapeHtml(run.id)}">Replay failures</button></div><div class="batch-failure-details">${failedRecords.map((record) => `<article tabindex="-1" data-batch-record="${escapeHtml(record.id)}"><strong>${escapeHtml(record.id)}</strong><p>${escapeHtml((record.errors || []).join(", ") || "Review required")}</p></article>`).join("")}</div>` : ""}</section>` : "";
  const actions = run.kind === "document" ? `<div class="result-actions">${editable ? `<button class="button primary" data-save-corrections="${escapeHtml(run.id)}" data-version-id="${escapeHtml(presentation.version?.id)}">Save reviewed version</button>` : ""}<button class="button secondary" data-export-result="${escapeHtml(run.id)}" data-version-id="${escapeHtml(presentation.version?.id)}" data-export-format="report">Export report</button>${deliverable.type === "table" ? `<button class="button secondary" data-export-result="${escapeHtml(run.id)}" data-version-id="${escapeHtml(presentation.version?.id)}" data-export-format="csv">Export CSV</button>` : ""}</div>` : "";
  return `<div class="result-deliverable">${content}${actions}${batchSummary}${citations ? `<div class="result-citations"><h3>${cited.length ? "Citations" : "Source material"}</h3>${citations}</div>` : ""}${checkCopy || failedLinks ? `<div class="check-summary ${failed.length || stale.length ? "attention" : ""}">${escapeHtml(checkCopy)}${failedLinks ? `<div class="check-actions">${failedLinks}</div>` : ""}</div>` : ""}${(presentation.limitations || []).length ? `<ul class="limitations">${presentation.limitations.map((value) => `<li>${escapeHtml(value)}</li>`).join("")}</ul>` : ""}<details class="technical-trace"><summary>How this was produced</summary><pre class="json-view">${escapeHtml(JSON.stringify({ version: presentation.version, checks, routes: run.routes, tools: run.tools, timeline: run.timeline }, null, 2))}</pre></details></div>`;
}

export function renderWorkspaceResult({ draft, job, editable = false }) {
  if (!job) return `<section class="workspace-result" data-workspace-result data-result-signature="draft" aria-labelledby="result-heading"><div class="workspace-section-heading"><p>Result</p><h2 id="result-heading">Your result will appear here</h2></div>${renderDeliverableView(null)}</section>`;
  const run = job.result || job;
  const status = taskState(job);
  const signature = `${job.state || run.state}:${job.progress?.stage_id || "none"}:${job.progress?.current ?? ""}:${run.presentation?.version?.id || "pending"}`;
  const terminal = ["failed", "cancelled"].includes(job.state) ? terminalTaskMessage(job) : null;
  const pending = terminal
    ? `<div class="result-empty"><h3>${escapeHtml(terminal.title)}</h3><p>${escapeHtml(terminal.detail)}</p>${job.error_type ? `<details class="technical-trace"><summary>Technical details</summary><code>${escapeHtml(job.error_type)}</code></details>` : ""}</div>`
    : `<div class="result-empty"><div class="spinner"></div><h3>${escapeHtml(job.progress?.label || status.label)}</h3><p>Working on your request.</p></div>`;
  return `<section class="workspace-result" data-workspace-result data-result-signature="${escapeHtml(signature)}" aria-labelledby="result-heading"><div class="workspace-section-heading"><p>Result</p><h2 id="result-heading">${escapeHtml(run.title || draft.title || "Task result")}</h2><span class="status-label ${escapeHtml(status.tone)}" data-live-task-status="${escapeHtml(job.id)}">${escapeHtml(status.label)}</span></div>${job.result || run.presentation ? renderDeliverableView(run, { editable }) : pending}</section>`;
}

function responseMarkup(record, { editable = false } = {}) {
  if (!record) {
    return `<div class="result-empty compact"><div class="spinner"></div><h3>Starting task</h3><p>Starting your request.</p></div>`;
  }
  const run = record.result || record;
  const status = taskState(record);
  const terminal = ["failed", "cancelled"].includes(record.state) ? terminalTaskMessage(record) : null;
  if (record.result || run.presentation) return renderDeliverableView(run, { editable });
  if (terminal) {
    return `<div class="result-empty compact"><h3>${escapeHtml(terminal.title)}</h3><p>${escapeHtml(terminal.detail)}</p>${record.error_type ? `<details class="technical-trace"><summary>Technical details</summary><code>${escapeHtml(record.error_type)}</code></details>` : ""}</div>`;
  }
  return `<div class="result-empty compact"><div class="spinner"></div><h3>${escapeHtml(record.progress?.label || status.label)}</h3><p>Working on your request.</p></div>`;
}

export function renderWorkspaceThread({
  draft,
  records = [],
  editable = false,
  canFollowUp = false,
}) {
  const turns = Array.isArray(draft.turns) ? draft.turns : [];
  const recordById = new Map(records.filter(Boolean).map((record) => [record.id, record]));
  const runIds = Array.isArray(draft.runIds) && draft.runIds.length
    ? draft.runIds
    : draft.runId ? [draft.runId] : [];
  const ordered = runIds.map((id) => recordById.get(id) || { id, state: "queued" });
  const signatures = ordered.map((record) => {
    const run = record.result || record;
    return `${record.id}:${record.state || run.state}:${record.progress?.stage_id || "none"}:${run.presentation?.version?.id || "pending"}`;
  });
  if (!ordered.length) signatures.push("draft");
  const firstPrompt = String(draft.prompt || "").trim();
  const thread = (ordered.length ? ordered : [null]).map((record, index) => {
    const turn = turns.find((value) => value.runId === record?.id) || turns[index];
    const prompt = turn?.prompt || (index === 0 ? firstPrompt : "Follow-up request");
    const status = record ? taskState(record) : null;
    const isLatest = index === ordered.length - 1;
    return `<article class="conversation-turn"><div class="turn-row user-turn"><div class="turn-label">You</div><div class="turn-content"><p>${escapeHtml(prompt || "Request details needed")}</p></div></div>${record ? `<div class="turn-row sparse-turn"><div class="turn-label">Sparse</div><div class="turn-content"><div class="turn-status"><span>${index === 0 ? "Task result" : `Follow-up ${index}`}</span><span class="status-label ${escapeHtml(status.tone)}" data-live-task-status="${escapeHtml(record.id)}">${escapeHtml(status.label)}</span></div>${responseMarkup(record, { editable: editable && isLatest })}</div></div>` : ""}</article>`;
  }).join("");
  const latest = ordered.at(-1);
  const latestRun = latest?.result || latest;
  const sourceCount = (latestRun?.inputs || []).filter((value) => value.kind !== "prior_result").length;
  const preparation = Boolean(draft.followUpNeedsPreparation);
  const followUp = canFollowUp && latestRun?.presentation
    ? `<form id="task-followup-form" class="followup-composer ${preparation ? "needs-preparation" : ""}"><label for="task-followup"><span>${preparation ? "Prepare models to continue" : "Continue this task"}</span><small>${preparation ? "This follow-up needs installed models loaded before execution." : `Uses the previous result${sourceCount ? ` and ${sourceCount} preserved source${sourceCount === 1 ? "" : "s"}` : ""}.`}</small></label><textarea id="task-followup" rows="3" placeholder="Ask a follow-up or refine this result…">${escapeHtml(draft.followUp || "")}</textarea><div><small>Enter to send · Shift+Enter for a new line</small><button class="button primary" type="submit">${preparation ? "Prepare models and send" : "Send follow-up"}</button></div></form>`
    : "";
  return `<section class="workspace-thread" data-workspace-result data-result-signature="${escapeHtml(signatures.join("|"))}" aria-label="Task conversation">${thread}${followUp}</section>`;
}
