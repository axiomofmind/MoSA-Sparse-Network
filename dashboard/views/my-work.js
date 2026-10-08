import { escapeHtml } from "../model.js";
import { taskState } from "../task-state.js";
import { taskCopy } from "../task-workspace.js";

export function renderMyWorkView({ drafts, entries, templates, canCancel, actionButton, emptyState }) {
  const draftCards = drafts.filter((draft) => !draft.runId).map((draft) => {
    const template = templates.find((value) => value.id === draft.templateId);
    return `<article class="work-card"><div><span class="status-label neutral">Draft</span><h2>${escapeHtml(draft.title || taskCopy(template || {}).action || "Untitled task")}</h2><p>${escapeHtml(draft.prompt || "No request entered yet")}</p></div><div class="row-actions"><button class="button primary small" data-resume-draft="${escapeHtml(draft.id)}">Continue</button><button class="button secondary small" data-delete-draft="${escapeHtml(draft.id)}">Delete draft</button></div></article>`;
  }).join("");
  const runCards = entries.map((entry) => {
    const run = entry.result || entry;
    const status = taskState(entry);
    return `<article class="work-card"><div><span class="status-label ${escapeHtml(status.tone)}">${escapeHtml(status.label)}</span><h2>${escapeHtml(run.title || entry.title || "Task")}</h2><p>${escapeHtml(run.presentation?.deliverable?.text || entry.error_type || "Task details are preserved.")}</p></div><div class="row-actions"><button class="button secondary small" data-open-run="${escapeHtml(entry.id)}">Open</button>${["queued", "preparing_models", "running", "cancelling"].includes(entry.state) && canCancel ? actionButton("Cancel", `data-cancel-usecase="${escapeHtml(entry.id)}"`, "danger") : ""}</div></article>`;
  }).join("");
  return `<section class="page-heading"><p class="eyebrow">MY WORK</p><h2>Drafts and recent tasks</h2><p>Resume the exact task without creating a duplicate run.</p></section><section class="work-list">${draftCards}${runCards || (!draftCards ? emptyState("No work yet", "Start a task from Home.") : "")}</section>`;
}
