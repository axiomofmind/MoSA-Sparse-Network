import { escapeHtml } from "../model.js";

export function renderWorkspaceView({
  template,
  draft,
  copy,
  readiness,
  issues,
  sourceFields,
  documentFields,
  specificFields,
  planMarkup,
  resultMarkup,
  showRequestForm = true,
}) {
  const readinessCopy = readiness?.state === "blocked" ? readiness.blocking_reasons.join(" ") : readiness?.state === "needs_preparation" ? (readiness.preparation?.permitted ? "Installed models will be prepared when this task starts." : "The installed models for this task need preparation by an authorized role.") : readiness?.state === "ready" ? "Everything required for this request is ready." : "Sparse will check inputs and required capabilities before starting.";
  const issueSummary = issues?.global?.length
    ? `<div class="task-error-summary" role="alert"><strong>Check the highlighted items</strong><ul>${issues.global.map((message) => `<li>${escapeHtml(message)}</li>`).join("")}</ul></div>`
    : "";
  const promptError = issues?.fields?.prompt;
  const submitLabel = readiness?.state === "needs_preparation"
    ? readiness.preparation?.permitted ? "Prepare models and start" : "Preparation required"
    : readiness?.state === "blocked" ? "Resolve required information" : copy.action || "Start task";
  const promptField = `<label class="task-prompt ${promptError ? "has-error" : ""}"><span class="sr-only">${escapeHtml(copy.prompt_label || "What should be done?")}</span><textarea rows="5" data-task-field="prompt" required ${promptError ? 'aria-invalid="true" aria-describedby="prompt-error"' : ""} placeholder="${escapeHtml(copy.prompt_placeholder || template.description)}">${escapeHtml(draft.prompt)}</textarea>${promptError ? `<small class="field-error" id="prompt-error">${escapeHtml(promptError)}</small>` : ""}</label>`;
  const requestForm = showRequestForm ? `<form id="task-workspace-form" class="workspace-request" novalidate>${issueSummary}<div class="workspace-section-heading"><p>${issues?.global?.length ? "Information needed" : readiness?.state === "needs_preparation" ? "Preparation" : "Before starting"}</p><h2>${issues?.global?.length ? "Complete the highlighted information" : readiness?.state === "needs_preparation" ? "Sparse needs to prepare the required models" : "Task details"}</h2></div>${promptError ? promptField : `<details class="edit-request"><summary>Edit the original request</summary>${promptField}</details>`}${sourceFields}${documentFields}<details class="more-options"><summary>Optional task settings</summary>${specificFields}</details><details class="review-steps"><summary>Workflow details</summary><div id="task-plan-body">${planMarkup}</div></details><div class="readiness-summary ${readiness?.state || "unknown"}"><strong>${escapeHtml(readinessCopy)}</strong>${(readiness?.alternatives || []).map((value) => `<button type="button" class="text-button" data-readiness-alternative="${escapeHtml(value.action)}">${escapeHtml(value.label)}</button>`).join("")}</div><div class="task-submit-bar"><span>Your sources and request stay attached to this task.</span><button class="button primary" type="submit" value="run" ${readiness?.state === "blocked" || (readiness?.state === "needs_preparation" && !readiness.preparation?.permitted) ? "disabled" : ""}>${escapeHtml(submitLabel)}</button></div></form>` : "";
  return `<div class="task-workspace-page"><header class="workspace-header"><button class="back-link" type="button" data-go-home>← Home</button><div><p class="eyebrow">${escapeHtml(copy.action || template.title)}</p><input class="workspace-title" aria-label="Task name" data-task-field="title" value="${escapeHtml(draft.title)}" placeholder="Untitled task"></div><button class="text-button" type="button" data-change-task>New task</button></header><div class="workspace-layout"><div class="workspace-main">${resultMarkup}${requestForm}</div><aside class="workspace-sources" aria-labelledby="sources-heading"><div class="workspace-section-heading"><p>Sources</p><h2 id="sources-heading">Attached material</h2></div><p>Sources remain linked to this task and its result.</p>${(draft.files || []).length || draft.paths || draft.pastedContent ? `<ul class="source-summary">${(draft.files || []).map((file) => `<li>${escapeHtml(file.name)}${file.needsReselection ? " — choose again" : ""}</li>`).join("")}${draft.pastedContent ? "<li>Pasted text</li>" : ""}${String(draft.paths || "").split(/\r?\n/).filter(Boolean).map((path) => `<li>${escapeHtml(path)}</li>`).join("")}</ul>` : `<p class="muted">No sources added yet. Follow-ups can use preserved task evidence.</p>`}</aside></div></div>`;
}
