import { escapeHtml } from "../model.js";
import { taskCopy } from "../task-workspace.js";

export const STARTER_EXAMPLES = {
  developer: { request: "Find why these tests fail.", result: "Diagnosis, proposed changes, and test evidence." },
  document: { request: "Extract the totals and dates from these invoices.", result: "An answer or table with source references." },
  incident: { request: "Build a timeline of this outage from these logs.", result: "Timeline, likely causes, impact, and next steps." },
  batch: { request: "Extract these fields and flag invalid records.", result: "Structured records, exceptions, and an export." },
  research: { request: "Where do these reports agree and disagree?", result: "A source-linked synthesis with unresolved questions." },
  troubleshooting: { request: "This service will not start. What should I check?", result: "Ranked hypotheses and reversible diagnostic steps." },
  meeting: { request: "Capture decisions, owners, and follow-up actions.", result: "Summary, decisions, and editable action items." },
};

export function renderHomeView({ state, templates, drafts }) {
  const ordinary = templates.filter((template) => taskCopy(template).category !== "advanced");
  const recentDrafts = drafts.slice(0, 3);
  const files = state.homeFiles.map((file) => `<li><span>${escapeHtml(file.name)}</span><button type="button" data-remove-home-file="${escapeHtml(file.name)}">Remove</button></li>`).join("");
  const suggestion = state.taskSuggestion;
  const suggestionChoice = suggestion?.status === "ambiguous" ? `<div class="intent-question"><strong>Which result do you need?</strong>${suggestion.scores.slice(0, 2).map((value) => {
    const template = templates.find((item) => item.id === value.templateId);
    return template ? `<button type="button" class="button secondary" data-home-template="${escapeHtml(template.id)}">${escapeHtml(taskCopy(template).action)}</button>` : "";
  }).join("")}</div>` : suggestion?.status === "no_match" ? `<div class="intent-question"><strong>Choose a task below so Sparse knows what result to prepare.</strong></div>` : "";
  const cards = ordinary.map((template) => {
    const copy = taskCopy(template);
    const example = STARTER_EXAMPLES[template.kind] || { request: template.description, result: copy.summary };
    return `<button class="starter-card" data-home-template="${escapeHtml(template.id)}"><span class="starter-icon" aria-hidden="true">${escapeHtml((copy.action || template.kind).slice(0, 2).toUpperCase())}</span><span><strong>${escapeHtml(copy.action)}</strong><small>“${escapeHtml(example.request)}”</small><em>${escapeHtml(example.result)}</em></span></button>`;
  }).join("");
  const continueWorking = recentDrafts.length ? `<section class="continue-working"><div class="home-section-heading"><div><p class="eyebrow">CONTINUE WORKING</p><h2>Pick up where you left off</h2></div><a href="#work">View all</a></div><div class="continue-grid">${recentDrafts.map((draft) => `<button data-resume-draft="${escapeHtml(draft.id)}"><strong>${escapeHtml(draft.title || "Untitled task")}</strong><span>${escapeHtml(draft.prompt || "Draft")}</span><small>${draft.files?.some((file) => file.needsReselection) ? "Some files need to be selected again" : "Draft saved locally"}</small></button>`).join("")}</div></section>` : "";
  return `<section class="home-hero"><p class="eyebrow">PRIVATE LOCAL AI</p><h2>What would you like to get done?</h2><form id="home-task-form"><label class="sr-only" for="home-intent">Describe the result you need</label><textarea id="home-intent" rows="4" placeholder="Describe the outcome you need…">${escapeHtml(state.taskIntent)}</textarea><div class="home-composer-actions"><label class="button secondary file-button">Add files<input id="home-files" type="file" multiple></label><button class="button primary" type="submit">Continue</button></div>${files ? `<ul class="home-file-list">${files}</ul>` : ""}${suggestionChoice}</form></section>${continueWorking}<section class="starters"><div class="home-section-heading"><div><p class="eyebrow">TASK STARTERS</p><h2>Start with a useful outcome</h2></div></div><div class="starter-grid">${cards}</div></section>`;
}
