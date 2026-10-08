import { classifyUpload } from "./task-workspace.js";

export function emptyTaskIssues() {
  return { global: [], fields: {}, sources: {} };
}

export function hasTaskIssues(issues) {
  return Boolean(
    issues?.global?.length
    || Object.keys(issues?.fields || {}).length
    || Object.keys(issues?.sources || {}).length,
  );
}

function addGlobal(issues, message) {
  const normalized = String(message || "The task could not start.");
  if (!issues.global.includes(normalized)) issues.global.push(normalized);
}

export function validateTaskDraft(template, draft) {
  const issues = emptyTaskIssues();
  if (!String(draft?.prompt || "").trim()) {
    issues.fields.prompt = "Describe what you want this task to accomplish.";
  }
  (draft?.files || []).forEach((file, index) => {
    const classification = classifyUpload(template, file);
    const selected = draft.fileKinds?.[index];
    if (file.needsReselection || typeof file.arrayBuffer !== "function") {
      issues.sources[index] = `${file.name} must be selected again before this task can start.`;
    } else if ((!classification.kind || classification.status !== "matched") && !selected) {
      issues.sources[index] = `${file.name}: ${classification.reason}. Choose a supported source type.`;
    } else if (selected && !(template.input_kinds || []).includes(selected)) {
      issues.sources[index] = `${file.name}: the selected source type is not supported by this task.`;
    }
  });
  if (template?.kind === "document" && draft?.outputIntent === "table") {
    if (!(draft.documentFields || []).length) {
      issues.fields.documentFields = "Add at least one field for the extracted table.";
    } else {
      const unnamed = draft.documentFields.findIndex((field) => !String(field.label || "").trim());
      if (unnamed >= 0) issues.fields[`documentField:${unnamed}`] = "Give this field a name.";
    }
  }
  if (issues.fields.prompt) addGlobal(issues, issues.fields.prompt);
  if (Object.keys(issues.sources).length) addGlobal(issues, "One or more sources need attention.");
  if (Object.keys(issues.fields).some((key) => key.startsWith("documentField"))) {
    addGlobal(issues, "The extracted table has an incomplete field definition.");
  }
  if (issues.fields.documentFields) addGlobal(issues, issues.fields.documentFields);
  return issues;
}

export function issuesFromError(error, draft = {}) {
  const issues = emptyTaskIssues();
  const message = String(error?.message || error || "The task could not start.");
  const sourceIndex = (draft.files || []).findIndex((file) => message.includes(file.name));
  if (sourceIndex >= 0) issues.sources[sourceIndex] = message;
  else if (/describe what you want|prompt/i.test(message)) issues.fields.prompt = message;
  else if (/batch records/i.test(message)) issues.fields.recordsJson = message;
  else if (/record schema/i.test(message)) issues.fields.schemaJson = message;
  else if (/ocr mode/i.test(message)) issues.fields.ocrMode = message;
  else if (/document field|extracted table/i.test(message)) issues.fields.documentFields = message;
  else if (/approved|allowlist|source path|path is not/i.test(message)) issues.fields.paths = message;
  addGlobal(issues, message);
  return issues;
}

export function mergeTaskIssues(...values) {
  const merged = emptyTaskIssues();
  for (const issues of values) {
    for (const message of issues?.global || []) addGlobal(merged, message);
    Object.assign(merged.fields, issues?.fields || {});
    Object.assign(merged.sources, issues?.sources || {});
  }
  return merged;
}
