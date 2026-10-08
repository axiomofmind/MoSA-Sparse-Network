import test from "node:test";
import assert from "node:assert/strict";

import { issuesFromError, validateTaskDraft } from "../task-errors.js";
import { documentTaskFixture } from "./fixtures/document-journey.js";

test("document table validation identifies the prompt and exact incomplete field", () => {
  const issues = validateTaskDraft(documentTaskFixture, {
    prompt: "",
    files: [],
    outputIntent: "table",
    documentFields: [{ id: "total", label: "", type: "number", required: true }],
  });
  assert.match(issues.fields.prompt, /Describe/);
  assert.equal(issues.fields["documentField:0"], "Give this field a name.");
});

test("submission errors map to JSON and approved-path fields", () => {
  assert.match(issuesFromError(new Error("Batch records must be valid JSON")).fields.recordsJson, /valid JSON/);
  assert.match(issuesFromError(new Error("source path is outside the approved roots")).fields.paths, /approved roots/);
});

test("source-specific server error preserves other valid draft inputs", () => {
  const draft = { prompt: "Extract totals", files: [{ name: "invoice.pdf" }] };
  const issues = issuesFromError(new Error("invoice.pdf could not be opened"), draft);
  assert.equal(issues.sources[0], "invoice.pdf could not be opened");
  assert.equal(draft.prompt, "Extract totals");
});
