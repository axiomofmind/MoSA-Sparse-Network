import test from "node:test";
import assert from "node:assert/strict";

import { validateTaskDraft } from "../task-errors.js";
import { renderDeliverableView, renderWorkspaceResult, renderWorkspaceThread } from "../views/results.js";
import {
  DOCUMENT_JOURNEY_STEPS,
  documentJourneyFixtures,
  documentTaskFixture,
} from "./fixtures/document-journey.js";

test("document journey fixture freezes the complete reviewed-result path", () => {
  assert.deepEqual(DOCUMENT_JOURNEY_STEPS, [
    "attach_document", "choose_output", "check_readiness", "run",
    "read_cited_result", "inspect_source", "correct_field",
    "export_reviewed_version", "run_again",
  ]);
  assert.deepEqual(Object.keys(documentJourneyFixtures), [
    "success", "unsupportedInput", "missingRequiredModel",
    "optionalVerifierUnavailable", "unresolvedCitation", "partialExtraction",
    "cancellation",
  ]);
});

test("successful document result leads with the answer and opens cited evidence", () => {
  const markup = renderDeliverableView(documentJourneyFixtures.success, { editable: true });
  assert.match(markup, /The invoice total is \$42\.00/);
  assert.match(markup, /data-detail-artifact="artifact-page-1"/);
  assert.match(markup, /Save reviewed version/);
  assert.match(markup, /Export report/);
});

test("unsupported document input is attached to its source", () => {
  const issues = validateTaskDraft(documentTaskFixture, {
    prompt: "Read this file",
    files: [documentJourneyFixtures.unsupportedInput.file],
    fileKinds: {},
    outputIntent: "answer",
  });
  assert.match(issues.sources[0], /could not be identified safely/);
  assert.match(issues.global.join(" "), /sources need attention/i);
});

test("required model gaps block readiness while optional verifier gaps do not", () => {
  assert.equal(documentJourneyFixtures.missingRequiredModel.state, "blocked");
  assert.equal(documentJourneyFixtures.missingRequiredModel.blocking_reasons.length, 1);
  assert.equal(documentJourneyFixtures.optionalVerifierUnavailable.state, "ready");
  assert.equal(documentJourneyFixtures.optionalVerifierUnavailable.blocking_reasons.length, 0);
});

test("unresolved citations link the failed check to evidence", () => {
  const markup = renderDeliverableView(documentJourneyFixtures.unresolvedCitation);
  assert.match(markup, /1 recorded check needs attention/);
  assert.match(markup, /Review source references evidence/);
  assert.match(markup, /data-detail-artifact="artifact-page-1"/);
});

test("partial extraction links the failed check to the affected cell", () => {
  const markup = renderDeliverableView(documentJourneyFixtures.partialExtraction, { editable: true });
  assert.match(markup, /data-focus-result-field="total"/);
  assert.match(markup, /data-row-index="0"/);
  assert.match(markup, /A required value was not extracted/);
});

test("cancelled document keeps a recoverable workspace message", () => {
  const markup = renderWorkspaceResult({
    draft: { title: "Invoice review" },
    job: documentJourneyFixtures.cancellation,
  });
  assert.match(markup, /Task cancelled/);
  assert.match(markup, /request and sources are still here/);
});

test("completed task renders as turns with a contextual follow-up composer", () => {
  const completed = {
    id: "usecase-one",
    state: "completed",
    result: documentJourneyFixtures.success,
  };
  const markup = renderWorkspaceThread({
    draft: {
      title: "Invoice review",
      prompt: "What is the total?",
      runId: completed.id,
      runIds: [completed.id],
      turns: [{ runId: completed.id, prompt: "What is the total?" }],
      followUp: "",
    },
    records: [completed],
    canFollowUp: true,
  });
  assert.match(markup, />You</);
  assert.match(markup, /What is the total/);
  assert.match(markup, />Sparse</);
  assert.match(markup, /Ask a follow-up or refine this result/);
  assert.match(markup, /Uses the previous result/);
});

test("follow-up preparation is an explicit action", () => {
  const completed = {
    id: "usecase-one",
    state: "completed",
    result: documentJourneyFixtures.success,
  };
  const markup = renderWorkspaceThread({
    draft: {
      prompt: "What is the total?",
      runId: completed.id,
      runIds: [completed.id],
      turns: [{ runId: completed.id, prompt: "What is the total?" }],
      followUp: "Check the table too",
      followUpNeedsPreparation: true,
    },
    records: [completed],
    canFollowUp: true,
  });
  assert.match(markup, /Prepare models to continue/);
  assert.match(markup, /Prepare models and send/);
  assert.match(markup, /needs installed models loaded/);
});
