import test from "node:test";
import assert from "node:assert/strict";
import {
  buildTaskPayload,
  buildFollowUpPayload,
  classifyUpload,
  createTaskDraft,
  suggestTask,
  taskPlan,
} from "../task-workspace.js";

const developer = {
  id: "developer-workstation",
  kind: "developer",
  title: "Developer workstation",
  input_kinds: ["repository", "source_file", "log", "screenshot", "patch"],
  task_ui: {
    action: "Work on code",
    default_path_kind: "repository",
    default_paste_kind: "source_file",
    category: "everyday",
  },
  stages: [
    { id: "primary", role: "repair", capability: "coding", condition: "always", available: true, available_endpoints: ["qwen"] },
    { id: "challenge", role: "critique", capability: "plan_review", condition: "failed_or_ambiguous", available: false, available_endpoints: [] },
  ],
  tools: [{ id: "test", label: "Tests", mode: "read_only", available: true }],
};

const documentTask = {
  id: "private-document-analysis",
  kind: "document",
  input_kinds: ["pdf", "page_image", "text_document"],
  task_ui: {
    default_path_kind: "pdf",
    default_paste_kind: "text_document",
    default_ocr_mode: "fast",
    ocr_modes: [
      { id: "fast", label: "Fast" },
      { id: "complex", label: "Complex" },
    ],
    category: "everyday",
  },
  stages: [{
    id: "extract",
    role: "ocr",
    capability: "document_ocr",
    condition: "visual",
    available: true,
    available_endpoints: ["pp-ocrv6-medium", "paddleocr-vl-1.6"],
    available_by_ocr_mode: {
      fast: ["pp-ocrv6-medium"],
      complex: ["paddleocr-vl-1.6"],
    },
  }],
  tools: [],
};

test("suggests a task from intent and common file extensions", () => {
  assert.equal(suggestTask("repair the failing Python function", [], [documentTask, developer]).templateId, developer.id);
  assert.equal(suggestTask("extract the invoice table", ["invoice.pdf"], [developer, documentTask]).templateId, documentTask.id);
  assert.equal(suggestTask("hello", [], [developer, documentTask]).status, "no_match");
});

test("reports extension-only matches and equal-score ambiguity", () => {
  const extensionMatch = suggestTask("", ["scan.pdf"], [developer, documentTask]);
  assert.equal(extensionMatch.status, "matched");
  assert.equal(extensionMatch.templateId, documentTask.id);
  assert.ok(extensionMatch.scores[0].score >= 5);
  assert.equal(
    suggestTask("review this code document", [], [developer, documentTask]).status,
    "ambiguous",
  );
});

test("distinguishes incident analysis from troubleshooting", () => {
  const incident = { id: "incident", kind: "incident", task_ui: { category: "everyday" } };
  const troubleshooting = { id: "troubleshooting", kind: "troubleshooting", task_ui: { category: "everyday" } };
  assert.equal(suggestTask("build an outage timeline", [], [troubleshooting, incident]).templateId, "incident");
  assert.equal(suggestTask("troubleshoot this configuration error", [], [incident, troubleshooting]).templateId, "troubleshooting");
});

test("builds a bounded guided developer payload", () => {
  const draft = createTaskDraft(developer, {
    prompt: "Find why the tests fail",
    paths: "E:\\projects\\demo",
    pastedContent: "AssertionError",
    tools: ["test"],
    flags: ["ambiguous"],
  });
  const payload = buildTaskPayload(developer, draft, [{ kind: "screenshot", content_base64: "AA==" }], false);
  assert.equal(payload.template_id, developer.id);
  assert.equal(payload.plan_only, false);
  assert.equal(payload.working_directory, "E:\\projects\\demo");
  assert.deepEqual(payload.tools, ["test"]);
  assert.deepEqual(payload.flags, ["ambiguous"]);
  assert.deepEqual(payload.inputs.map((value) => value.kind), ["repository", "source_file", "screenshot"]);
});

test("builds an auditable follow-up without resubmitting local sources", () => {
  const draft = createTaskDraft(developer, {
    title: "Failing tests",
    prompt: "Find why the tests fail",
    paths: "E:\\projects\\demo",
    pastedContent: "AssertionError",
    tools: ["test"],
    runId: "usecase-first",
    runIds: ["usecase-first"],
  });
  const payload = buildFollowUpPayload(
    developer,
    draft,
    "Can you narrow that down to one function?",
    "usecase-first",
  );
  assert.equal(payload.parent_run_id, "usecase-first");
  assert.equal(payload.prompt, "Can you narrow that down to one function?");
  assert.deepEqual(payload.inputs, []);
  assert.deepEqual(payload.tools, []);
  assert.equal(payload.working_directory, undefined);
});

test("validates batch JSON before submission", () => {
  const batch = {
    id: "structured-batch",
    kind: "batch",
    input_kinds: ["batch_manifest", "text_record"],
    task_ui: { default_path_kind: "batch_manifest", default_paste_kind: "text_record" },
    stages: [],
    tools: [],
  };
  const empty = createTaskDraft(batch, { prompt: "Validate values" });
  assert.throws(() => buildTaskPayload(batch, empty), /at least one batch record/);
  const draft = createTaskDraft(batch, { prompt: "Validate values", recordsJson: "not-json" });
  assert.throws(() => buildTaskPayload(batch, draft), /Batch records must be valid JSON/);
});

test("persists and submits the selected OCR mode", () => {
  const fast = createTaskDraft(documentTask, { prompt: "Read this scan" });
  assert.equal(fast.ocrMode, "fast");
  assert.equal(buildTaskPayload(documentTask, fast).ocr_mode, "fast");

  const complex = createTaskDraft(documentTask, {
    prompt: "Parse this report",
    ocrMode: "complex",
  });
  assert.equal(buildTaskPayload(documentTask, complex).ocr_mode, "complex");
  const plan = taskPlan(documentTask, complex, 3);
  assert.equal(plan.steps[1].title, "Complex OCR");
  assert.deepEqual(plan.steps[1].endpoints, ["paddleocr-vl-1.6"]);

  complex.ocrMode = "unsupported";
  assert.throws(
    () => buildTaskPayload(documentTask, complex),
    /Choose a supported OCR mode/,
  );
});

test("classifies uploads only into admitted task input kinds", () => {
  assert.equal(classifyUpload(documentTask, { name: "scan.png", type: "image/png" }).kind, "page_image");
  assert.equal(classifyUpload(documentTask, { name: "report.pdf", type: "application/pdf" }).kind, "pdf");
  assert.equal(classifyUpload(developer, { name: "fix.patch", type: "text/plain" }).kind, "patch");
  assert.equal(classifyUpload(documentTask, { name: "archive.bin", type: "application/octet-stream" }).status, "unsupported");
  assert.equal(classifyUpload(developer, { name: "recording.mp3", type: "audio/mpeg" }).status, "unsupported");
});

test("classifies media and record formats without silent fallback", () => {
  const meeting = { input_kinds: ["audio", "video", "transcript"] };
  const batch = { input_kinds: ["batch_manifest", "text_record"] };
  assert.equal(classifyUpload(meeting, { name: "call.mp3", type: "audio/mpeg" }).kind, "audio");
  assert.equal(classifyUpload(meeting, { name: "call.mp4", type: "video/mp4" }).kind, "video");
  assert.equal(classifyUpload(batch, { name: "rows.csv", type: "text/csv" }).kind, "batch_manifest");
  assert.equal(classifyUpload(batch, { name: "rows.jsonl", type: "application/jsonl" }).kind, "batch_manifest");
});

test("requires correction when generic text has multiple admitted meanings", () => {
  const mixed = { input_kinds: ["source_file", "diagnostic"] };
  const result = classifyUpload(mixed, { name: "notes.txt", type: "text/plain" });
  assert.equal(result.status, "ambiguous");
  assert.deepEqual(result.options, ["source_file", "diagnostic"]);
  assert.equal(result.kind, null);
});

test("renders a finite human-readable plan with explicit gaps", () => {
  const draft = createTaskDraft(developer, { paths: ".", prompt: "Review this", tools: ["test"] });
  const plan = taskPlan(developer, draft, 3);
  assert.deepEqual(plan.steps.map((value) => value.id), ["ingest", "primary", "challenge", "tool-test", "verify"]);
  assert.deepEqual(plan.unavailable.map((value) => value.id), ["challenge"]);
  assert.equal(plan.maximumModelCalls, 2);
});
