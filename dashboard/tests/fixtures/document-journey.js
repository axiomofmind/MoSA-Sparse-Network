export const DOCUMENT_JOURNEY_STEPS = [
  "attach_document",
  "choose_output",
  "check_readiness",
  "run",
  "read_cited_result",
  "inspect_source",
  "correct_field",
  "export_reviewed_version",
  "run_again",
];

const basePresentation = {
  schema: "sparse-network-result-view.v1",
  version: { id: "run-document:v1", number: 1, parent_id: null, immutable_original: true, corrections: [] },
  deliverable: { type: "answer", text: "The invoice total is $42.00." },
  sources: [{ id: "artifact-page-1", name: "invoice.pdf", artifact_id: "artifact-page-1", page: 1 }],
  citations: [{ claim_id: "total", label: "invoice.pdf", artifact_id: "artifact-page-1", page: 1, supported: true }],
  checks: [{ name: "source_references", passed: true, status: "recorded", version_id: "run-document:v1" }],
  limitations: [],
  next_actions: ["inspect_sources", "run_again"],
};

export const documentJourneyFixtures = {
  success: {
    id: "run-document",
    kind: "document",
    state: "accepted",
    title: "Invoice review",
    presentation: basePresentation,
  },
  unsupportedInput: {
    file: { name: "archive.bin", type: "application/octet-stream", size: 12, arrayBuffer() {} },
    expectedMessage: "This file type could not be identified safely",
  },
  missingRequiredModel: {
    state: "blocked",
    required_capabilities: [{ capability: "document_ocr", state: "missing" }],
    optional_capabilities: [],
    blocking_reasons: ["Fast OCR is not installed."],
    alternatives: [{ action: "add_transcript", label: "Add extracted text instead" }],
  },
  optionalVerifierUnavailable: {
    state: "ready",
    required_capabilities: [{ capability: "document_ocr", state: "ready" }],
    optional_capabilities: [{ capability: "vision_critic", state: "unavailable" }],
    blocking_reasons: [],
  },
  unresolvedCitation: {
    id: "run-unresolved",
    kind: "document",
    state: "needs_review",
    presentation: {
      ...basePresentation,
      version: { ...basePresentation.version, id: "run-unresolved:v1" },
      citations: [{ claim_id: "total", label: "invoice.pdf", artifact_id: "artifact-page-1", page: 1, supported: false }],
      checks: [{ name: "source_references", passed: false, failures: ["source reference missing"], artifact_id: "artifact-page-1" }],
      limitations: ["One claim-level source reference could not be resolved."],
    },
  },
  partialExtraction: {
    id: "run-partial",
    kind: "document",
    state: "needs_review",
    presentation: {
      ...basePresentation,
      version: { ...basePresentation.version, id: "run-partial:v1" },
      deliverable: {
        type: "table",
        text: "One required value is missing.",
        columns: [
          { id: "invoice_number", label: "Invoice number", required: true },
          { id: "total", label: "Total", required: true },
        ],
        rows: [{ invoice_number: "INV-42", total: "" }],
      },
      checks: [{ name: "required_fields", passed: false, failures: ["row 1: total"] }],
      limitations: ["A required value was not extracted."],
    },
  },
  cancellation: {
    id: "job-cancelled",
    kind: "document",
    state: "cancelled",
    title: "Cancelled invoice review",
    error_type: "CancelledError",
  },
};

export const documentTaskFixture = {
  id: "private-document-analysis",
  kind: "document",
  title: "Private document analysis",
  input_kinds: ["pdf", "page_image", "text_document"],
  task_ui: { default_path_kind: "pdf", default_paste_kind: "text_document" },
};
