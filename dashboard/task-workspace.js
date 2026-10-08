const FALLBACK_COPY = {
  developer: { action: "Work on code", summary: "Inspect and verify a local project." },
  document: { action: "Analyze documents", summary: "Extract and verify source-linked information." },
  incident: { action: "Investigate an incident", summary: "Build an evidence and action timeline." },
  batch: { action: "Process a batch", summary: "Validate many identity-preserving records." },
  research: { action: "Research local sources", summary: "Produce a claim-level evidence synthesis." },
  troubleshooting: { action: "Troubleshoot a problem", summary: "Propose reversible diagnostic steps." },
  meeting: { action: "Analyze a meeting", summary: "Extract decisions and evidence-linked actions." },
  experiment: { action: "Evaluate routing and models", summary: "Compare frozen model and routing baselines." },
};

const SUGGESTION_TERMS = {
  developer: ["code", "repo", "repository", "function", "bug", "test", "python", "javascript", "typescript", "implement", "refactor"],
  document: ["document", "pdf", "invoice", "contract", "form", "ocr", "extract fields", "table", "page"],
  incident: ["incident", "outage", "service down", "production", "alert", "timeline", "impact", "postmortem"],
  batch: ["batch", "records", "rows", "csv", "jsonl", "many files", "schema", "dead letter"],
  research: ["research", "sources", "corpus", "evidence", "literature", "citations", "compare papers"],
  troubleshooting: ["troubleshoot", "not working", "error", "diagnose", "configuration", "manual", "steps to fix"],
  meeting: ["meeting", "transcript", "recording", "speakers", "action items", "minutes", "audio"],
  experiment: ["benchmark", "evaluate models", "routing", "antidoom", "regression", "quantization", "baseline"],
};

const EXTENSION_HINTS = {
  developer: [".py", ".js", ".ts", ".tsx", ".jsx", ".patch", ".diff"],
  document: [".pdf", ".doc", ".docx", ".odt"],
  batch: [".csv", ".jsonl"],
  meeting: [".wav", ".mp3", ".m4a", ".flac", ".mp4", ".webm"],
};

function normalizedLines(value) {
  return String(value || "").split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
}

function parseJson(value, label, fallback) {
  const text = String(value || "").trim();
  if (!text) return fallback;
  try {
    return JSON.parse(text);
  } catch (error) {
    throw new Error(`${label} must be valid JSON: ${error.message}`);
  }
}

export function taskCopy(template) {
  return { ...(FALLBACK_COPY[template?.kind] || {}), ...(template?.task_ui || {}) };
}

export function createTaskDraft(template, previous = {}) {
  const copy = taskCopy(template);
  const kind = template?.kind || "developer";
  const runIds = Array.isArray(previous.runIds)
    ? [...previous.runIds]
    : previous.runId ? [previous.runId] : [];
  const turns = Array.isArray(previous.turns) ? [...previous.turns] : [];
  return {
    id: previous.id || "",
    templateId: template?.id || "",
    title: previous.title || "",
    prompt: previous.prompt || "",
    paths: previous.paths || "",
    pathKind: previous.pathKind || copy.default_path_kind || template?.input_kinds?.[0] || "",
    pastedContent: previous.pastedContent || "",
    pasteKind: previous.pasteKind || copy.default_paste_kind || template?.input_kinds?.[0] || "",
    tools: Array.isArray(previous.tools) ? [...previous.tools] : [],
    flags: Array.isArray(previous.flags) ? [...previous.flags] : [],
    highRisk: Boolean(previous.highRisk),
    humanReview: previous.humanReview !== false,
    indexSources: Boolean(previous.indexSources),
    topK: Number(previous.topK || 5),
    language: previous.language || "auto",
    hotwords: previous.hotwords || "",
    maximumRetries: Number(previous.maximumRetries ?? 1),
    recordsJson: previous.recordsJson || (kind === "batch" ? "[]" : ""),
    schemaJson: previous.schemaJson || (kind === "batch" ? '{\n  "required": ["value"],\n  "properties": {"value": {"type": "string"}}\n}' : ""),
    datasetRevision: previous.datasetRevision || "",
    promptRevision: previous.promptRevision || "",
    baselinesJson: previous.baselinesJson || "[]",
    files: Array.isArray(previous.files) ? [...previous.files] : [],
    fileKinds: previous.fileKinds && typeof previous.fileKinds === "object" ? { ...previous.fileKinds } : {},
    outputIntent: previous.outputIntent || "answer",
    ocrMode: previous.ocrMode || template?.task_ui?.default_ocr_mode || "fast",
    documentFields: Array.isArray(previous.documentFields) ? [...previous.documentFields] : [],
    runId: previous.runId || null,
    runIds,
    turns,
    followUp: previous.followUp || "",
    followUpNeedsPreparation: Boolean(previous.followUpNeedsPreparation),
  };
}

export function suggestTask(text, fileNames = [], templates = []) {
  const haystack = `${String(text || "")} ${fileNames.join(" ")}`.toLowerCase();
  const scores = new Map();
  for (const template of templates) {
    let score = 0;
    for (const term of SUGGESTION_TERMS[template.kind] || []) {
      if (haystack.includes(term)) score += term.includes(" ") ? 4 : 2;
    }
    for (const extension of EXTENSION_HINTS[template.kind] || []) {
      if (fileNames.some((name) => name.toLowerCase().endsWith(extension))) score += 5;
    }
    if (template.task_ui?.category === "advanced") score -= 1;
    scores.set(template.id, score);
  }
  const ordinary = templates.filter((template) => template.task_ui?.category !== "advanced");
  const ranked = [...ordinary].sort((left, right) => {
    const difference = (scores.get(right.id) || 0) - (scores.get(left.id) || 0);
    if (difference) return difference;
    const leftAdvanced = left.task_ui?.category === "advanced" ? 1 : 0;
    const rightAdvanced = right.task_ui?.category === "advanced" ? 1 : 0;
    return leftAdvanced - rightAdvanced;
  });
  if (!ranked.length) return { status: "no_match", templateId: null, scores: [] };
  const result = ranked.map((template) => ({
    templateId: template.id,
    score: scores.get(template.id) || 0,
  }));
  const first = result[0];
  const second = result[1];
  if (first.score <= 0) return { status: "no_match", templateId: null, scores: result };
  if (second && first.score === second.score) {
    return { status: "ambiguous", templateId: first.templateId, scores: result };
  }
  return { status: "matched", templateId: first.templateId, scores: result };
}

export function classifyUpload(template, file) {
  const name = String(file?.name || "").toLowerCase();
  const type = String(file?.type || "").toLowerCase();
  const admitted = new Set(template?.input_kinds || []);
  const choose = (...values) => values.find((value) => admitted.has(value));
  const matched = (kind, reason) => ({ status: "matched", kind, confidence: "high", reason });
  const ambiguous = (options, reason) => ({
    status: "ambiguous", kind: null, confidence: "low", reason, options,
  });
  const unsupported = (reason) => ({
    status: "unsupported", kind: null, confidence: "none", reason, options: [...admitted],
  });
  if (type.startsWith("image/") || /\.(png|jpe?g|webp|gif)$/.test(name)) {
    const kind = choose("screenshot", "page_image", "image", "figure", "image_record");
    return kind ? matched(kind, "Recognized image") : unsupported("Images are not supported by this task");
  }
  if (type.startsWith("audio/") || /\.(wav|mp3|m4a|flac|ogg)$/.test(name)) {
    const kind = choose("audio");
    return kind ? matched(kind, "Recognized audio") : unsupported("Audio is not supported by this task");
  }
  if (type.startsWith("video/") || /\.(mp4|webm|mov)$/.test(name)) {
    const kind = choose("video");
    return kind ? matched(kind, "Recognized video") : unsupported("Video is not supported by this task");
  }
  if (type === "application/pdf" || name.endsWith(".pdf")) {
    const kind = choose("pdf");
    return kind ? matched(kind, "Recognized PDF") : unsupported("PDF files are not supported by this task");
  }
  const mappings = [
    [/\.(log|out)$/, ["log", "diagnostic", "incident_note"], "Recognized log"],
    [/\.(patch|diff)$/, ["patch", "source_file"], "Recognized patch"],
    [/\.(csv|jsonl)$/, ["batch_manifest", "text_record", "dataset"], "Recognized records"],
    [/\.(txt|md)$/, ["transcript", "text_document", "note", "source_file", "diagnostic"], "Recognized text"],
    [/\.(py|js|ts|tsx|jsx|json|ya?ml|toml|ini)$/, ["source_file", "configuration", "text_document"], "Recognized text source"],
    [/\.(doc|docx|odt)$/, ["office_document"], "Recognized office document"],
  ];
  for (const [pattern, candidates, reason] of mappings) {
    if (pattern.test(name)) {
      const options = candidates.filter((value) => admitted.has(value));
      if (options.length > 1 && /\.(txt|md)$/.test(name)) {
        return ambiguous(options, "This text could be used in more than one way");
      }
      const kind = options[0];
      return kind ? matched(kind, reason) : unsupported(`${reason.replace("Recognized", "This")} is not supported by this task`);
    }
  }
  return unsupported("This file type could not be identified safely");
}

export function buildTaskPayload(template, draft, uploadedInputs = [], planOnly = false) {
  if (!template?.id) throw new Error("Select a task before submitting.");
  const prompt = String(draft.prompt || "").trim();
  if (!prompt) throw new Error("Describe what you want this task to accomplish.");
  const admitted = new Set(template.input_kinds || []);
  if (!admitted.has(draft.pathKind) || !admitted.has(draft.pasteKind)) {
    throw new Error("The selected source type is not admitted by this task.");
  }
  const inputs = normalizedLines(draft.paths).map((path) => ({
    kind: draft.pathKind,
    path,
    ...(draft.pathKind === "pdf" ? { render_pages: true } : {}),
  }));
  if (String(draft.pastedContent || "").trim()) {
    inputs.push({ kind: draft.pasteKind, content: String(draft.pastedContent).trim() });
  }
  for (const input of uploadedInputs) {
    if (!admitted.has(input.kind)) throw new Error(`Attachment type is not admitted: ${input.kind}`);
    inputs.push(input);
  }
  const copy = taskCopy(template);
  const payload = {
    template_id: template.id,
    title: String(draft.title || "").trim() || copy.action || template.title,
    prompt,
    inputs,
    tools: [...(draft.tools || [])],
    flags: [...(draft.flags || [])],
    claims: [],
    plan_only: planOnly,
    output_intent: draft.outputIntent || "answer",
  };
  if (template.kind === "document" && draft.outputIntent === "table") {
    payload.document_fields = (draft.documentFields || []).map((field, index) => ({
      id: String(field.id || `field-${index + 1}`),
      label: String(field.label || `Field ${index + 1}`),
      type: String(field.type || "string"),
      required: Boolean(field.required),
    }));
  }
  if (template.kind === "document") {
    const allowedOcrModes = new Set(
      (template.task_ui?.ocr_modes || [{ id: "fast" }, { id: "complex" }])
        .map((value) => String(value.id || "")),
    );
    const ocrMode = String(draft.ocrMode || "fast");
    if (!allowedOcrModes.has(ocrMode)) throw new Error("Choose a supported OCR mode.");
    payload.ocr_mode = ocrMode;
  }
  const firstDirectory = normalizedLines(draft.paths)[0];
  if (
    template.kind === "developer"
    && firstDirectory
    && ["repository", "working_directory"].includes(draft.pathKind)
  ) payload.working_directory = firstDirectory;
  if (["document", "incident", "troubleshooting"].includes(template.kind)) {
    payload.high_risk = Boolean(draft.highRisk);
    payload.human_review_required = Boolean(draft.highRisk && draft.humanReview);
  }
  if (template.kind === "research") {
    payload.index_sources = Boolean(draft.indexSources);
    payload.query = prompt;
    payload.top_k = Math.min(20, Math.max(1, Number(draft.topK || 5)));
  }
  if (template.kind === "batch") {
    const records = parseJson(draft.recordsJson, "Batch records", []);
    const schema = parseJson(draft.schemaJson, "Record schema", {});
    if (!Array.isArray(records)) throw new Error("Batch records must be a JSON array.");
    if (!records.length) throw new Error("Add at least one batch record.");
    if (!schema || Array.isArray(schema) || typeof schema !== "object") throw new Error("Record schema must be a JSON object.");
    payload.records = records;
    payload.record_schema = schema;
    payload.maximum_retries = Math.min(2, Math.max(0, Number(draft.maximumRetries ?? 1)));
  }
  if (template.kind === "meeting") {
    payload.language = String(draft.language || "auto");
    payload.hotwords = String(draft.hotwords || "").split(",").map((value) => value.trim()).filter(Boolean);
    payload.speaker_corrections = [];
  }
  if (template.kind === "experiment") {
    payload.dataset_revision = String(draft.datasetRevision || "").trim();
    payload.prompt_revision = String(draft.promptRevision || "").trim();
    payload.baselines = parseJson(draft.baselinesJson, "Baselines", []);
    payload.settings = { temperature: 0 };
    payload.seed = 0;
    payload.metrics = {};
  }
  return payload;
}

export function buildFollowUpPayload(template, draft, prompt, parentRunId) {
  const value = String(prompt || "").trim();
  if (!value) throw new Error("Enter a follow-up question or requested change.");
  if (!String(parentRunId || "").startsWith("usecase-")) {
    throw new Error("The previous task result is unavailable for this follow-up.");
  }
  const payload = buildTaskPayload(
    template,
    {
      ...draft,
      prompt: value,
      paths: "",
      pastedContent: "",
      files: [],
      tools: [],
      indexSources: false,
      title: draft.title || taskCopy(template).action,
    },
    [],
    false,
  );
  payload.inputs = [];
  payload.parent_run_id = parentRunId;
  return payload;
}

const CONDITION_COPY = {
  always: "always",
  visual: "when visual evidence is attached",
  failed_or_ambiguous: "when the first answer fails or is ambiguous",
  ambiguous_or_high_risk: "when evidence is ambiguous or high risk",
  exceptional: "for malformed, visual, ambiguous, or conflicting records",
  audio_without_transcript: "when media has no transcript",
  multi_speaker: "when multiple speakers are requested",
};

export function taskPlan(template, draft, maximumStages = 3) {
  const inputs = normalizedLines(draft.paths).length
    + (String(draft.pastedContent || "").trim() ? 1 : 0)
    + (draft.files?.length || 0);
  const steps = [{
    id: "ingest",
    title: inputs ? `Preserve ${inputs} source${inputs === 1 ? "" : "s"}` : "Prepare task context",
    detail: "Inputs become immutable artifacts; local paths must pass the controller allowlist.",
    state: inputs ? "ready" : "optional",
  }];
  if (template?.kind === "research" && draft.indexSources) {
    steps.push({ id: "retrieve", title: "Index and retrieve evidence", detail: `Return up to ${draft.topK || 5} source-linked passages.`, state: "ready" });
  }
  for (const stage of (template?.stages || []).slice(0, maximumStages)) {
    const ocrMode = template?.kind === "document" && stage.id === "extract"
      ? String(draft.ocrMode || "fast")
      : null;
    const availableForMode = ocrMode && stage.available_by_ocr_mode
      ? stage.available_by_ocr_mode[ocrMode] || []
      : stage.available_endpoints || [];
    steps.push({
      id: stage.id,
      title: ocrMode ? `${ocrMode === "complex" ? "Complex" : "Fast"} OCR` : `${stage.role || stage.id}`,
      detail: `${stage.capability || "model work"} · ${CONDITION_COPY[stage.condition] || stage.condition || "as configured"}`,
      state: ocrMode ? (availableForMode.length ? "ready" : "unavailable") : (stage.available ? "ready" : "unavailable"),
      endpoints: availableForMode,
    });
  }
  for (const tool of (template?.tools || []).filter((value) => draft.tools?.includes(value.id))) {
    steps.push({ id: `tool-${tool.id}`, title: tool.label || tool.id, detail: tool.mode === "read_only" ? "Read-only controller tool" : "Requires administrator authorization and rollback information", state: tool.available ? "ready" : "unavailable" });
  }
  steps.push({ id: "verify", title: "Verify and preserve the result", detail: "Check capabilities, tool policy, evidence references, and human-review requirements.", state: "ready" });
  return {
    steps,
    unavailable: steps.filter((step) => step.state === "unavailable"),
    maximumModelCalls: Math.min(maximumStages, template?.stages?.length || 0),
  };
}
