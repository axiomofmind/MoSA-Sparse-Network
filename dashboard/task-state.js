const EXECUTION_STATES = {
  draft: { label: "Draft", tone: "neutral" },
  checking_inputs: { label: "Checking inputs", tone: "neutral" },
  queued: { label: "Queued", tone: "neutral" },
  preparing_models: { label: "Preparing models", tone: "warn" },
  running: { label: "Working", tone: "warn" },
  completed: { label: "Ready", tone: "good" },
  accepted: { label: "Ready", tone: "good" },
  needs_human_review: { label: "Needs review", tone: "warn" },
  rejected: { label: "Needs review", tone: "warn" },
  failed: { label: "Could not finish", tone: "bad" },
  cancelling: { label: "Cancelling", tone: "warn" },
  cancelled: { label: "Cancelled", tone: "neutral" },
};

export function taskState(record = {}) {
  const result = record.result && typeof record.result === "object" ? record.result : record;
  const raw = String(record.state || result.state || "unknown").toLowerCase();
  const execution = EXECUTION_STATES[raw] || { label: "Status unavailable", tone: "neutral" };
  const checks = Array.isArray(result.verification?.checks) ? result.verification.checks : [];
  const verification = !checks.length
    ? "not_recorded"
    : result.verification?.accepted ? "passed" : "needs_review";
  const review = result.state === "needs_human_review" ? "required" : "not_required";
  const display = review === "required" && ["completed", "accepted"].includes(raw)
    ? EXECUTION_STATES.needs_human_review
    : execution;
  return {
    raw,
    label: display.label,
    tone: display.tone,
    execution: raw,
    verification,
    review,
    known: raw in EXECUTION_STATES,
  };
}

export function taskStateFromEvent(event = {}) {
  const name = String(event.event || "");
  if (name === "use_case_started") return "running";
  if (name === "use_case_preparing_models") return "preparing_models";
  if (name === "use_case_completed") return "completed";
  if (name === "use_case_cancellation_requested") return "cancelling";
  if (name === "use_case_cancelled") return "cancelled";
  if (name === "use_case_failed") return "failed";
  return null;
}
