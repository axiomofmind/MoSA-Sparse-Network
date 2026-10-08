# Task-first workspace

The Sparse dashboard opens on **Home** and keeps everyday work separate from
controller operations. Primary navigation contains Home and My work. Settings
contains connection, local draft policy, hardware profiles, and role-aware
advanced tools.

## Starting and resuming work

Home has one request field, file selection, and one **Continue** action. Seven
starter cards show a concrete request and expected result. Local deterministic
suggestion returns a confident match, an ambiguity that the user resolves, or
no match; it never silently chooses an unrelated first template. Advanced
evaluation is excluded from ordinary suggestions.

The request and selected files move into one stable `#workspace/{draft_id}`
route. A ready request starts immediately. A request with missing fields,
unavailable capabilities, or required model preparation stays in the same
workspace and presents the specific interruption and action. My work lists
saved drafts and controller tasks, and resumes an existing workspace without
resubmitting it.

Serializable draft fields are retained in browser local storage for 30 days.
The connection token is session-only. Attachment contents are never put in local
storage; only metadata is retained and reloaded attachments are marked **choose
again**. Settings provides a clear-all action.

## Task thread and sources

The workspace shows the original request as the first user turn, execution as a
Sparse status/result turn, and a follow-up composer immediately below a completed
result. Optional task settings and workflow details remain collapsed. Sources
stay attached beside the thread. Browser files, pasted text, and approved local
paths are clearly distinguished. Local paths are still checked by the controller against
`allowed_source_roots`; browser filenames are never treated as filesystem paths.

Attachment classification is constrained to the selected template's admitted
input kinds. Unsupported and ambiguous files remain visible with an inline type
choice rather than falling back silently. Binary uploads remain bounded base64;
text files remain text inputs.

Before submission the browser validates obvious fields and calls
`POST /v1/use-cases/readiness`. The controller determines which conditional
stages are required for that request, separates required from optional stages,
and reports ready, needs preparation, or unavailable. The browser does not
weaken a required verifier or download a model. If installed models need loading,
the workspace explains the preparation and requires the user to start that step.

## Execution and updates

Ready requests submitted from Home start without a second form confirmation.
When additional information is necessary, the workspace retains the original
request and exposes only the relevant fields plus collapsed optional settings.
Review workflow steps remains a collapsed disclosure.

The controller returns a stable job ID. Event polling reconnects to that ID and
never resubmits after a connection loss. In a workspace, polling patches the
task-thread region instead of replacing an active editor, preserving typing,
selection, expanded disclosures, draft state, and scroll position.

Each follow-up posts a new bounded use-case run with `parent_run_id`. The
controller verifies that the parent uses the same task template, inherits its
immutable source/evidence references, stores the prior deliverable as a citable
artifact, and records `thread_id` and `turn_index`. Follow-ups therefore use
auditable task context rather than an unbounded browser-authored chat transcript.

## Results, review, and export

The result view leads with the answer or document table. Its versioned contract
contains named sources, artifact-backed locations, claim citations, recorded
checks, limitations, next actions, and technical provenance. Models, routes,
timing, events, IDs, and raw structures remain under **How this was produced**.

The original result is immutable (`v1`). Saving a document correction creates a
new parent-linked reviewed version with the original value, edited value,
editor, time, and source reference. Original checks stay on the original;
affected checks become stale on the reviewed version and applicable structural
checks are rerun and labeled with that version.

Report and table CSV exports are generated from a persisted selected version,
not transient browser state. CSV export uses stable columns, UTF-8 with BOM, and
formula-injection protection. Export artifacts record version and provenance.

## Verification

Run the portable frontend and controller gates with:

```powershell
node --test dashboard/tests/*.test.mjs
uv run ruff check .
uv run mypy
uv run pytest -q
uv run python scripts/smoke_dashboard.py
uv run python scripts/smoke_dashboard_controls.py
node scripts/smoke_dashboard_browser.mjs http://127.0.0.1:8765 $env:SPARSE_API_TOKEN
```
