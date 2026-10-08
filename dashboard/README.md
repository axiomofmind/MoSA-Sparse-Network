# Sparse Network Dashboard

The dashboard is a build-free, role-aware web application served
by the local controller. It uses semantic HTML, CSS, and native ES modules; no
Node packages or frontend build step are required.

The landing view is a focused Home page. A user describes the outcome, adds
files, and selects **Continue**. Sparse either opens a confidently matched task
and starts it when the request is ready, or asks the user to choose between
plausible result types. Missing information, model preparation, and review are
shown as explicit interruptions without hiding the original request. The stable
workspace presents the request and result as a task thread. A follow-up composer
creates a parent-linked run that inherits the prior result and verified evidence.
Optional controls and the evidence-preserving plan preview live behind
disclosures; raw controller tools remain under Settings > Advanced tools.

Draft text and settings are retained in browser local storage for 30 days.
Browser file contents and the connection token are not stored. After a reload,
attachment metadata remains visible and the file is explicitly marked for
reselection. My work resumes the same draft or controller task by stable ID.

Start it with the model-free mock profile:

```powershell
$env:SPARSE_API_TOKEN = "replace-with-at-least-16-characters"
uv run sparse-network --config configs/mock.yaml api serve --load-fleet
```

Open `http://127.0.0.1:8765/dashboard/` and enter the same token. The token is
kept only in browser session storage. Viewer sessions call read APIs only.
Operator, evaluator, and administrator sessions receive controls allowed by
their server-enforced permissions. The dashboard never launches runtimes or
reads model paths directly.

Run the frontend data-model tests with:

```powershell
node --test dashboard/tests/*.test.mjs
```

Run the controller/browser-boundary smoke test with:

```powershell
uv run python scripts/smoke_dashboard.py
```

The smoke test uses an ephemeral loopback port and confirms authentication,
redaction, the simplified shell, profile discovery, event replay, and host-path
isolation.

Run the controlled-operation smoke test with:

```powershell
uv run python scripts/smoke_dashboard_controls.py
```

With a mock controller already running, the optional installed-browser boundary
check exercises keyboard focus, contrast, reduced motion, 200% zoom, 360 px
layout, automatic task start, contextual follow-up, and
focus/cursor/disclosure retention during event polling:

```powershell
node scripts/smoke_dashboard_browser.mjs http://127.0.0.1:8765 $env:SPARSE_API_TOKEN
```

Role-token setup, local-only deployment expectations, and vulnerability
reporting are documented in the root `README.md` and `SECURITY.md`.

The guided task behavior and payload boundary are documented in
`docs/task-first-workspace.md`.
