# Scripts

This directory contains operator conveniences and reproducible release
validation. None of these scripts run during ordinary Sparse startup. Generated
fixtures and reports belong under the ignored `.sparse-data/` directory unless
an explicit output path is supplied.

The portable automated test suite lives under `tests/` and `dashboard/tests/`.
Scripts named `profile_*`, `smoke_*`, or `test_real_*` are manual qualification
tools because they may require local models, a GPU, a browser, or substantial
memory.

## Start and stop on Windows

- `start_sparse.ps1` starts the authenticated controller, optionally loads the
  fleet, opens the dashboard, and records owned process identities.
- `stop_sparse.ps1` safely stops processes recorded by the launcher after
  checking their creation times.

Linux users should follow [`docs/linux.md`](../docs/linux.md); the PowerShell
launch helpers are Windows-only.

## Runtime setup and downloads

- `setup_qwen_fp8.ps1` creates the isolated Qwen/vision Transformers runtime
  and installs the pinned FP8 kernel.
- `setup_ocr.ps1` creates the isolated PP-OCRv6 Transformers runtime and checks
  its imports.
- `setup_embeddings.ps1` creates the CPU embedding runtime.
- `setup_decisions.ps1` creates the CUDA decision-router runtime.
- `download_embeddinggemma2_d1.ps1` downloads the pinned full snapshots used by
  the embedding and decision workers.

Use `uv run sparse-network models pull-all` for the normal registry-driven
model download. There is intentionally no separate PaddleOCR-VL downloader;
`pull-all` and repeated `--endpoint` options cover it.

## Admission fixtures and evaluation

- `create_vision_fixtures.py` creates deterministic images for vision admission.
- `create_retrieval_fixtures.py` creates the frozen multilingual retrieval set.
- `benchmark_retrieval.py` evaluates one embedding endpoint against that set.
- `evaluate_coordination.py` aggregates coordination metrics from saved
  execution traces.

These tools support the suites under `configs/admission/` and the evaluation
procedure in `docs/coordination-evaluation.md`.

## Portable dashboard and API smoke checks

- `smoke_dashboard.py` checks the dashboard against the mock controller.
- `smoke_dashboard_controls.py` checks roles and controlled mock operations.
- `smoke_dashboard_browser.mjs` drives the browser journey against a running
  controller.
- `smoke_usecases.py` exercises the authenticated use-case API without models.

The first three are also documented in `dashboard/README.md`. These checks
complement, rather than replace, the automated unit tests.

## Real-model workflow checks

- `smoke_workflows.py` runs repeatable top-2 and MoSA workflow cases.
- `test_real_ocr.py` runs fast or complex OCR on a generated image-only PDF and
  verifies exact invoice values.
- `test_real_usecases.py` runs evidence-grounded use cases against installed
  models and local test inputs.

Run these only after configuring `config.local.yaml` and verifying the required
model artifacts. They load and unload local runtimes and may consume most of the
selected hardware profile.

## Hardware profiling

- `profile_roster.py` loads a selected roster and records a smoke measurement.
- `profile_fleet.py` measures the complete resident fleet, including optional
  parallel requests.
- `profile_context.py` measures a selected endpoint with a substantial working
  context.
- `profile_swap.py` repeats the Qwen 27B swap-and-restore path.
- `profile_coexistence.py` measures two independent endpoints before fleet
  scheduling.
- `profile_qwen38_splits.py` compares candidate Qwen 27B GPU-layer splits.
- `profile_vision_components.py` separates vision model, projector, image, and
  generation memory costs.
- `profile_routing.py` exercises static routes through the resident fleet.
- `profile_workflows.py` exercises admitted top-2 and MoSA workflows through the
  resident fleet.

These profilers are retained because the hardware rosters and context limits
contain measured claims. They provide a repeatable way for maintainers to
re-qualify those claims after model, runtime, or driver changes. They are not
portable benchmarks and are not part of CI.
