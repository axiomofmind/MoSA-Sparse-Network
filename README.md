# MoSA Sparse Network

MoSA Sparse Network is a portable controller and dashboard project for routing work
across a bounded fleet of local models. The project includes provisional 16 GB
and 24 GB profiles plus a measured 32 GB multimodal reference system.

The controller includes bounded top-1, diverse top-2, and MoSA execution,
AntiDoom loop rejection, a
measured Qwen3.8-27B swap/restoration path, and a versioned loopback API, CLI,
typed client, replayable event stream, role-aware dashboard, confirmed fleet
controls, portable hardware-profile switching, and evaluation comparisons.

## Development quick start

```powershell
uv sync
uv run sparse-network doctor
uv run sparse-network models verify mock-echo
uv run sparse-network smoke mock-echo --prompt "hello"
uv run ruff check .
uv run mypy
uv run pytest
```

On Windows, launch the controller and dashboard from PowerShell:

```powershell
.\scripts\start_sparse.ps1
```

The launcher uses `config.local.yaml`, copies the connection
token to the clipboard, and opens the dashboard when ready. Paste the token
into the dashboard's Connect dialog. Keep the terminal open; press Ctrl+C to
stop. Load your models with **Models & Hardware > Start fleet** after connecting.
The dashboard starts without waiting for model loading. It requires `uv` and works from any current directory when invoked by
its full path.

Use `-Mock` for a model-free demo, `-Config <path>` for another configuration
(relative paths resolve from the repository root), or `-Port 8766` for another
port. `-LoadFleet` loads models before serving the dashboard; this can delay
dashboard availability. Mock mode loads its model-free fleet automatically.
`-NoBrowser` and `-NoClipboard` disable those conveniences. An existing
`SPARSE_API_TOKEN` is reused; otherwise the launcher generates a session token.

To stop this checkout's controllers and model runtimes from another PowerShell
terminal, including runtimes recorded by the launcher that outlived their controller:

```powershell
.\scripts\stop_sparse.ps1
```

This interrupts active tasks. Use `-WhatIf` to preview the processes it would stop.
The launcher stores process identities under `.sparse-data/launcher/`; the stop
script checks process creation times to avoid stopping reused PIDs. Untracked
orphan model servers from older launches are left alone when ownership cannot be
established. Other applications are not targeted.

Alternatively, run the model-free dashboard manually with:

```powershell
$env:SPARSE_API_TOKEN = "replace-with-at-least-16-characters"
uv run sparse-network --config configs/mock.yaml api serve --load-fleet
```

Open `http://127.0.0.1:8765/dashboard/`. See
[`dashboard/README.md`](dashboard/README.md) for the views, security boundary,
permissions, and repeatable smoke tests.

For a real local endpoint, copy `config.example.yaml` to `config.local.yaml`,
set the local cache and runtime details, then run:

```powershell
uv run sparse-network models verify qwen35-4b
uv run sparse-network smoke qwen35-4b --prompt "Reply with exactly: ready" --timeout 60
```

## Downloading the pinned model set

Preview the complete smallest-to-largest download order without changing the
model cache:

```powershell
uv run sparse-network models pull-all --dry-run
```

The current admitted set contains 14 model entries and requires up to 80.35 GiB
for a new cache. The command includes both PP-OCRv6 components, downloads only
the files declared by the registry, validates each model before continuing, and
skips valid files already present. `mock-echo` has no files and is omitted.

After reviewing the upstream licenses, run the sequential download with:

```powershell
uv run sparse-network models pull-all `
  --acknowledge-licenses
```

Use repeated `--endpoint <id>` options for a subset, `--continue-on-error` to
finish the remaining queue after a failure, or `--force` to re-check every
snapshot through Hugging Face. The destination defaults to the ignored
`models/` folder. A configured `paths.model_cache` or `SPARSE_MODEL_CACHE`
overrides that default, and `--cache-dir <path>` takes precedence over both.
Model acquisition remains explicit; normal startup never downloads weights.

Milestone 4 admissions use deterministic frozen cases:

```powershell
uv run python scripts/create_vision_fixtures.py .sparse-data/fixtures/vision
uv run sparse-network models verify
uv run sparse-network admission gemma-e2b-vision --suite configs/admission/milestone4.yaml
```

Qwen FP8 needs an isolated CUDA runtime and the pinned fine-grained FP8 kernel.
Create it explicitly with `scripts/setup_qwen_fp8.ps1 -CacheDir <hub-cache>`;
ordinary startup never downloads weights or kernels.

CPU retrieval is available through the same CLI:

```powershell
uv run sparse-network retrieval index README.md docs/task-first-workspace.md
uv run sparse-network retrieval search --query "How does model escalation work?" --top-k 3
uv run sparse-network retrieval ask qwen35-4b --query "Explain the escalation policy" --top-k 3
```

Harrier is the configured model-backed default. Pass `--endpoint
hash-embedding` for a portable model-free demonstration or `--endpoint
minilm-l6-v2` for the smaller baseline.

EmbeddingGemma 2 is the admitted 256-dimensional long-context and multilingual
retriever used by the research-synthesis workflow. Harrier remains the faster
general retrieval default. After explicit setup and download, verify the frozen
retrieval gate with:

```powershell
scripts/setup_embeddings.ps1
scripts/download_embeddinggemma2_d1.ps1 -CacheDir <hub-cache>
uv run python scripts/benchmark_retrieval.py embeddinggemma2-256
```

d1-3B is the admitted typed advisory decision router. It supplies bounded lane
probabilities while deterministic routing, modality, authorization, and risk
rules keep precedence:

```powershell
scripts/setup_decisions.ps1
uv run sparse-network route plan --decision-endpoint d1-3b `
  --prompt "Fix this Python function" --json
```

See `docs/embeddinggemma2-d1.md` for pinned revisions, isolation boundaries,
license status, and admission scope.

Create a separate CPU embedding environment with
`scripts/setup_embeddings.ps1` and point
`runtime_executables.embeddings` at its Python executable in local
configuration. Model acquisition remains a separate explicit action.

Run one request with the complete resident roster (the command loads and safely
shuts down the fleet around it):

```powershell
uv run sparse-network fleet run qwen35-4b --prompt "Reply with exactly: READY"
```

Plan a route without loading models, or run it through the full resident fleet:

```powershell
uv run sparse-network route plan --prompt "Fix this Python function" --json
uv run sparse-network route run --lane reason `
  --prompt "Reconcile these conclusions" --json
```

Start the authenticated local controller API and operate it from another shell:

```powershell
$env:SPARSE_API_TOKEN = "replace-with-a-random-local-token"
uv run sparse-network api serve --load-fleet
uv run sparse-network api status
uv run sparse-network api request qwen35-4b --prompt "hello"
uv run sparse-network api workflow --mode top-2 --trigger disputed --prompt "check this"
uv run sparse-network api events --after 0
```

Workflow stages exchange bounded controller-authored context and record their
handoffs and coordination decisions in the execution trace. Aggregate saved
traces with `scripts/evaluate_coordination.py`; see
`docs/coordination-evaluation.md` for the policy and metrics.

Hardware-tier rosters are data files and do not contain local model paths:

- `configs/rosters/reference-16gb.yaml`: four small specialists plus Gemma 12B.
- `configs/rosters/reference-24gb.yaml`: the five-agent baseline plus Gemma 12B.
- `configs/rosters/reference-32gb-gemma12.yaml`: the measured six-resident tier
  with the Qwen 27B Q8 quality-control swap; this is the default 32 GB profile.
- `configs/rosters/reference-32gb.yaml`: the measured five-resident 32 GB Lean
  variant, retained for lower steady-state memory use.

The 16/24 GB profiles require validation on matching cards. The six-resident
layout has been measured at 17.06 GB total VRAM on the reference machine.
The controller binds its exclusive verifier to the selected roster: Q4 on the
16/24 GB profiles and Q8 on both 32 GB profiles.

Use-case workflows are available from the dashboard and the versioned
`/v1/use-cases` API. Describe an outcome on Home: a confidently matched, ready
request opens its task thread and starts immediately, while missing information
or model preparation remains visible as a focused interruption. Results include
a follow-up composer whose parent-linked runs inherit verified evidence without
forming an unbounded chat transcript. Optional settings and workflow details stay
collapsed. The eighth routing/AntiDoom workflow remains under advanced
evaluation. Start with `plan_only: true` to inspect routes and capability gaps
without invoking a model. See
`docs/task-first-workspace.md` for the guided workspace, contracts, permissions,
and verification commands.

Real-model integration tests are opt-in:

```powershell
$env:SPARSE_RUN_REAL_MODEL_TEST = "1"
uv run pytest -q tests/test_real_qwen_opt_in.py
$env:SPARSE_RUN_MILESTONE4_TESTS = "1"
uv run pytest -q tests/test_real_milestone4_opt_in.py
```

Model weights, local configuration, user artifacts, logs, and run data are not
stored in this repository. The model sources and pinned revisions are also
listed at [axiomofmind/MoSA-Sparse-Network](https://huggingface.co/axiomofmind/MoSA-Sparse-Network).
