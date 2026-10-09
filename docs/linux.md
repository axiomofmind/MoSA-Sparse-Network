# Linux setup and compatibility

MoSA Sparse Network's controller, dashboard, CLI, and model workers are written
in Python and are intended to run on Linux. The current reference system and
full model fleet have been qualified on Windows, however. Treat Linux support as
manual setup that still needs hardware-specific validation.

The scripts ending in `.ps1` are PowerShell helpers for Windows. They do not run
in Bash and are not required by the controller. Linux users should use the
equivalent commands below.

## What works without the PowerShell scripts

- The `uv run sparse-network ...` CLI and local HTTP API.
- The browser dashboard served by the controller.
- Model verification and sequential `models pull-all` downloads.
- llama.cpp endpoints when a compatible `llama-server` is installed.
- Isolated Transformers, OCR, embedding, and decision workers when their Linux
  Python environments are configured.
- PDF page rendering when Poppler's `pdftoppm` is installed.

The repository does not currently include a Linux launcher, automatic browser
opening, clipboard token handling, or an equivalent to `stop_sparse.ps1` for
tracked orphan-process cleanup.

## Basic setup

Install Python 3.12 and `uv`, clone the repository, and run from its root:

```bash
uv sync
cp config.example.yaml config.local.yaml
```

For a model-free validation and dashboard:

```bash
uv run sparse-network doctor
uv run sparse-network models verify mock-echo
uv run sparse-network smoke mock-echo --prompt "hello"
export SPARSE_API_TOKEN="$(uv run python -c 'import secrets; print(secrets.token_hex(16))')"
uv run sparse-network --config configs/mock.yaml api serve --load-fleet
```

Open `http://127.0.0.1:8765/dashboard/` and paste the token into the Connect
dialog. Keep the terminal open and press Ctrl+C to stop the controller.

For real models, edit `config.local.yaml` and start without loading the fleet:

```bash
export SPARSE_API_TOKEN="$(uv run python -c 'import secrets; print(secrets.token_hex(16))')"
uv run sparse-network --config config.local.yaml api serve
```

The dashboard starts immediately. Use **Models & Hardware > Start fleet** after
connecting. Add `--load-fleet` only when startup should wait for every resident
model. The API binds to loopback by default; do not expose it on a network
without adding an appropriate security boundary.

## Local configuration

Linux virtual-environment executables live under `bin`, not `Scripts`. A typical
configuration uses paths like these:

```yaml
paths:
  model_cache: /home/you/.cache/huggingface/hub

runtime_executables:
  llama_cpp: /usr/local/bin/llama-server
  transformers: .venv-qwen/bin/python
  ocr: .venv-ocr/bin/python
  embeddings: .venv-embeddings/bin/python
  decisions: .venv-decisions/bin/python
  pdf_renderer: pdftoppm

use_cases:
  allowed_source_roots:
    - /home/you/projects
```

Keep machine-specific paths in the ignored `config.local.yaml`. Do not commit
model-cache paths, tokens, or private source roots. `SPARSE_MODEL_CACHE` and
`SPARSE_LLAMA_SERVER` can override the corresponding configured paths.

Install PDF rendering support through the distribution package manager. For
example, Debian and Ubuntu provide `pdftoppm` in `poppler-utils`:

```bash
sudo apt install poppler-utils
```

## Download models

The cross-platform CLI replaces the PowerShell download helpers:

```bash
uv run sparse-network --config config.local.yaml models pull-all --dry-run
uv run sparse-network --config config.local.yaml models pull-all --acknowledge-licenses
```

Use repeated `--endpoint <id>` options to download a subset. Downloads are
sequential, smallest to largest, and normal startup never downloads weights.

## Isolated model environments

The following commands mirror the pinned Windows helpers. NVIDIA wheels and
Triton compatibility depend on the installed driver, GPU architecture, and
Linux distribution. Review the selected PyTorch index before installing it on a
different CUDA stack.

### Qwen, vision, and complex OCR

```bash
uv venv .venv-qwen --python 3.12
uv pip install --python .venv-qwen/bin/python \
  torch==2.11.0 torchvision==0.26.0 pillow==12.3.0 \
  --index-url https://download.pytorch.org/whl/cu130
uv pip install --python .venv-qwen/bin/python \
  transformers==5.15.1 accelerate==1.14.0 safetensors==0.8.0 \
  kernels==0.16.0 triton==3.6.0
```

Qwen FP8 also requires the pinned fine-grained FP8 kernel snapshot:

```bash
CACHE_DIR=/home/you/.cache/huggingface/hub
KERNEL_REVISION=7cdb05d472d6c954c7d03182ed836ebfd4610df0
uvx --from huggingface-hub hf download kernels-community/finegrained-fp8 \
  --repo-type kernel --revision "$KERNEL_REVISION" --cache-dir "$CACHE_DIR"
mkdir -p "$CACHE_DIR/kernels--kernels-community--finegrained-fp8/refs"
printf '%s' "$KERNEL_REVISION" \
  > "$CACHE_DIR/kernels--kernels-community--finegrained-fp8/refs/v4"
```

The Windows helper uses `triton-windows`; Linux must use the compatible Linux
Triton package. This Qwen FP8 path has not yet been qualified by this project on
Linux.

### Fast OCR

```bash
uv venv .venv-ocr --python 3.12
uv pip install --python .venv-ocr/bin/python \
  torch==2.11.0 torchvision==0.26.0 pillow==12.3.0 \
  --index-url https://download.pytorch.org/whl/cu130
uv pip install --python .venv-ocr/bin/python \
  paddleocr==3.7.0 transformers==5.15.1 accelerate==1.14.0 safetensors==0.8.0
.venv-ocr/bin/python -c \
  "import torch, transformers; from paddleocr import PaddleOCR; print('OCR runtime imports verified')"
```

PP-OCRv6 uses PaddleOCR's Transformers engine in Sparse. It does not require a
separate PaddlePaddle inference-engine package.

### CPU embeddings

```bash
uv venv .venv-embeddings --python 3.12
uv pip install --python .venv-embeddings/bin/python \
  torch==2.11.0 torchvision==0.26.0 \
  --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv-embeddings/bin/python \
  transformers==5.19.0 accelerate==1.14.0 safetensors==0.8.0 \
  pillow==12.0.0 sentence-transformers==6.1.0
```

### Decision router

```bash
uv venv .venv-decisions --python 3.12
uv pip install --python .venv-decisions/bin/python \
  torch==2.11.0 torchvision==0.26.0 \
  --index-url https://download.pytorch.org/whl/cu130
uv pip install --python .venv-decisions/bin/python \
  transformers==5.15.1 accelerate==1.14.0 safetensors==0.8.0 pillow==12.0.0
```

## Validate before loading the fleet

Verify configuration, artifacts, and one endpoint at a time:

```bash
uv run sparse-network --config config.local.yaml doctor
uv run sparse-network --config config.local.yaml models verify
uv run sparse-network --config config.local.yaml smoke qwen35-4b \
  --prompt "Reply with exactly: ready" --timeout 60
```

`models verify` validates registered files; it does not prove that every CUDA
kernel is compatible with the host. Smoke-test each runtime family before a
full fleet load. Hardware-profile memory measurements in the main README are
reference measurements, not Linux guarantees.

## Stopping and recovery

Ctrl+C stops a foreground controller and the runtimes it owns. There is no
Linux orphan-cleanup helper yet. If the controller exits abnormally, inspect
processes before terminating anything:

```bash
ps -ef | grep -E 'sparse-network|llama-server|transformers_worker|paddle_ocr_worker'
```

Do not use broad process-name termination on a shared host. Confirm process IDs
and command lines belong to the intended Sparse checkout first.

## Current qualification boundary

The Windows reference system has exercised resident fleets, context limits,
Qwen 27B swap-and-restore, fast and complex OCR, and workflow tests. Equivalent
Linux full-fleet and GPU-memory qualification is still outstanding. Reports of
working distributions, driver versions, and GPU configurations are welcome.
