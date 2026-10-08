# EmbeddingGemma 2 and d1-3B admission

Date: 2026-10-07

Both endpoints passed their minimum role-specific admission gates on
2026-10-08. EmbeddingGemma 2 is the research-synthesis retriever; d1-3B is the
explicit advisory route scorer. Neither is a resident generative endpoint.

## Pinned artifacts

- `google/embeddinggemma-2` at
  `914f7f89142e33e77833254d9c9b90c3cef7303b`.
- `LiquidAI/d1-3B` at
  `051bcc464b01b9f92942b364d9586b0ef5912432`.

Use `scripts/download_embeddinggemma2_d1.ps1 -CacheDir <hub-cache>` for an
explicit download. Ordinary application startup remains offline and never
acquires weights.

## EmbeddingGemma 2

The admitted surface is text retrieval at 256 dimensions. It uses the
model's `SearchQuery` and `Document` prompt names, disables the unused vision
and audio encoders, normalizes the truncated vectors, and runs in float32 on
CPU. The existing endpoint/revision/dimension index identity keeps its vectors
separate from Harrier and MiniLM indexes.

It returned the expected document at rank one for all five frozen English,
Spanish, and German retrieval cases. Native image, audio, and video indexing
needs a media-coordinate index and is deliberately outside this text-only role.

## d1-3B

d1 is a typed decision endpoint, not a generator. Its admitted integration
surface is explicit advisory route scoring through the CLI. The minimum gate
selected `coding_repair` for a code-repair request with 0.948 confidence. The
controller's deterministic modality, authorization, high-risk, and
explicit-lane rules retain precedence.

The repository's custom Python files were reviewed at the pinned revision for
process creation, socket/network clients, dynamic `eval`/`exec`, and direct
file opening. No such behavior was found. The code performs model execution,
prompt construction, tensor operations, and local image processing. It still
runs in an isolated process with Hugging Face and Transformers offline modes
enabled. The LFM 1.0 license still requires separate redistribution review;
admission does not authorize republishing the weights.

Create the CUDA environment with `scripts/setup_decisions.ps1`, then set
`runtime_executables.decisions` in ignored local configuration.
