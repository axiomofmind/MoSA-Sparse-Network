# Security policy

## Reporting a vulnerability

Please report security issues privately through
[GitHub Security Advisories](https://github.com/axiomofmind/MoSA-Sparse-Network/security/advisories/new).
Do not open a public issue for a suspected vulnerability and do not attach
private documents, prompts, model caches, tokens, logs, or generated artifacts.

Include a concise description, affected version or commit, reproduction steps,
and the expected impact. Reports will be acknowledged as soon as practical.

## Security boundary

MoSA Sparse Network is designed for loopback, local-first operation. Keep the
controller bound to `127.0.0.1`, use a unique API token of at least 16
characters, and do not expose the service directly to an untrusted network.
Model code requiring `trust_remote_code` is pinned by revision and should be
reviewed again whenever that revision changes.

Local configuration, model weights, source documents, vector indexes, logs,
run records, and generated artifacts are excluded from the source repository.

## Supported versions

Until the first stable release, security fixes are applied to the latest commit
on the default branch only.
