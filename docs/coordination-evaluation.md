# Coordination and evaluation

Sparse uses `sparse-network-coordination-context.v1` for bounded stage-to-stage
handoffs. The controller, not a model, owns the envelope. It carries the original
objective, trigger, evidence artifact IDs, prior answer artifact IDs, endpoint and
role metadata, deterministic verification outcomes, and recent untrusted model
output. Model-facing handoffs are capped by `coordination.context_character_limit`
in `configs/workflows.yaml`; the full reference manifest remains in the workflow
trace.

Top-2 candidates remain independent unless a text-only endpoint must critique a
visual candidate. MoSA stops after a verified draft when no continuation trigger
applies. For configured triggers such as `maximum_quality`, it may also stop after
a schema-valid independent critique reports no errors. Disputed, failed, and
high-risk paths retain bounded reconciliation or human review.

Every saved workflow trace now contains:

- the context visible at each stage;
- the controller's continuation or stop reasons;
- critique finding counts and verification outcomes;
- stable evidence and answer artifact references;
- per-stage latency, RAM, and VRAM telemetry.

Aggregate existing traces without invoking models:

```powershell
uv run python scripts/evaluate_coordination.py .sparse-data/runs/execution-graphs
```

Use `--output <report.json>` to preserve the report. The evaluator reports
acceptance, first-pass acceptance, escalation, reconciliation, disagreement,
repetition recovery, critic yield, clean reconciliation, context coverage,
context truncation, model calls, latency, RAM, and VRAM. These are operational
and contract metrics; they do not claim factual correctness or semantic answer
equivalence.

The authenticated evaluation API can create the same immutable aggregate from
up to 500 recent traces by posting `{"suite":"coordination"}`. Supplying
`workflow_trace_ids` restricts it to named JSON files under the configured
`runs/execution-graphs` directory.
