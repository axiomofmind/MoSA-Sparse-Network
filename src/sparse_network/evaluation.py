"""Deterministic coordination metrics for saved workflow results and traces."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def evaluate_coordination_results(
    results: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate comparable metrics without invoking a model or judging semantics."""

    values = list(results)
    statuses: Counter[str] = Counter()
    endpoints: Counter[str] = Counter()
    triggers: Counter[str] = Counter()
    stage_count = 0
    verified_stage_count = 0
    escalated = 0
    reconciled = 0
    first_pass = 0
    human_review = 0
    recovered = 0
    disagreements = 0
    top2_runs = 0
    handoff_opportunities = 0
    handoff_present = 0
    truncated_handoffs = 0
    critique_count = 0
    critiques_with_findings = 0
    clean_reconciliations = 0
    total_elapsed_ms = 0
    peak_vram_bytes = 0
    peak_ram_bytes = 0

    for value in values:
        status = str(value.get("status", "unknown"))
        statuses[status] += 1
        trigger = str(value.get("trigger", "unknown"))
        triggers[trigger] += 1
        stages = value.get("stages", [])
        if not isinstance(stages, list):
            stages = []
        if len(stages) > 1:
            escalated += 1
        if len(stages) == 1 and status in {"accepted", "needs_human_review"}:
            first_pass += 1
        if status == "needs_human_review":
            human_review += 1
        if any(
            isinstance(stage, Mapping) and stage.get("stage") == "reconciliation"
            for stage in stages
        ):
            reconciled += 1

        comparison = value.get("comparison", {})
        if not isinstance(comparison, Mapping):
            comparison = {}
        if comparison.get("repetition_recovered") is True:
            recovered += 1
        if str(value.get("mode")) == "top-2":
            top2_runs += 1
            if comparison.get("exact_normalized_agreement") is False:
                disagreements += 1
        critique_error_count = comparison.get("critique_error_count")
        if isinstance(critique_error_count, int):
            critique_count += 1
            if critique_error_count > 0:
                critiques_with_findings += 1
            elif any(
                isinstance(stage, Mapping) and stage.get("stage") == "reconciliation"
                for stage in stages
            ):
                clean_reconciliations += 1

        for index, stage in enumerate(stages):
            if not isinstance(stage, Mapping):
                continue
            stage_count += 1
            endpoints[str(stage.get("endpoint", "unknown"))] += 1
            verification = stage.get("verification", {})
            if isinstance(verification, Mapping) and verification.get("accepted") is True:
                verified_stage_count += 1
            context = stage.get("context", {})
            independent_top2_candidate = (
                str(value.get("mode")) == "top-2"
                and index == 1
                and comparison.get("independent") is True
            )
            if index > 0 and not independent_top2_candidate:
                handoff_opportunities += 1
                if isinstance(context, Mapping) and context.get("stages"):
                    handoff_present += 1
            if isinstance(context, Mapping) and context.get("handoff_truncated") is True:
                truncated_handoffs += 1
            result = stage.get("result", {})
            envelope = result.get("envelope", {}) if isinstance(result, Mapping) else {}
            usage = (
                envelope.get("resource_usage", {})
                if isinstance(envelope, Mapping)
                else {}
            )
            if isinstance(usage, Mapping):
                total_elapsed_ms += int(usage.get("elapsed_ms", 0) or 0)
                peak_vram_bytes = max(
                    peak_vram_bytes, int(usage.get("peak_vram_bytes", 0) or 0)
                )
                peak_ram_bytes = max(
                    peak_ram_bytes, int(usage.get("peak_ram_bytes", 0) or 0)
                )

    runs = len(values)
    successful = statuses["accepted"] + statuses["needs_human_review"]
    metrics = {
        "run_count": float(runs),
        "acceptance_rate": _ratio(successful, runs),
        "failure_rate": _ratio(runs - successful, runs),
        "human_review_rate": _ratio(human_review, runs),
        "average_model_calls": round(stage_count / runs, 4) if runs else 0.0,
        "first_pass_acceptance_rate": _ratio(first_pass, runs),
        "escalation_rate": _ratio(escalated, runs),
        "reconciliation_rate": _ratio(reconciled, runs),
        "verified_stage_rate": _ratio(verified_stage_count, stage_count),
        "top2_disagreement_rate": _ratio(disagreements, top2_runs),
        "repetition_recovery_rate": _ratio(recovered, runs),
        "context_handoff_coverage": _ratio(handoff_present, handoff_opportunities),
        "context_truncation_rate": _ratio(truncated_handoffs, stage_count),
        "critic_finding_rate": _ratio(critiques_with_findings, critique_count),
        "clean_reconciliation_rate": _ratio(clean_reconciliations, reconciled),
        "average_stage_latency_ms": (
            round(total_elapsed_ms / stage_count, 2) if stage_count else 0.0
        ),
        "peak_vram_bytes": float(peak_vram_bytes),
        "peak_ram_bytes": float(peak_ram_bytes),
    }
    return {
        "schema": "sparse-network-coordination-evaluation.v1",
        "metrics": metrics,
        "counts": {
            "statuses": dict(sorted(statuses.items())),
            "triggers": dict(sorted(triggers.items())),
            "endpoints": dict(sorted(endpoints.items())),
        },
        "limitations": [
            "Deterministic checks measure contract compliance, not factual correctness.",
            "Exact-answer disagreement is lexical and is not a semantic quality score.",
        ],
    }
