from __future__ import annotations

from sparse_network.coordination import CoordinationContext, StageHandoff
from sparse_network.evaluation import evaluate_coordination_results


def test_coordination_context_is_bounded_and_keeps_stable_references() -> None:
    context = CoordinationContext(
        objective="Review the supplied evidence",
        trigger="maximum_quality",
        source_evidence=["artifact-source"],
    )
    context.add_stage(
        StageHandoff(
            stage="draft",
            role="candidate",
            endpoint="small-model",
            answer_reference="artifact-answer",
            evidence_references=("artifact-source",),
            answer="candidate material " * 100,
            verification_accepted=True,
        )
    )
    rendered, truncated = context.model_handoff(maximum_characters=600)
    assert len(rendered) <= 600
    assert "artifact-source" in rendered
    assert "artifact-answer" in rendered
    assert "Prior stage draft" in rendered
    assert truncated
    assert context.to_dict()["stages"][0]["answer_reference"] == "artifact-answer"


def test_coordination_evaluation_reports_quality_cost_and_handoff_metrics() -> None:
    usage = {"elapsed_ms": 25, "peak_vram_bytes": 100, "peak_ram_bytes": 200}
    stage = {
        "stage": "draft",
        "endpoint": "small-model",
        "verification": {"accepted": True},
        "context": {"stages": []},
        "result": {"envelope": {"resource_usage": usage}},
    }
    critique = {
        "stage": "critique",
        "endpoint": "critic",
        "verification": {"accepted": True},
        "context": {"stages": [{"stage": "draft"}]},
        "result": {"envelope": {"resource_usage": usage}},
    }
    report = evaluate_coordination_results(
        [
            {
                "status": "accepted",
                "mode": "mosa",
                "trigger": "deterministic_failure",
                "stages": [stage],
                "comparison": {"early_stopped": True},
            },
            {
                "status": "accepted",
                "mode": "mosa",
                "trigger": "maximum_quality",
                "stages": [stage, critique],
                "comparison": {"critique_error_count": 0},
            },
        ]
    )
    metrics = report["metrics"]
    assert metrics["acceptance_rate"] == 1.0
    assert metrics["average_model_calls"] == 1.5
    assert metrics["escalation_rate"] == 0.5
    assert metrics["context_handoff_coverage"] == 1.0
    assert metrics["verified_stage_rate"] == 1.0
    assert metrics["peak_vram_bytes"] == 100.0


def test_independent_top2_candidate_is_not_counted_as_a_context_handoff() -> None:
    report = evaluate_coordination_results(
        [
            {
                "status": "accepted",
                "mode": "top-2",
                "trigger": "disputed",
                "stages": [
                    {"stage": "primary", "context": {"stages": []}},
                    {"stage": "secondary", "context": {"stages": []}},
                ],
                "comparison": {"independent": True},
            }
        ]
    )
    assert report["metrics"]["context_handoff_coverage"] == 0.0
