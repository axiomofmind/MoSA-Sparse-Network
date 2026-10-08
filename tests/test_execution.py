from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from sparse_network.config import AppConfig
from sparse_network.errors import RequestFailedError
from sparse_network.execution import (
    DeterministicVerifier,
    ExecutionGraph,
    ExecutionNode,
    ExecutionRequest,
    Top1Executor,
    validate_json_schema_subset,
)
from sparse_network.fleet import FleetManager
from sparse_network.models import ModelRegistry
from sparse_network.routing import RouteRequest, StaticRouter
from sparse_network.workflows import WorkflowExecutor, WorkflowRequest


def _endpoint(endpoint_id: str, *, family: str, vision: bool = False) -> dict:
    return {
        "display_name": endpoint_id,
        "role": "fixture",
        "family": family,
        "source": {"type": "builtin"},
        "runtime": {"adapter": "mock"},
        "modalities": ["text", "image"] if vision else ["text"],
        "capabilities": ["visual_understanding"] if vision else ["text_generation", "verification"],
        "context_size": 4096,
        "max_output_tokens": 512,
        "max_input_characters": 3584,
        "proposed_budget": {
            "incremental_vram_bytes": 0,
            "kv_cache_bytes": 0,
            "request_workspace_bytes": 0,
        },
        "admission": {"state": "admitted"},
    }


def _system(tmp_path: Path) -> tuple[FleetManager, Top1Executor]:
    configs = tmp_path / "configs"
    (configs / "rosters").mkdir(parents=True)
    (configs / "hardware").mkdir(parents=True)
    (configs / "models.yaml").write_text(
        yaml.safe_dump(
            {
                "schema": "sparse-network-model-registry.v1",
                "endpoints": {
                    "text-worker": _endpoint("text-worker", family="qwen"),
                    "critic-worker": _endpoint("critic-worker", family="nemotron"),
                    "reason-worker": _endpoint("reason-worker", family="lfm"),
                    "vision-worker": _endpoint(
                        "vision-worker", family="gemma", vision=True
                    ),
                },
            }
        ),
        encoding="utf-8",
    )
    routes = {
        "schema": "sparse-network-route-config.v1",
        "defaults": {
            "lane": "general_generation",
            "maximum_graph_nodes": 8,
            "router_score_threshold": 0.7,
            "allowed_tools": ["artifact_read", "retrieval_search"],
        },
        "macros": {"general": "general_generation", "vision": "visual_understanding"},
        "router_score_lanes": ["general_generation"],
        "lanes": {
            "general_generation": {
                "endpoint": "text-worker",
                "capability": "text_generation",
                "modalities": ["text"],
                "keywords": [],
            },
            "tool_planning": {
                "endpoint": "text-worker",
                "capability": "text_generation",
                "modalities": ["text"],
                "keywords": [],
            },
            "visual_understanding": {
                "endpoint": "vision-worker",
                "capability": "visual_understanding",
                "modalities": ["text", "image"],
                "keywords": [],
            },
            "visual_document_extraction": {
                "endpoint": "vision-worker",
                "capability": "visual_understanding",
                "modalities": ["text", "image"],
                "keywords": [],
            },
            "gui_screen_grounding": {
                "endpoint": "vision-worker",
                "capability": "visual_understanding",
                "modalities": ["text", "image"],
                "keywords": [],
            },
            "audio_transcription": {
                "endpoint": None,
                "capability": "audio_transcription",
                "modalities": ["text", "audio"],
                "keywords": [],
            },
            "high_risk": {
                "endpoint": "text-worker",
                "capability": "verification",
                "modalities": ["text"],
                "keywords": [],
                "verification_required": True,
                "human_review_required": True,
            },
        },
    }
    (configs / "routes.yaml").write_text(yaml.safe_dump(routes), encoding="utf-8")
    (configs / "workflows.yaml").write_text(
        yaml.safe_dump(
            {
                "schema": "sparse-network-workflow-config.v1",
                "maximum_stages": 3,
                "frozen_triggers": [
                    "deterministic_failure",
                    "disputed",
                    "high_risk",
                    "repeated_failure",
                ],
                "top2": {
                    "general_generation": ["text-worker", "critic-worker"],
                    "high_risk": ["text-worker", "critic-worker"],
                },
                "mosa": {
                    "general_generation": [
                        "text-worker",
                        "critic-worker",
                        "reason-worker",
                    ],
                    "high_risk": ["text-worker", "critic-worker", "reason-worker"],
                },
            }
        ),
        encoding="utf-8",
    )
    (configs / "rosters" / "test.yaml").write_text(
        yaml.safe_dump(
            {
                "id": "test-roster",
                "hardware_profile": "test",
                "resident": [
                    "text-worker",
                    "critic-worker",
                    "reason-worker",
                    "vision-worker",
                ],
            }
        ),
        encoding="utf-8",
    )
    (configs / "hardware" / "test.yaml").write_text(
        yaml.safe_dump(
            {
                "scheduler": {"initial_parallel_generations": 1},
                "budgets": {"minimum_vram_reserve_gb": 0, "total_kv_cache_gb": 1},
            }
        ),
        encoding="utf-8",
    )
    data = {
        "paths": {
            "model_cache": str(tmp_path / "cache"),
            "artifacts": str(tmp_path / "artifacts"),
            "logs": str(tmp_path / "logs"),
            "runs": str(tmp_path / "runs"),
            "indexes": str(tmp_path / "indexes"),
        },
        "runtime_executables": {"llama_cpp": "llama-server"},
        "model_registry": str(configs / "models.yaml"),
        "endpoint_overrides": {},
        "controller": {"request_timeout_seconds": 5, "telemetry_interval_seconds": 0.05},
        "fleet": {
            "roster": str(configs / "rosters" / "test.yaml"),
            "queue_capacity": 4,
            "require_gpu_telemetry": True,
        },
        "routing": {
            "routes": str(configs / "routes.yaml"),
            "workflows": str(configs / "workflows.yaml"),
        },
    }
    config = AppConfig(root=tmp_path, data=data, sources=())
    registry = ModelRegistry.load(config)
    manager = FleetManager(config, registry)
    router = StaticRouter(config, registry, resident_endpoint_ids=set(manager.entries))
    return manager, Top1Executor(manager, router)


def test_top1_accepts_after_one_invocation_and_persists_trace(tmp_path: Path) -> None:
    manager, executor = _system(tmp_path)
    manager.load_all()
    result = executor.run(
        ExecutionRequest(
            route=RouteRequest(prompt="@route:general hello"),
            expected_contains="Mock response",
        )
    )
    assert result.status == "accepted"
    assert sum(node.kind == "model_invoke" for node in result.graph.nodes) == 1
    assert result.verification is not None and result.verification.accepted
    assert result.model_result is not None
    assert result.model_result.envelope.verification["accepted"]
    original_id = result.model_result.lifecycle["original_request_reference"]
    assert manager.controller.artifacts.read_text(original_id) == "@route:general hello"
    assert result.model_result.answer == "Mock response: hello"
    assert result.trace_path.exists()
    manager.shutdown()


def test_objective_verification_failure_cannot_be_overridden(tmp_path: Path) -> None:
    manager, executor = _system(tmp_path)
    manager.load_all()
    result = executor.run(
        ExecutionRequest(
            route=RouteRequest(prompt="not JSON"),
            response_schema={"type": "object", "required": ["label"]},
        )
    )
    assert result.status == "verification_failed"
    assert result.verification is not None
    assert "response_json_schema" in result.verification.to_dict()["failed_checks"]
    manager.shutdown()


def test_high_risk_result_waits_for_human_review(tmp_path: Path) -> None:
    manager, executor = _system(tmp_path)
    manager.load_all()
    result = executor.run(ExecutionRequest(route=RouteRequest(prompt="assess", high_risk=True)))
    assert result.status == "needs_human_review"
    assert result.graph.nodes[-1].kind == "human_review"
    assert result.graph.nodes[-1].state == "pending"
    manager.shutdown()


def test_evidence_citation_and_tool_argument_checks(tmp_path: Path) -> None:
    manager, executor = _system(tmp_path)
    manager.load_all()
    evidence = manager.controller.artifacts.put_text(
        request_id="request-evidence",
        kind="evidence_chunk",
        text="The release color is cobalt.",
        metadata={"source_name": "fixture.txt", "line_start": 1, "line_end": 1},
    )
    cited = executor.run(
        ExecutionRequest(
            route=RouteRequest(prompt=f"Cite {evidence.id}"),
            evidence_references=(evidence.id,),
            require_citations=True,
        )
    )
    assert cited.status == "accepted"
    assert cited.verification is not None
    assert "source_references" in cited.verification.to_dict()["passed_checks"]
    valid_tool = DeterministicVerifier.verify_tool_arguments(
        {"retrieval_search": {"query": "release color", "top_k": 3}}
    )
    invalid_tool = DeterministicVerifier.verify_tool_arguments(
        {"retrieval_search": {"query": "release color", "top_k": "three"}}
    )
    assert valid_tool.passed
    assert not invalid_tool.passed
    manager.shutdown()


def test_rejected_route_has_no_model_invocation(tmp_path: Path) -> None:
    manager, executor = _system(tmp_path)
    manager.load_all()
    result = executor.run(
        ExecutionRequest(
            route=RouteRequest(
                prompt="run shell",
                requested_tools=("shell",),
                authorized_tools=("shell",),
            )
        )
    )
    assert result.status == "route_rejected"
    assert all(node.kind != "model_invoke" for node in result.graph.nodes)
    manager.shutdown()


def test_graph_rejects_cycles_multiple_invocations_and_recursion() -> None:
    cyclic = ExecutionGraph(
        graph_id="cycle",
        maximum_nodes=4,
        nodes=[
            ExecutionNode("a", "route", depends_on=("b",)),
            ExecutionNode("b", "verify", depends_on=("a",)),
        ],
    )
    with pytest.raises(RequestFailedError, match="cycle"):
        cyclic.validate()
    multiple = ExecutionGraph(
        graph_id="multiple",
        maximum_nodes=4,
        nodes=[
            ExecutionNode("route", "route"),
            ExecutionNode("one", "model_invoke", depends_on=("route",)),
            ExecutionNode("two", "model_invoke", depends_on=("route",)),
        ],
    )
    with pytest.raises(RequestFailedError, match="invocation bound"):
        multiple.validate()
    recursive = ExecutionGraph(
        graph_id="recursive",
        maximum_nodes=4,
        nodes=[ExecutionNode("route", "route", metadata={"expands_graph": True})],
    )
    with pytest.raises(RequestFailedError, match="recursively"):
        recursive.validate()


def test_bounded_json_schema_subset() -> None:
    schema = {
        "type": "object",
        "required": ["label", "scores"],
        "properties": {
            "label": {"type": "string"},
            "scores": {"type": "array", "items": {"type": "number"}},
        },
    }
    assert validate_json_schema_subset({"label": "a", "scores": [0.5]}, schema) == []
    assert validate_json_schema_subset({"label": 3}, schema)


def test_diverse_top2_requires_frozen_trigger_and_runs_two_families(tmp_path: Path) -> None:
    manager, executor = _system(tmp_path)
    manager.load_all()
    workflows = WorkflowExecutor(manager, executor.router)
    result = workflows.run(
        WorkflowRequest(
            execution=ExecutionRequest(
                route=RouteRequest(prompt="Reply TOKEN"),
                expected_contains="TOKEN",
            ),
            mode="top-2",
            trigger="disputed",
        )
    )
    assert result.status == "accepted"
    assert len(result.stages) == 2
    assert result.comparison["family_diverse"]
    assert sum(node.kind == "model_invoke" for node in result.graph.nodes) == 2
    with pytest.raises(RequestFailedError, match="not frozen"):
        workflows.run(
            WorkflowRequest(
                execution=ExecutionRequest(route=RouteRequest(prompt="hello")),
                mode="top-2",
                trigger="because_model_asked",
            )
        )
    manager.shutdown()


def test_workflow_checks_only_the_selected_lane_against_the_active_profile(
    tmp_path: Path,
) -> None:
    manager, executor = _system(tmp_path)
    manager.load_all()
    manager.entries.pop("reason-worker")
    workflows = WorkflowExecutor(manager, executor.router)

    top2 = workflows.run(
        WorkflowRequest(
            execution=ExecutionRequest(route=RouteRequest(prompt="general request")),
            mode="top-2",
            trigger="disputed",
        )
    )
    assert top2.status == "accepted"

    with pytest.raises(RequestFailedError, match="models are not assigned: reason-worker"):
        workflows.run(
            WorkflowRequest(
                execution=ExecutionRequest(route=RouteRequest(prompt="general request")),
                mode="mosa",
                trigger="disputed",
            )
        )
    manager.shutdown()


def test_mosa_early_stop_and_bounded_three_stage_reconciliation(tmp_path: Path) -> None:
    manager, executor = _system(tmp_path)
    manager.load_all()
    workflows = WorkflowExecutor(manager, executor.router)
    early = workflows.run(
        WorkflowRequest(
            execution=ExecutionRequest(
                route=RouteRequest(prompt="Reply EARLY_OK"),
                expected_contains="EARLY_OK",
            ),
            mode="mosa",
            trigger="deterministic_failure",
        )
    )
    assert early.status == "accepted"
    assert len(early.stages) == 1
    assert early.comparison["early_stopped"]
    full = workflows.run(
        WorkflowRequest(
            execution=ExecutionRequest(
                route=RouteRequest(prompt="Reply FINAL_OK"),
                expected_contains="FINAL_OK",
            ),
            mode="mosa",
            trigger="disputed",
        )
    )
    assert full.status == "accepted"
    assert [stage.stage for stage in full.stages] == [
        "draft",
        "critique",
        "reconciliation",
    ]
    assert full.stages[1].context["stages"][0]["stage"] == "draft"
    assert full.stages[2].context["stages"][1]["stage"] == "critique"
    assert full.coordination["stage_count"] == 3
    assert full.comparison["critique_error_count"] == 0
    assert len(full.graph.nodes) == 8
    manager.shutdown()


def test_repetition_retry_uses_one_different_family_and_is_finite(tmp_path: Path) -> None:
    manager, executor = _system(tmp_path)
    manager.load_all()
    workflows = WorkflowExecutor(manager, executor.router)
    recovered = workflows.run(
        WorkflowRequest(
            execution=ExecutionRequest(
                route=RouteRequest(
                    prompt="[mock:loop:text-worker] Reply with RECOVERY_OK"
                ),
                expected_contains="RECOVERY_OK",
            ),
            mode="top-2",
            trigger="repeated_failure",
        )
    )
    assert recovered.status == "accepted"
    assert recovered.selected_stage == "secondary"
    assert recovered.comparison["repetition_recovered"]
    assert len(recovered.stages) == 2
    assert recovered.stages[0].result.envelope.repetition["detected"]
    assert recovered.stages[1].result.envelope.repetition["retry_count"] == 1
    assert (
        manager.registry.get(recovered.stages[0].endpoint).family
        != manager.registry.get(recovered.stages[1].endpoint).family
    )
    manager.shutdown()


def test_mosa_skips_reconciliation_after_a_clean_critic_when_policy_allows(
    tmp_path: Path,
) -> None:
    manager, executor = _system(tmp_path)
    workflow_path = Path(str(manager.config.data["routing"]["workflows"]))
    configured = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
    configured["frozen_triggers"].append("maximum_quality")
    configured["coordination"] = {
        "context_character_limit": 1600,
        "continue_after_verified": ["maximum_quality"],
        "accept_after_clean_critique": ["maximum_quality"],
    }
    workflow_path.write_text(yaml.safe_dump(configured), encoding="utf-8")
    manager.load_all()
    workflows = WorkflowExecutor(manager, executor.router)
    result = workflows.run(
        WorkflowRequest(
            execution=ExecutionRequest(
                route=RouteRequest(prompt="Reply QUALITY_OK"),
                expected_contains="QUALITY_OK",
            ),
            mode="mosa",
            trigger="maximum_quality",
        )
    )
    assert result.status == "accepted"
    assert [stage.stage for stage in result.stages] == ["draft", "critique"]
    assert result.selected_stage == "draft"
    assert result.comparison["early_stopped_after"] == "critique"
    assert result.coordination["decision"] == "accepted_after_clean_critique"
    manager.shutdown()
