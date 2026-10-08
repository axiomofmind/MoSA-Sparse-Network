"""Finite top-1 execution graphs and controller-owned verification."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from .artifacts import ArtifactStore
from .contracts import SmokeResult
from .errors import RequestFailedError
from .fleet import FleetManager
from .routing import RouteDecision, RouteRequest, StaticRouter

GRAPH_NODE_KINDS = {
    "route",
    "model_invoke",
    "verify",
    "compare",
    "accept",
    "human_review",
    "reject",
}
GRAPH_NODE_STATES = {"pending", "running", "completed", "failed", "skipped"}
TOOL_ARGUMENT_SCHEMAS: dict[str, dict[str, type[Any]]] = {
    "artifact_read": {"artifact_id": str},
    "retrieval_search": {"query": str, "top_k": int},
}


@dataclass
class ExecutionNode:
    id: str
    kind: str
    depends_on: tuple[str, ...] = ()
    endpoint: str | None = None
    state: str = "pending"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["depends_on"] = list(self.depends_on)
        return value


@dataclass
class ExecutionGraph:
    graph_id: str
    nodes: list[ExecutionNode]
    maximum_nodes: int
    mode: str = "top-1"
    bounded: bool = True
    schema: str = "sparse-network-execution-graph.v1"

    def validate(self) -> None:
        if self.schema != "sparse-network-execution-graph.v1":
            raise RequestFailedError(f"Unsupported execution graph schema: {self.schema}")
        invocation_limits = {"top-1": 1, "top-2": 2, "mosa": 3}
        if self.mode not in invocation_limits or not self.bounded:
            raise RequestFailedError("Execution graph mode is unknown or unbounded")
        if not self.nodes or len(self.nodes) > self.maximum_nodes:
            raise RequestFailedError("Execution graph is empty or exceeds its node bound")
        by_id = {node.id: node for node in self.nodes}
        if len(by_id) != len(self.nodes):
            raise RequestFailedError("Execution graph node identifiers must be unique")
        if (
            sum(node.kind == "model_invoke" for node in self.nodes)
            > invocation_limits[self.mode]
        ):
            raise RequestFailedError(
                f"{self.mode} execution graph exceeds its model invocation bound"
            )
        for node in self.nodes:
            if node.kind not in GRAPH_NODE_KINDS:
                raise RequestFailedError(f"Unknown execution graph node kind: {node.kind}")
            if node.state not in GRAPH_NODE_STATES:
                raise RequestFailedError(f"Unknown execution graph node state: {node.state}")
            if node.metadata.get("expands_graph"):
                raise RequestFailedError(
                    "Model nodes cannot recursively expand the execution graph"
                )
            for dependency in node.depends_on:
                if dependency not in by_id:
                    raise RequestFailedError(
                        f"Execution graph dependency does not exist: {dependency}"
                    )

        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(node_id: str) -> None:
            if node_id in visiting:
                raise RequestFailedError("Execution graph contains a cycle")
            if node_id in visited:
                return
            visiting.add(node_id)
            for dependency in by_id[node_id].depends_on:
                visit(dependency)
            visiting.remove(node_id)
            visited.add(node_id)

        for node_id in by_id:
            visit(node_id)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": self.schema,
            "graph_id": self.graph_id,
            "mode": self.mode,
            "bounded": self.bounded,
            "maximum_nodes": self.maximum_nodes,
            "nodes": [node.to_dict() for node in self.nodes],
        }


@dataclass(frozen=True)
class VerificationCheck:
    name: str
    passed: bool
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VerificationReport:
    accepted: bool
    checks: tuple[VerificationCheck, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "checks": [check.to_dict() for check in self.checks],
            "passed_checks": [check.name for check in self.checks if check.passed],
            "failed_checks": [check.name for check in self.checks if not check.passed],
        }


@dataclass(frozen=True)
class ExecutionRequest:
    route: RouteRequest
    evidence_references: tuple[str, ...] = ()
    priority: int = 10
    timeout_seconds: float | None = None
    expected_contains: str | None = None
    response_schema: dict[str, Any] | None = None
    require_citations: bool = False
    tool_arguments: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class ExecutionResult:
    status: str
    decision: RouteDecision
    graph: ExecutionGraph
    verification: VerificationReport | None
    model_result: SmokeResult | None
    trace_path: Path
    error: str | None = None
    schema: str = "sparse-network-execution-result.v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "status": self.status,
            "decision": self.decision.to_dict(),
            "graph": self.graph.to_dict(),
            "verification": self.verification.to_dict() if self.verification else None,
            "model_result": self.model_result.to_dict() if self.model_result else None,
            "trace_path": str(self.trace_path),
            "error": self.error,
        }


def build_top1_graph(decision: RouteDecision, *, maximum_nodes: int) -> ExecutionGraph:
    route_node = ExecutionNode(
        id="route",
        kind="route",
        state="completed",
        metadata={"decision": decision.to_dict()},
    )
    if decision.rejected or decision.endpoint is None:
        nodes = [
            route_node,
            ExecutionNode(
                id="reject",
                kind="reject",
                depends_on=("route",),
                state="completed",
                metadata={"reason": decision.reason},
            ),
        ]
    else:
        terminal_kind = "human_review" if decision.human_review_required else "accept"
        nodes = [
            route_node,
            ExecutionNode(
                id="invoke",
                kind="model_invoke",
                depends_on=("route",),
                endpoint=decision.endpoint,
                metadata={
                    "routing_reason": decision.reason,
                    "lane": decision.lane,
                    "recursive_invocation_allowed": False,
                },
            ),
            ExecutionNode(id="verify", kind="verify", depends_on=("invoke",)),
            ExecutionNode(id="terminal", kind=terminal_kind, depends_on=("verify",)),
        ]
    graph = ExecutionGraph(
        graph_id=f"graph-{uuid4()}",
        nodes=nodes,
        maximum_nodes=maximum_nodes,
    )
    graph.validate()
    return graph


def _json_type_matches(value: Any, expected: str) -> bool:
    mappings: dict[str, type[Any] | tuple[type[Any], ...]] = {
        "object": dict,
        "array": list,
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "null": type(None),
    }
    expected_type = mappings.get(expected)
    if expected_type is None:
        return False
    if expected == "integer" and isinstance(value, bool):
        return False
    if expected == "number" and isinstance(value, bool):
        return False
    return isinstance(value, expected_type)


def validate_json_schema_subset(value: Any, schema: dict[str, Any], path: str = "$") -> list[str]:
    """Validate the bounded JSON-schema subset used by controller requests."""
    errors: list[str] = []
    expected_type = schema.get("type")
    if isinstance(expected_type, str) and not _json_type_matches(value, expected_type):
        return [f"{path} expected {expected_type}"]
    if isinstance(value, dict):
        required = schema.get("required", [])
        if isinstance(required, list):
            for key in required:
                if isinstance(key, str) and key not in value:
                    errors.append(f"{path}.{key} is required")
        properties = schema.get("properties", {})
        if isinstance(properties, dict):
            for key, child_schema in properties.items():
                if key in value and isinstance(child_schema, dict):
                    errors.extend(
                        validate_json_schema_subset(value[key], child_schema, f"{path}.{key}")
                    )
    elif isinstance(value, list) and isinstance(schema.get("items"), dict):
        for index, item in enumerate(value):
            errors.extend(validate_json_schema_subset(item, schema["items"], f"{path}[{index}]"))
    return errors


class DeterministicVerifier:
    def __init__(self, artifacts: ArtifactStore, manager: FleetManager) -> None:
        self.artifacts = artifacts
        self.manager = manager

    @staticmethod
    def verify_tool_arguments(arguments: dict[str, dict[str, Any]]) -> VerificationCheck:
        errors: list[str] = []
        for tool, values in arguments.items():
            schema = TOOL_ARGUMENT_SCHEMAS.get(tool)
            if schema is None:
                errors.append(f"{tool}: no argument schema")
                continue
            if not isinstance(values, dict):
                errors.append(f"{tool}: arguments must be an object")
                continue
            unknown = set(values) - set(schema)
            missing = set(schema) - set(values)
            if unknown:
                errors.append(f"{tool}: unknown arguments {sorted(unknown)}")
            if missing:
                errors.append(f"{tool}: missing arguments {sorted(missing)}")
            for name, expected in schema.items():
                if name in values and (
                    not isinstance(values[name], expected)
                    or (expected is int and isinstance(values[name], bool))
                ):
                    errors.append(f"{tool}.{name}: expected {expected.__name__}")
        return VerificationCheck(
            name="tool_argument_schema",
            passed=not errors,
            detail="valid" if not errors else "; ".join(errors),
        )

    def verify(
        self,
        request: ExecutionRequest,
        decision: RouteDecision,
        result: SmokeResult,
    ) -> VerificationReport:
        checks: list[VerificationCheck] = []
        try:
            result.envelope.validate()
            envelope_error = "valid"
            envelope_passed = True
        except RequestFailedError as exc:
            envelope_error = str(exc)
            envelope_passed = False
        checks.append(VerificationCheck("envelope_schema", envelope_passed, envelope_error))
        answer_passed = result.envelope.status == "answer" and bool(result.answer)
        checks.append(
            VerificationCheck(
                "answer_status",
                answer_passed,
                result.envelope.status if result.envelope.status else "missing status",
            )
        )
        try:
            if result.envelope.answer_reference is None:
                raise FileNotFoundError("answer reference is missing")
            self.artifacts.get(result.envelope.answer_reference, verify_content=True)
            artifact_passed, artifact_detail = True, "answer artifact exists and hash matches"
        except (FileNotFoundError, OSError, ValueError) as exc:
            artifact_passed, artifact_detail = False, str(exc)
        checks.append(VerificationCheck("answer_artifact", artifact_passed, artifact_detail))

        consumed = set(result.envelope.modalities_consumed)
        required = set(decision.required_modalities)
        checks.append(
            VerificationCheck(
                "modality_consistency",
                required.issubset(consumed),
                f"required={sorted(required)}, consumed={sorted(consumed)}",
            )
        )
        endpoint = self.manager.registry.get(result.envelope.endpoint)
        elapsed_ok = (
            request.timeout_seconds is None
            or result.envelope.resource_usage.elapsed_ms <= request.timeout_seconds * 1000
        )
        output_ok = result.envelope.resource_usage.output_tokens <= endpoint.max_output_tokens
        reserve_ok = (
            not self.manager.total_vram_bytes
            or result.envelope.resource_usage.peak_vram_bytes
            + self.manager.minimum_vram_reserve_bytes
            <= self.manager.total_vram_bytes
        )
        checks.append(
            VerificationCheck(
                "resource_budget",
                elapsed_ok and output_ok and reserve_ok,
                f"elapsed_ok={elapsed_ok}, output_ok={output_ok}, reserve_ok={reserve_ok}",
            )
        )
        if request.expected_contains is not None:
            checks.append(
                VerificationCheck(
                    "expected_content",
                    request.expected_contains in (result.answer or ""),
                    f"required substring={request.expected_contains!r}",
                )
            )
        if request.response_schema is not None:
            try:
                parsed = json.loads(result.answer or "")
                schema_errors = validate_json_schema_subset(parsed, request.response_schema)
            except json.JSONDecodeError as exc:
                schema_errors = [f"invalid JSON: {exc.msg}"]
            checks.append(
                VerificationCheck(
                    "response_json_schema",
                    not schema_errors,
                    "valid" if not schema_errors else "; ".join(schema_errors),
                )
            )
        if request.evidence_references:
            evidence_errors: list[str] = []
            for artifact_id in request.evidence_references:
                try:
                    self.artifacts.get(artifact_id, verify_content=True)
                except (FileNotFoundError, OSError, ValueError) as exc:
                    evidence_errors.append(str(exc))
                if request.require_citations and artifact_id not in (result.answer or ""):
                    evidence_errors.append(f"answer does not cite {artifact_id}")
            checks.append(
                VerificationCheck(
                    "source_references",
                    not evidence_errors,
                    "valid" if not evidence_errors else "; ".join(evidence_errors),
                )
            )
        if request.tool_arguments:
            checks.append(self.verify_tool_arguments(request.tool_arguments))
        return VerificationReport(
            accepted=all(check.passed for check in checks),
            checks=tuple(checks),
        )


class Top1Executor:
    def __init__(self, manager: FleetManager, router: StaticRouter) -> None:
        self.manager = manager
        self.router = router
        runs = manager.config.paths["runs"]
        if runs is None:
            raise ValueError("paths.runs must be configured")
        self.runs = runs
        self.verifier = DeterministicVerifier(manager.controller.artifacts, manager)

    def plan(self, decision: RouteDecision) -> ExecutionGraph:
        return build_top1_graph(decision, maximum_nodes=self.router.maximum_graph_nodes)

    def _trace_path(self, graph: ExecutionGraph) -> Path:
        return (self.runs / "execution-graphs" / f"{graph.graph_id}.json").resolve(strict=False)

    def _persist(
        self,
        graph: ExecutionGraph,
        decision: RouteDecision,
        *,
        status: str,
        verification: VerificationReport | None = None,
        error: str | None = None,
    ) -> Path:
        path = self._trace_path(graph)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": "sparse-network-execution-trace.v1",
            "updated_at": datetime.now(UTC).isoformat(),
            "status": status,
            "decision": decision.to_dict(),
            "graph": graph.to_dict(),
            "verification": verification.to_dict() if verification else None,
            "error": error,
        }
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, path)
        return path

    def run(self, request: ExecutionRequest) -> ExecutionResult:
        decision = self.router.route(request.route)
        graph = self.plan(decision)
        trace_path = self._persist(graph, decision, status="planned")
        self.manager.event_log.emit(
            event="route_decision",
            endpoint=decision.endpoint or "controller",
            details={
                "graph_id": graph.graph_id,
                "lane": decision.lane,
                "reason": decision.reason,
                "source": decision.source,
                "rejected": decision.rejected,
            },
        )
        if decision.rejected or decision.endpoint is None:
            return ExecutionResult(
                status="route_rejected",
                decision=decision,
                graph=graph,
                verification=None,
                model_result=None,
                trace_path=trace_path,
                error=decision.reason,
            )

        invoke = next(node for node in graph.nodes if node.id == "invoke")
        verify_node = next(node for node in graph.nodes if node.id == "verify")
        terminal = next(node for node in graph.nodes if node.id == "terminal")
        invoke.state = "running"
        self._persist(graph, decision, status="running")
        try:
            model_result = self.manager.submit(
                endpoint_id=decision.endpoint,
                prompt=request.route.runtime_prompt,
                original_prompt=request.route.prompt,
                images=request.route.images,
                evidence_references=request.evidence_references,
                priority=request.priority,
                timeout_seconds=request.timeout_seconds,
            )
        except Exception as exc:
            invoke.state = "failed"
            verify_node.state = "skipped"
            terminal.state = "skipped"
            error = f"{type(exc).__name__}: {exc}"
            trace_path = self._persist(graph, decision, status="failed", error=error)
            return ExecutionResult(
                status="failed",
                decision=decision,
                graph=graph,
                verification=None,
                model_result=None,
                trace_path=trace_path,
                error=error,
            )
        invoke.state = "completed"
        verify_node.state = "running"
        verification = self.verifier.verify(request, decision, model_result)
        verify_node.state = "completed" if verification.accepted else "failed"
        verification_value = {
            "required": decision.verification_required,
            **verification.to_dict(),
        }
        model_result = replace(
            model_result,
            envelope=replace(model_result.envelope, verification=verification_value),
        )
        if model_result.envelope.status == "cancelled":
            status = "cancelled"
            terminal.state = "skipped"
        elif not verification.accepted:
            status = "verification_failed"
            terminal.state = "skipped"
        elif decision.human_review_required:
            status = "needs_human_review"
            terminal.state = "pending"
        else:
            status = "accepted"
            terminal.state = "completed"
        trace_path = self._persist(graph, decision, status=status, verification=verification)
        self.manager.event_log.emit(
            event="execution_verified",
            endpoint=decision.endpoint,
            request_id=model_result.envelope.request_id,
            execution_id=model_result.envelope.execution_id,
            details={
                "graph_id": graph.graph_id,
                "accepted": verification.accepted,
                "status": status,
                "checks": verification.to_dict()["passed_checks"],
            },
        )
        return ExecutionResult(
            status=status,
            decision=decision,
            graph=graph,
            verification=verification,
            model_result=model_result,
            trace_path=trace_path,
        )
