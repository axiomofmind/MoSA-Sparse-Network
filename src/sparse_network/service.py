"""Long-lived controller service used by the local API and CLI."""

from __future__ import annotations

import copy
import json
import threading
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from time import monotonic, sleep
from typing import Any
from uuid import uuid4

import yaml

from .config import AppConfig
from .doctor import run_doctor
from .errors import ResourceAdmissionError
from .evaluation import evaluate_coordination_results
from .execution import ExecutionRequest, Top1Executor
from .fleet import FleetManager
from .models import ModelRegistry
from .routing import RouteRequest, StaticRouter
from .swap import SwapCoordinator, SwapRequest
from .usecases import UseCaseManager
from .validation import validate_endpoint_artifacts
from .workflows import WorkflowExecutor, WorkflowRequest

SENSITIVE_DASHBOARD_KEYS = {
    "answer",
    "base_path",
    "content",
    "environment",
    "error",
    "failure",
    "path",
    "prompt",
    "resolved_path",
    "sources",
    "text",
}


def _redact_dashboard(value: Any, *, key: str | None = None) -> Any:
    normalized_key = (key or "").lower()
    if normalized_key in SENSITIVE_DASHBOARD_KEYS or normalized_key.endswith(
        ("_path", "_paths", "_directory", "_dir")
    ):
        return "[redacted]"
    if isinstance(value, Path):
        return "[redacted]"
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        return {
            str(child_key): _redact_dashboard(child, key=str(child_key))
            for child_key, child in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_redact_dashboard(child) for child in value]
    return value


@dataclass
class ServiceRequest:
    id: str
    endpoint: str
    prompt: str
    state: str = "queued"
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    result: dict[str, Any] | None = None
    error: str | None = None
    cancellation: threading.Event = field(default_factory=threading.Event, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "sparse-network-service-request.v1",
            "id": self.id,
            "endpoint": self.endpoint,
            "prompt": self.prompt,
            "state": self.state,
            "created_at": self.created_at,
            "result": self.result,
            "error": self.error,
        }


@dataclass
class UseCaseJob:
    id: str
    template_id: str
    title: str
    actor: str
    state: str = "queued"
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    completed_at: str | None = None
    result: dict[str, Any] | None = None
    error_type: str | None = None
    error: str | None = None
    progress: dict[str, Any] | None = None
    parent_run_id: str | None = None
    cancellation: threading.Event = field(default_factory=threading.Event, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "sparse-network-use-case-job.v1",
            "id": self.id,
            "template_id": self.template_id,
            "title": self.title,
            "actor": self.actor,
            "state": self.state,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "result": self.result,
            "error_type": self.error_type,
            "error": self.error,
            "progress": self.progress,
            "parent_run_id": self.parent_run_id,
        }


class ControllerService:
    """Enforce controller invariants across every API and CLI operation."""

    CONFIG_ALLOWLIST = {
        "controller.request_timeout_seconds",
        "controller.telemetry_interval_seconds",
        "fleet.queue_capacity",
    }

    def __init__(self, config: AppConfig, registry: ModelRegistry) -> None:
        self.config = self._restore_active_profile(config)
        self.registry = registry
        self.fleet = FleetManager(self.config, registry)
        self.swap = self._new_swap_coordinator(self.fleet)
        self.use_cases = UseCaseManager(self.config, registry, self.fleet)
        self.requests: dict[str, ServiceRequest] = {}
        self.use_case_jobs: dict[str, UseCaseJob] = {}
        self.evaluations: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._preparation_lock = threading.Lock()
        self._profile_lock = threading.Lock()
        self._profile_plans: dict[str, dict[str, Any]] = {}
        self._switching = False
        self._dashboard_catalog_cache: dict[str, Any] | None = None

    @staticmethod
    def _new_swap_coordinator(fleet: FleetManager) -> SwapCoordinator:
        endpoint_ids = fleet.exclusive_swap_endpoint_ids
        return SwapCoordinator(
            fleet,
            endpoint_id=endpoint_ids[0] if endpoint_ids else "qwen38-27b",
        )

    @staticmethod
    def _restore_active_profile(config: AppConfig) -> AppConfig:
        runs = config.paths["runs"]
        if runs is None:
            return config
        state_path = runs / "active-profile.json"
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            roster_name = str(state["roster_file"])
            roster_kind = str(state.get("roster_kind", "builtin"))
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return config
        root = (
            config.root / "configs" / "rosters"
            if roster_kind == "builtin"
            else runs / "custom-profiles"
        ).resolve(strict=False)
        candidate = (root / Path(roster_name).name).resolve(strict=False)
        if not candidate.is_relative_to(root) or not candidate.is_file():
            return config
        data = copy.deepcopy(config.data)
        data.setdefault("fleet", {})["roster"] = str(candidate)
        return AppConfig(root=config.root, data=data, sources=config.sources)

    def start(self) -> dict[str, Any]:
        if self.fleet.residents_ready():
            return self.fleet.status()
        self.swap.recover_if_interrupted()
        if self.fleet.residents_ready():
            return self.fleet.status()
        return self.fleet.load_all()

    def submit(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._switching:
            raise ValueError("fleet profile switch is in progress")
        endpoint = str(payload.get("endpoint", ""))
        prompt = str(payload.get("prompt", ""))
        if endpoint not in self.fleet.entries:
            raise ValueError("endpoint must be a resident fleet endpoint")
        if not prompt.strip():
            raise ValueError("prompt must not be empty")
        record = ServiceRequest(id=f"service-request-{uuid4()}", endpoint=endpoint, prompt=prompt)
        with self._lock:
            self.requests[record.id] = record
        self.fleet.event_log.emit(
            event="service_request_created",
            endpoint=endpoint,
            request_id=record.id,
            details={"state": "queued"},
        )
        thread = threading.Thread(
            target=self._execute,
            args=(record, payload),
            name=f"service-{record.id}",
            daemon=True,
        )
        thread.start()
        return record.to_dict()

    def _execute(self, record: ServiceRequest, payload: dict[str, Any]) -> None:
        record.state = "running"
        self.fleet.event_log.emit(
            event="service_request_running",
            endpoint=record.endpoint,
            request_id=record.id,
            details={"state": record.state},
        )
        try:
            images = tuple(Path(str(value)) for value in payload.get("images", []))
            evidence = tuple(str(value) for value in payload.get("evidence_references", []))
            result = self.fleet.submit(
                endpoint_id=record.endpoint,
                prompt=record.prompt,
                images=images,
                evidence_references=evidence,
                cancellation=record.cancellation,
                timeout_seconds=(
                    float(payload["timeout_seconds"])
                    if payload.get("timeout_seconds") is not None
                    else None
                ),
                priority=int(payload.get("priority", 10)),
            )
            record.result = result.to_dict()
            if result.envelope.status in {"cancelled", "failed"}:
                record.state = result.envelope.status
            else:
                record.state = "completed"
            resource = result.envelope.resource_usage
            self.fleet.event_log.emit(
                event="resource_telemetry",
                endpoint=record.endpoint,
                request_id=record.id,
                execution_id=result.envelope.execution_id,
                details={
                    "elapsed_ms": resource.elapsed_ms,
                    "peak_vram_bytes": resource.peak_vram_bytes,
                    "peak_ram_bytes": resource.peak_ram_bytes,
                },
            )
            if images:
                self.fleet.event_log.emit(
                    event="vision_completed",
                    endpoint=record.endpoint,
                    request_id=record.id,
                    execution_id=result.envelope.execution_id,
                    details={"image_count": len(images), "status": result.envelope.status},
                )
            if result.envelope.verification:
                self.fleet.event_log.emit(
                    event="verification_completed",
                    endpoint=record.endpoint,
                    request_id=record.id,
                    execution_id=result.envelope.execution_id,
                    details=result.envelope.verification,
                )
        except BaseException as exc:
            record.error = str(exc)
            record.state = "cancelled" if record.cancellation.is_set() else "failed"
        self.fleet.event_log.emit(
            event="service_request_finished",
            endpoint=record.endpoint,
            request_id=record.id,
            details={"state": record.state, "error": record.error},
        )

    def request(self, request_id: str) -> dict[str, Any]:
        try:
            return self.requests[request_id].to_dict()
        except KeyError as exc:
            raise KeyError(f"Unknown request: {request_id}") from exc

    def cancel(
        self,
        request_id: str,
        *,
        confirmation: str,
        actor: str = "administrator",
    ) -> dict[str, Any]:
        expected = f"CANCEL {request_id}"
        if confirmation != expected:
            raise ValueError(f"request cancellation requires confirmation phrase: {expected}")
        record = self.requests.get(request_id)
        if record is None:
            raise KeyError(f"Unknown request: {request_id}")
        if record.state in {"completed", "failed", "cancelled"}:
            return record.to_dict()
        record.cancellation.set()
        self.fleet.event_log.emit(
            event="service_request_cancel_requested",
            endpoint=record.endpoint,
            request_id=record.id,
            details={"actor": actor},
        )
        return record.to_dict()

    def trace(self, request_id: str) -> dict[str, Any]:
        record = self.request(request_id)
        events = [
            event
            for event in self.fleet.event_log.read()
            if event.get("request_id") == request_id
            or event.get("request_id") == (record.get("result") or {}).get("envelope", {}).get(
                "request_id"
            )
        ]
        return {"schema": "sparse-network-request-trace.v1", "request": record, "events": events}

    def fleet_operation(
        self,
        operation: str,
        endpoint: str | None = None,
        *,
        confirmation: str = "",
        actor: str = "administrator",
    ) -> dict[str, Any]:
        if operation != "smoke":
            expected = f"CONFIRM {operation} {endpoint or 'fleet'}"
            if confirmation != expected:
                raise ValueError(f"fleet operation requires confirmation phrase: {expected}")
        self.fleet.event_log.emit(
            event="operator_action_requested",
            endpoint=endpoint or "controller",
            details={"actor": actor, "operation": operation},
        )
        if operation == "start":
            return self.start()
        if operation == "drain" and endpoint:
            self.fleet.drain_endpoint(endpoint)
            return self.fleet.status()
        if operation == "unload" and endpoint:
            self.fleet.unload_endpoint(endpoint)
            return self.fleet.status()
        if operation == "unload_all":
            self.fleet.pause_admission(reason="API unload all")
            return self.fleet.unload_all()
        if operation in {"reload", "restore"}:
            self.fleet.pause_admission(reason=f"API {operation}")
            try:
                self.fleet.unload_all()
                fleet_result = self.fleet.load_all()
            finally:
                healthy = self.fleet.residents_ready()
                if healthy:
                    self.fleet.resume_admission(reason=f"API {operation} complete")
            return fleet_result
        if operation == "quarantine" and endpoint:
            self.fleet.quarantine_endpoint(endpoint)
            return self.fleet.status()
        if operation == "smoke" and endpoint:
            smoke_result = self.fleet.submit(
                endpoint_id=endpoint, prompt="Reply with exactly: OK"
            )
            return smoke_result.to_dict()
        raise ValueError(f"Unsupported or incomplete fleet operation: {operation}")

    def create_evaluation(
        self, payload: dict[str, Any], *, actor: str = "administrator"
    ) -> dict[str, Any]:
        evaluation_id = f"evaluation-{uuid4()}"
        baselines = payload.get("baselines", [])
        if not isinstance(baselines, list):
            raise ValueError("evaluation baselines must be a list")
        normalized_baselines: list[dict[str, Any]] = []
        for baseline in baselines:
            if not isinstance(baseline, dict) or not str(baseline.get("id", "")).strip():
                raise ValueError("each evaluation baseline needs an id")
            metrics = baseline.get("metrics", {})
            if not isinstance(metrics, dict):
                raise ValueError("baseline metrics must be an object")
            normalized_metrics: dict[str, float] = {}
            for name, metric in metrics.items():
                if not isinstance(metric, (int, float)):
                    raise ValueError("evaluation metrics must be numeric")
                normalized_metrics[str(name)] = float(metric)
            normalized_baselines.append(
                {"id": str(baseline["id"]), "metrics": normalized_metrics}
            )
        coordination_report: dict[str, Any] | None = None
        if str(payload.get("suite", "")) == "coordination" and not normalized_baselines:
            runs = self.config.paths["runs"]
            if runs is None:
                raise ValueError("paths.runs must be configured")
            trace_root = runs / "execution-graphs"
            requested_ids = payload.get("workflow_trace_ids", [])
            if not isinstance(requested_ids, list):
                raise ValueError("workflow_trace_ids must be a list")
            if requested_ids:
                trace_paths = []
                for trace_id in requested_ids:
                    name = str(trace_id)
                    if Path(name).name != name or not name.endswith(".json"):
                        raise ValueError("workflow trace IDs must be JSON filenames")
                    trace_paths.append((trace_root / name).resolve(strict=True))
            else:
                trace_paths = sorted(
                    trace_root.glob("*.json"),
                    key=lambda path: path.stat().st_mtime_ns,
                    reverse=True,
                )[:500]
            traces: list[dict[str, Any]] = []
            for trace_path in trace_paths:
                if not trace_path.is_relative_to(trace_root.resolve(strict=False)):
                    raise ValueError("workflow trace escapes the configured run directory")
                trace = json.loads(trace_path.read_text(encoding="utf-8"))
                supported = isinstance(trace, dict) and trace.get("schema") == (
                    "sparse-network-workflow-trace.v1"
                )
                if not supported:
                    if requested_ids:
                        raise ValueError(f"unsupported workflow trace: {trace_path.name}")
                    continue
                traces.append(trace)
            coordination_report = evaluate_coordination_results(traces)
            normalized_baselines.append(
                {
                    "id": "coordination-current",
                    "metrics": coordination_report["metrics"],
                }
            )
        value = {
            "schema": "sparse-network-evaluation.v1",
            "id": evaluation_id,
            "state": "created",
            "suite": str(payload.get("suite", "")),
            "actor": actor,
            "created_at": datetime.now(UTC).isoformat(),
            "baselines": normalized_baselines,
            "result": {"status": "recorded", "baseline_count": len(normalized_baselines)},
            **(
                {"coordination": coordination_report}
                if coordination_report is not None
                else {}
            ),
        }
        self.evaluations[evaluation_id] = value
        runs = self.config.paths["runs"]
        if runs is None:
            raise ValueError("paths.runs must be configured")
        manifest = runs / f"{evaluation_id}.json"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        self.fleet.event_log.emit(
            event="evaluation_created",
            endpoint="controller",
            details={"id": evaluation_id, "actor": actor},
        )
        return value

    def antidoom_status(self) -> dict[str, Any]:
        return {
            "schema": "sparse-network-antidoom-status.v1",
            "runtime_detector": self.fleet.controller.repetition_policy.to_dict(),
            "datasets": [],
            "training_runs": [],
            "note": "Offline training requires an explicitly created maintenance-window run.",
        }

    def record_tool_event(
        self,
        *,
        request_id: str,
        tool: str,
        state: str,
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Internal hook for the future allowlisted tool runner."""

        if state not in {"proposed", "authorized", "running", "completed", "failed"}:
            raise ValueError("invalid tool event state")
        return self.fleet.event_log.emit(
            event="tool_event",
            endpoint="controller",
            request_id=request_id,
            details={"tool": tool, "state": state, **(details or {})},
        )

    def config_view(self) -> dict[str, Any]:
        return {
            "schema": "sparse-network-config.v1",
            "sources": [str(path) for path in self.config.sources],
            "data": copy.deepcopy(self.config.data),
            "mutable": sorted(self.CONFIG_ALLOWLIST),
        }

    def update_config(
        self,
        changes: dict[str, Any],
        *,
        confirmation: str = "",
        actor: str = "administrator",
    ) -> dict[str, Any]:
        if confirmation != "APPLY CONFIG":
            raise ValueError("configuration update requires confirmation phrase: APPLY CONFIG")
        invalid = set(changes) - self.CONFIG_ALLOWLIST
        if invalid:
            raise ValueError(f"Configuration keys are not mutable: {sorted(invalid)}")
        candidate = copy.deepcopy(self.config.data)
        for dotted, value in changes.items():
            section, key = dotted.split(".", 1)
            candidate.setdefault(section, {})[key] = value
        if int(candidate.get("fleet", {}).get("queue_capacity", 0)) <= 0:
            raise ValueError("fleet.queue_capacity must be positive")
        for name in ("request_timeout_seconds", "telemetry_interval_seconds"):
            if float(candidate.get("controller", {}).get(name, 0)) <= 0:
                raise ValueError(f"controller.{name} must be positive")
        runs = self.config.paths["runs"]
        if runs is None:
            raise ValueError("paths.runs must be configured")
        update_id = f"config-update-{uuid4()}"
        destination = runs / "config-updates" / f"{update_id}.yaml"
        destination.parent.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema": "sparse-network-config-update.v1",
            "id": update_id,
            "created_at": datetime.now(UTC).isoformat(),
            "actor": actor,
            "changes": changes,
        }
        destination.write_text(yaml.safe_dump(manifest, sort_keys=True), encoding="utf-8")
        self.config.data.clear()
        self.config.data.update(candidate)
        self.fleet.queue_capacity = int(candidate["fleet"]["queue_capacity"])
        self.fleet.event_log.emit(
            event="configuration_updated",
            endpoint="controller",
            details={"keys": sorted(changes), "update_id": update_id, "actor": actor},
        )
        return manifest

    def state_snapshot(self) -> dict[str, Any]:
        return {
            "schema": "sparse-network-controller-snapshot.v1",
            "fleet": self.fleet.status(),
            "requests": [record.to_dict() for record in self.requests.values()],
            "evaluations": list(self.evaluations.values()),
            "last_event_id": max(
                (int(event.get("event_id", 0)) for event in self.fleet.event_log.read()), default=0
            ),
        }

    def _model_catalog(self) -> list[dict[str, Any]]:
        values: list[dict[str, Any]] = []
        for endpoint in self.registry.all():
            artifact_state = "not_required"
            artifact_error = None
            artifact_sizes: dict[str, int] = {}
            if endpoint.source.type != "builtin":
                try:
                    validation = validate_endpoint_artifacts(
                        endpoint, self.config.paths["model_cache"]
                    )
                    artifact_state = "validated"
                    artifact_sizes = {
                        artifact.name: artifact.size_bytes
                        for artifact in validation.artifacts
                    }
                except Exception as exc:
                    artifact_state = "missing_or_invalid"
                    # Validation errors commonly contain resolved cache paths. The
                    # dashboard needs a stable category, not host-specific details.
                    artifact_error = type(exc).__name__
            values.append(
                {
                    "id": endpoint.id,
                    "display_name": endpoint.display_name,
                    "role": endpoint.role,
                    "family": endpoint.family,
                    "source": {
                        "type": endpoint.source.type,
                        "repository": (
                            endpoint.source.repository
                            or ",".join(
                                sorted(
                                    {
                                        artifact.repository
                                        for artifact in endpoint.source.artifacts.values()
                                    }
                                )
                            )
                            or None
                        ),
                        "revision": endpoint.model_revision,
                        "files": {
                            name: Path(filename).name
                            for name, filename in (
                                endpoint.source.files
                                or {
                                    artifact_name: artifact.path
                                    for artifact_name, artifact in (
                                        endpoint.source.artifacts.items()
                                    )
                                }
                            ).items()
                        },
                    },
                    "runtime": {
                        "adapter": endpoint.adapter,
                        "quantization": endpoint.runtime.get("quantization"),
                        "artifact_class": endpoint.runtime.get("artifact_class"),
                        "gpu_layers": endpoint.runtime.get("gpu_layers"),
                    },
                    "modalities": list(endpoint.modalities),
                    "capabilities": list(endpoint.capabilities),
                    "context_size": endpoint.context_size,
                    "max_output_tokens": endpoint.max_output_tokens,
                    "max_input_tokens": endpoint.max_input_tokens,
                    "admission": copy.deepcopy(endpoint.admission),
                    "license": copy.deepcopy(endpoint.license),
                    "proposed_budget": copy.deepcopy(endpoint.proposed_budget),
                    "artifact_state": artifact_state,
                    "artifact_error": artifact_error,
                    "artifact_sizes": artifact_sizes,
                }
            )
        return values

    def _profile_catalog(self, models: list[dict[str, Any]]) -> dict[str, Any]:
        model_by_id = {str(model["id"]): model for model in models}
        profiles: list[dict[str, Any]] = []
        roster_root = self.config.root / "configs" / "rosters"
        roster_paths = list(roster_root.glob("*.yaml"))
        runs = self.config.paths["runs"]
        if runs is not None:
            roster_paths.extend((runs / "custom-profiles").glob("*.yaml"))
        for roster_path in sorted(roster_paths, key=lambda path: path.name):
            try:
                roster = yaml.safe_load(roster_path.read_text(encoding="utf-8")) or {}
            except (OSError, yaml.YAMLError):
                continue
            if not isinstance(roster, dict) or not isinstance(roster.get("resident"), list):
                continue
            hardware_id = str(roster.get("hardware_profile", ""))
            hardware_path = self.config.root / "configs" / "hardware" / f"{hardware_id}.yaml"
            try:
                hardware = yaml.safe_load(hardware_path.read_text(encoding="utf-8")) or {}
            except (OSError, yaml.YAMLError):
                hardware = {}
            residents = [str(value) for value in roster.get("resident", [])]
            optional = [str(value) for value in roster.get("optional", [])]
            exclusive = [str(value) for value in roster.get("exclusive_swap", [])]
            proposed_vram = sum(
                int(model_by_id.get(endpoint_id, {}).get("proposed_budget", {}).get(
                    "incremental_vram_bytes", 0
                ))
                for endpoint_id in residents
            )
            proposed_kv = sum(
                int(model_by_id.get(endpoint_id, {}).get("proposed_budget", {}).get(
                    "kv_cache_bytes", 0
                ))
                for endpoint_id in residents
            )
            proposed_ram = sum(
                int(model_by_id.get(endpoint_id, {}).get("proposed_budget", {}).get(
                    "process_ram_bytes", 0
                ))
                for endpoint_id in residents
            )
            proposed_workspace = sum(
                int(model_by_id.get(endpoint_id, {}).get("proposed_budget", {}).get(
                    "request_workspace_bytes", 0
                ))
                for endpoint_id in residents
            )
            missing = [
                endpoint_id
                for endpoint_id in (*residents, *exclusive)
                if model_by_id.get(endpoint_id, {}).get("artifact_state")
                not in {"validated", "not_required"}
            ]
            profiles.append(
                {
                    "id": str(roster.get("id", roster_path.stem)),
                    "display_name": str(
                        roster.get("display_name", roster.get("id", roster_path.stem))
                    ),
                    "variant": str(roster.get("variant", "unspecified")),
                    "selection_key": (
                        hardware_id.removeprefix("nvidia-").removesuffix("gb")
                        if hardware_id.startswith("nvidia-")
                        else hardware_id
                    ),
                    "hardware_profile": hardware_id,
                    "validation_state": str(
                        roster.get("validation_state", "unspecified")
                    ),
                    "active": str(roster.get("id", roster_path.stem)) == self.fleet.roster_id,
                    "resident": residents,
                    "optional": optional,
                    "exclusive_swap": exclusive,
                    "requirements": copy.deepcopy(hardware.get("requirements", {})),
                    "scheduler": copy.deepcopy(hardware.get("scheduler", {})),
                    "budgets": copy.deepcopy(hardware.get("budgets", {})),
                    "proposed_vram_bytes": proposed_vram,
                    "proposed_kv_cache_bytes": proposed_kv,
                    "proposed_ram_bytes": proposed_ram,
                    "proposed_workspace_bytes": proposed_workspace,
                    "missing_or_invalid": missing,
                }
            )
        total_vram = self.fleet.total_vram_bytes
        if total_vram >= 30 * 1024**3:
            recommendation = "32"
        elif total_vram >= 22 * 1024**3:
            recommendation = "24"
        elif total_vram >= 14 * 1024**3:
            recommendation = "16"
        else:
            recommendation = "mock"
        return {
            "schema": "sparse-network-profile-catalog.v1",
            "recommended_selection": recommendation,
            "profiles": profiles,
            "custom": {
                "id": "custom-preview",
                "selection_key": "custom",
                "validation_state": "preview_only",
                "active": False,
            },
        }

    def dashboard_catalog(self) -> dict[str, Any]:
        if self._dashboard_catalog_cache is None:
            models = self._model_catalog()
            self._dashboard_catalog_cache = {
                "models": models,
                "profiles": self._profile_catalog(models),
            }
        return copy.deepcopy(self._dashboard_catalog_cache)

    def _find_roster(self, profile_id: str) -> tuple[Path, dict[str, Any], str]:
        runs = self.config.paths["runs"]
        roots: list[tuple[Path, str]] = [
            (self.config.root / "configs" / "rosters", "builtin")
        ]
        if runs is not None:
            roots.append((runs / "custom-profiles", "custom"))
        for root, kind in roots:
            if not root.exists():
                continue
            for roster_path in sorted(root.glob("*.yaml")):
                try:
                    roster = yaml.safe_load(roster_path.read_text(encoding="utf-8")) or {}
                except (OSError, yaml.YAMLError):
                    continue
                if isinstance(roster, dict) and str(
                    roster.get("id", roster_path.stem)
                ) == profile_id:
                    return roster_path.resolve(strict=True), roster, kind
        raise KeyError(f"Unknown hardware profile roster: {profile_id}")

    def _register_custom_profile(self, value: Any) -> str:
        if not isinstance(value, dict):
            raise ValueError("custom_profile must be an object")
        if value.get("schema") != "sparse-network-custom-roster.v1":
            raise ValueError("custom profile needs schema sparse-network-custom-roster.v1")
        profile_id = str(value.get("id", ""))
        if not profile_id.startswith("custom-") or not profile_id.replace("-", "").isalnum():
            raise ValueError("custom profile id must start with custom- and be alphanumeric")
        allowed = {
            "schema",
            "id",
            "hardware_profile",
            "validation_state",
            "resident",
            "optional",
            "exclusive_swap",
            "cpu",
        }
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"unknown custom profile keys: {sorted(unknown)}")
        hardware_id = str(value.get("hardware_profile", ""))
        hardware_path = self.config.root / "configs" / "hardware" / f"{hardware_id}.yaml"
        if not hardware_path.is_file():
            raise ValueError(f"unknown hardware profile: {hardware_id}")
        residents = value.get("resident")
        if not isinstance(residents, list) or not residents:
            raise ValueError("custom profile resident must be a non-empty list")
        for section in ("resident", "optional", "exclusive_swap", "cpu"):
            endpoints = value.get(section, [])
            if not isinstance(endpoints, list) or not all(
                isinstance(endpoint, str) for endpoint in endpoints
            ):
                raise ValueError(f"custom profile {section} must be a list of endpoint ids")
            for endpoint_id in endpoints:
                self.registry.get(endpoint_id)
        runs = self.config.paths["runs"]
        if runs is None:
            raise ValueError("paths.runs must be configured")
        destination = runs / "custom-profiles" / f"{profile_id}.yaml"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            existing = yaml.safe_load(destination.read_text(encoding="utf-8"))
            if existing != value:
                raise FileExistsError("custom profile ids are immutable")
        else:
            destination.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
        self._dashboard_catalog_cache = None
        return profile_id

    def plan_profile_switch(
        self, payload: dict[str, Any], *, actor: str = "administrator"
    ) -> dict[str, Any]:
        profile_id = (
            self._register_custom_profile(payload.get("custom_profile"))
            if payload.get("custom_profile") is not None
            else str(payload.get("profile_id", ""))
        )
        roster_path, roster, roster_kind = self._find_roster(profile_id)
        residents = [str(value) for value in roster.get("resident", [])]
        exclusive = [str(value) for value in roster.get("exclusive_swap", [])]
        if not residents:
            raise ValueError("profile roster must contain resident endpoints")
        hardware_id = str(roster.get("hardware_profile", ""))
        hardware_path = self.config.root / "configs" / "hardware" / f"{hardware_id}.yaml"
        hardware = yaml.safe_load(hardware_path.read_text(encoding="utf-8")) or {}
        if not isinstance(hardware, dict):
            raise ValueError("hardware profile must be an object")
        requirements = hardware.get("requirements", {})
        budgets = hardware.get("budgets", {})
        if not isinstance(requirements, dict) or not isinstance(budgets, dict):
            raise ValueError("hardware requirements and budgets must be objects")

        invalid_artifacts: list[str] = []
        runtime_adapters: set[str] = set()
        required_license_reviews: list[str] = []
        proposed_vram = 0
        proposed_ram = 0
        proposed_kv = 0
        proposed_workspace = 0
        for endpoint_id in (*residents, *exclusive):
            endpoint = self.registry.get(endpoint_id)
            runtime_adapters.add(endpoint.adapter)
            if bool(endpoint.license.get("review_required", False)):
                required_license_reviews.append(endpoint_id)
            if endpoint_id in residents:
                proposed_vram += int(
                    endpoint.proposed_budget.get("incremental_vram_bytes", 0)
                )
                proposed_ram += int(endpoint.proposed_budget.get("process_ram_bytes", 0))
                proposed_kv += int(endpoint.proposed_budget.get("kv_cache_bytes", 0))
                proposed_workspace += int(
                    endpoint.proposed_budget.get("request_workspace_bytes", 0)
                )
            try:
                validate_endpoint_artifacts(endpoint, self.config.paths["model_cache"])
            except Exception:
                invalid_artifacts.append(endpoint_id)

        diagnostics = run_doctor(self.config, self.registry)
        unavailable_runtimes = sorted(
            adapter
            for adapter in runtime_adapters
            if adapter != "mock"
            and not bool(diagnostics.get("runtimes", {}).get(adapter, {}).get("available"))
        )
        total_vram = self.fleet.total_vram_bytes
        minimum_vram = int(float(requirements.get("minimum_usable_vram_gb", 0)) * 1024**3)
        reserve = int(float(budgets.get("minimum_vram_reserve_gb", 0)) * 1024**3)
        kv_budget = int(float(budgets.get("total_kv_cache_gb", 0)) * 1024**3)
        total_ram = int(diagnostics.get("memory", {}).get("total_bytes", 0))
        required_ram = int(
            float(requirements.get("recommended_system_ram_gb", 0)) * 1024**3
        )
        backend = str(requirements.get("backend", ""))
        resource_errors: list[str] = []
        if backend != "mock" and total_vram < minimum_vram:
            resource_errors.append("detected accelerator memory is below the profile minimum")
        if total_vram and proposed_vram + reserve > total_vram:
            resource_errors.append("resident VRAM plus reserve exceeds detected capacity")
        if proposed_kv > kv_budget:
            resource_errors.append("proposed KV caches exceed the profile KV budget")
        if total_ram < max(required_ram, proposed_ram + 4 * 1024**3):
            resource_errors.append("system RAM cannot preserve the required margin")

        current = set(self.fleet.entries)
        target = set(residents)
        current_roster = yaml.safe_load(
            self.fleet.roster_path.read_text(encoding="utf-8")
        ) or {}
        current_exclusive = set(
            current_roster.get("exclusive_swap", [])
            if isinstance(current_roster, dict)
            else []
        )
        plan_id = f"profile-plan-{uuid4()}"
        acknowledged = bool(payload.get("acknowledge_licenses", False))
        valid = not invalid_artifacts and not unavailable_runtimes and not resource_errors
        plan = {
            "schema": "sparse-network-profile-switch-plan.v1",
            "id": plan_id,
            "actor": actor,
            "created_at": datetime.now(UTC).isoformat(),
            "profile_id": profile_id,
            "hardware_profile": hardware_id,
            "validation_state": str(roster.get("validation_state", "unspecified")),
            "current_profile": self.fleet.roster_id,
            "resident": residents,
            "exclusive_swap": exclusive,
            "additions": sorted(target - current),
            "removals": sorted(current - target),
            "qwen_quant_change": {
                "from": sorted(value for value in current_exclusive if "qwen38" in value),
                "to": sorted(value for value in exclusive if "qwen38" in value),
            },
            "resources": {
                "detected_vram_bytes": total_vram,
                "proposed_vram_bytes": proposed_vram,
                "proposed_ram_bytes": proposed_ram,
                "proposed_kv_cache_bytes": proposed_kv,
                "proposed_workspace_bytes": proposed_workspace,
                "required_reserve_bytes": reserve,
                "detected_system_ram_bytes": total_ram,
                "recommended_system_ram_bytes": required_ram,
            },
            "queue_impact": {
                "queued": self.fleet.status()["queue_depth"],
                "active": self.fleet.status()["active_requests"],
                "new_admission_will_pause": True,
            },
            "validation": {
                "schema": True,
                "doctor": not unavailable_runtimes,
                "artifacts_and_hashes": not invalid_artifacts,
                "resources": not resource_errors,
                "invalid_artifacts": invalid_artifacts,
                "unavailable_runtimes": unavailable_runtimes,
                "resource_errors": resource_errors,
                "license_review_endpoints": sorted(set(required_license_reviews)),
                "licenses_acknowledged": acknowledged,
            },
            "valid": valid,
            "ready_to_apply": valid and (acknowledged or not required_license_reviews),
            "confirmation_phrase": f"APPLY {profile_id}",
            "expires_in_seconds": 600,
        }
        self._profile_plans[plan_id] = {
            **copy.deepcopy(plan),
            "roster_path": roster_path,
            "roster_file": roster_path.name,
            "roster_kind": roster_kind,
            "expires_at_monotonic": monotonic() + 600,
        }
        self.fleet.event_log.emit(
            event="profile_switch_planned",
            endpoint="controller",
            details={"actor": actor, "plan_id": plan_id, "profile_id": profile_id},
        )
        return plan

    def _persist_active_profile(self, plan: dict[str, Any]) -> None:
        runs = self.config.paths["runs"]
        if runs is None:
            raise ValueError("paths.runs must be configured")
        value = {
            "schema": "sparse-network-active-profile.v1",
            "profile_id": plan["profile_id"],
            "roster_file": plan["roster_file"],
            "roster_kind": plan["roster_kind"],
            "updated_at": datetime.now(UTC).isoformat(),
        }
        destination = runs / "active-profile.json"
        temporary = destination.with_suffix(".json.tmp")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        temporary.replace(destination)

    def apply_profile_switch(
        self, payload: dict[str, Any], *, actor: str = "administrator"
    ) -> dict[str, Any]:
        plan_id = str(payload.get("plan_id", ""))
        try:
            plan = self._profile_plans[plan_id]
        except KeyError as exc:
            raise KeyError(f"Unknown profile switch plan: {plan_id}") from exc
        if monotonic() > float(plan["expires_at_monotonic"]):
            raise ValueError("profile switch plan has expired")
        if payload.get("confirmation") != plan["confirmation_phrase"]:
            raise ValueError(
                f"profile switch requires confirmation phrase: {plan['confirmation_phrase']}"
            )
        if not plan["valid"]:
            raise ResourceAdmissionError("profile switch plan did not pass validation")
        license_reviews = plan["validation"]["license_review_endpoints"]
        if license_reviews and not bool(payload.get("acknowledge_licenses", False)):
            raise ValueError("profile switch requires explicit license acknowledgement")

        if not self._profile_lock.acquire(blocking=False):
            raise ValueError("another profile switch is already running")
        old_fleet = self.fleet
        old_config = self.config
        builtin_rosters = (self.config.root / "configs" / "rosters").resolve(strict=False)
        old_roster_kind = (
            "builtin"
            if old_fleet.roster_path.is_relative_to(builtin_rosters)
            else "custom"
        )
        previous_profile = {
            "profile_id": old_fleet.roster_id,
            "roster_file": old_fleet.roster_path.name,
            "roster_kind": old_roster_kind,
        }
        candidate_fleet: FleetManager | None = None
        old_shutdown = False
        switch_manifest: Path | None = None
        switch_record: dict[str, Any] | None = None
        self._switching = True
        try:
            if not old_fleet.residents_ready():
                raise ValueError("current fleet must be healthy before profile switching")
            for endpoint_id in (*plan["resident"], *plan["exclusive_swap"]):
                validate_endpoint_artifacts(
                    self.registry.get(endpoint_id), self.config.paths["model_cache"]
                )
            old_fleet.pause_admission(reason=f"profile switch {plan_id}")
            deadline = monotonic() + float(
                self.config.data.get("controller", {}).get("request_timeout_seconds", 300)
            )
            while monotonic() < deadline:
                status = old_fleet.status()
                if status["queue_depth"] == 0 and status["active_requests"] == 0:
                    break
                sleep(0.05)
            else:
                old_fleet.resume_admission(reason="profile switch drain timed out")
                raise TimeoutError("timed out draining active work for profile switch")

            runs = self.config.paths["runs"]
            if runs is None:
                raise ValueError("paths.runs must be configured")
            switch_id = f"profile-switch-{uuid4()}"
            switch_manifest = runs / "profile-switches" / f"{switch_id}.json"
            switch_manifest.parent.mkdir(parents=True, exist_ok=True)
            switch_record = {
                "schema": "sparse-network-profile-switch.v1",
                "id": switch_id,
                "plan_id": plan_id,
                "actor": actor,
                "created_at": datetime.now(UTC).isoformat(),
                "previous_profile": old_fleet.roster_id,
                "selected_profile": plan["profile_id"],
                "request_state": [record.to_dict() for record in self.requests.values()],
                "evidence_artifact_ids": [
                    record.id for record in old_fleet.controller.artifacts.list_records()
                ],
                "status": "switching",
            }
            switch_manifest.write_text(
                json.dumps(switch_record, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            old_fleet.event_log.emit(
                event="profile_switch_started",
                endpoint="controller",
                details={"actor": actor, "switch_id": switch_id, "plan_id": plan_id},
            )
            old_fleet.shutdown()
            old_shutdown = True

            candidate_data = copy.deepcopy(old_config.data)
            candidate_data.setdefault("fleet", {})["roster"] = str(plan["roster_path"])
            candidate_config = AppConfig(
                root=old_config.root,
                data=candidate_data,
                sources=old_config.sources,
            )
            candidate_fleet = FleetManager(candidate_config, self.registry)
            candidate_fleet.load_all()
            smoke_results: dict[str, str] = {}
            for endpoint_id in candidate_fleet.entries:
                smoke = candidate_fleet.submit(
                    endpoint_id=endpoint_id,
                    prompt="Reply with exactly: OK",
                    timeout_seconds=float(
                        candidate_config.data.get("controller", {}).get(
                            "request_timeout_seconds", 300
                        )
                    ),
                )
                if smoke.envelope.status != "answer":
                    raise RuntimeError(f"profile smoke failed for {endpoint_id}")
                smoke_results[endpoint_id] = smoke.envelope.status
            self.config = candidate_config
            self.fleet = candidate_fleet
            self.swap = self._new_swap_coordinator(candidate_fleet)
            self.use_cases = UseCaseManager(candidate_config, self.registry, candidate_fleet)
            self._dashboard_catalog_cache = None
            self._persist_active_profile(plan)
            candidate_fleet.resume_admission(reason=f"profile switch {switch_id} complete")
            switch_record.update(
                {
                    "status": "completed",
                    "completed_at": datetime.now(UTC).isoformat(),
                    "smoke_results": smoke_results,
                }
            )
            switch_manifest.write_text(
                json.dumps(switch_record, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            candidate_fleet.event_log.emit(
                event="profile_switch_completed",
                endpoint="controller",
                details={"actor": actor, "switch_id": switch_id, "profile_id": plan["profile_id"]},
            )
            self._profile_plans.pop(plan_id, None)
            return {
                "schema": "sparse-network-profile-switch-result.v1",
                "id": switch_id,
                "status": "completed",
                "profile_id": plan["profile_id"],
                "fleet": candidate_fleet.status(),
                "smoke_results": smoke_results,
            }
        except Exception as switch_error:
            if candidate_fleet is not None:
                with suppress(Exception):
                    candidate_fleet.shutdown()
            rollback_error: Exception | None = None
            if old_shutdown:
                rollback_fleet = FleetManager(old_config, self.registry)
                try:
                    rollback_fleet.load_all()
                except Exception as exc:
                    rollback_error = exc
                self.fleet = rollback_fleet
                self.swap = self._new_swap_coordinator(rollback_fleet)
                self.use_cases = UseCaseManager(old_config, self.registry, rollback_fleet)
            else:
                old_fleet.resume_admission(reason="profile switch validation failed")
                self.fleet = old_fleet
            self.config = old_config
            self._dashboard_catalog_cache = None
            if rollback_error is None:
                self._persist_active_profile(previous_profile)
            self.fleet.event_log.emit(
                event=(
                    "profile_switch_rollback_failed"
                    if rollback_error is not None
                    else "profile_switch_rolled_back"
                ),
                endpoint="controller",
                details={
                    "actor": actor,
                    "plan_id": plan_id,
                    "error_type": type(switch_error).__name__,
                    "rollback_error_type": (
                        type(rollback_error).__name__ if rollback_error is not None else None
                    ),
                },
            )
            if switch_record is not None and switch_manifest is not None:
                switch_record.update(
                    {
                        "status": (
                            "rollback_failed"
                            if rollback_error is not None
                            else "rolled_back"
                        ),
                        "completed_at": datetime.now(UTC).isoformat(),
                        "error_type": type(switch_error).__name__,
                        "rollback_error_type": (
                            type(rollback_error).__name__
                            if rollback_error is not None
                            else None
                        ),
                    }
                )
                switch_manifest.write_text(
                    json.dumps(switch_record, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            if rollback_error is not None:
                raise RuntimeError(
                    "profile switch failed and the previous fleet could not be restored: "
                    f"{type(switch_error).__name__}; {type(rollback_error).__name__}"
                ) from rollback_error
            raise
        finally:
            self._switching = False
            self._profile_lock.release()

    def _recent_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        runs = self.config.paths["runs"]
        if runs is None or not runs.exists():
            return []
        values: list[dict[str, Any]] = []
        candidates = sorted(
            runs.glob("*.json"), key=lambda path: path.stat().st_mtime_ns, reverse=True
        )[:limit]
        for path in candidates:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict):
                continue
            values.append(
                {
                    "id": path.name,
                    "schema": payload.get("schema"),
                    "passed": payload.get("passed"),
                    "status": payload.get("status"),
                    "updated_at": datetime.fromtimestamp(
                        path.stat().st_mtime, tz=UTC
                    ).isoformat(),
                    "summary": _redact_dashboard(
                        {
                            key: payload.get(key)
                            for key in (
                                "elapsed_ms",
                                "cycles",
                                "mode",
                                "trigger",
                                "metrics",
                                "suite",
                                "baselines",
                            )
                            if key in payload
                        }
                    ),
                }
            )
        return values

    def dashboard_run(self, run_id: str) -> dict[str, Any]:
        runs = self.config.paths["runs"]
        if runs is None or Path(run_id).name != run_id:
            raise FileNotFoundError(run_id)
        run_path = (runs / run_id).resolve(strict=True)
        if not run_path.is_relative_to(runs.resolve(strict=False)):
            raise FileNotFoundError(run_id)
        payload = json.loads(run_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("run manifest must be a JSON object")
        redacted = _redact_dashboard(payload)
        if not isinstance(redacted, dict):
            raise TypeError("redacted run manifest must remain an object")
        return redacted

    def dashboard_artifact(self, artifact_id: str) -> dict[str, Any]:
        value = self.fleet.controller.artifacts.get(artifact_id).to_dict()
        value.pop("path", None)
        value["content_available"] = True
        redacted = _redact_dashboard(value)
        if not isinstance(redacted, dict):
            raise TypeError("redacted artifact metadata must remain an object")
        return redacted

    def _execution_traces(self, limit: int = 50) -> list[dict[str, Any]]:
        runs = self.config.paths["runs"]
        if runs is None:
            return []
        root = runs / "execution-graphs"
        if not root.exists():
            return []
        values: list[dict[str, Any]] = []
        candidates = sorted(
            root.glob("*.json"), key=lambda path: path.stat().st_mtime_ns, reverse=True
        )[:limit]
        for path in candidates:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(payload, dict):
                values.append({"id": path.name, **_redact_dashboard(payload)})
        return values

    def dashboard_snapshot(
        self, *, session: dict[str, object] | None = None
    ) -> dict[str, Any]:
        catalog = self.dashboard_catalog()
        raw_events = self.fleet.event_log.read()
        fleet_status = self.fleet.status()
        for event in raw_events:
            if event.get("event") != "resource_telemetry":
                continue
            endpoint_status = fleet_status["endpoints"].get(event.get("endpoint"))
            details = event.get("details", {})
            if endpoint_status is None or not isinstance(details, dict):
                continue
            endpoint_status["peak_request_ram_bytes"] = max(
                int(endpoint_status.get("peak_request_ram_bytes", 0)),
                int(details.get("peak_ram_bytes", 0)),
            )
            endpoint_status["peak_request_vram_bytes"] = max(
                int(endpoint_status.get("peak_request_vram_bytes", 0)),
                int(details.get("peak_vram_bytes", 0)),
            )
            endpoint_status["last_request_elapsed_ms"] = details.get("elapsed_ms")
        records = self.fleet.controller.artifacts.list_records(limit=200)
        artifacts = []
        for record in records:
            value = record.to_dict()
            value.pop("path", None)
            value["content_available"] = True
            artifacts.append(_redact_dashboard(value))
        routing_summary: dict[str, Any] = {}
        configured_routing = self.config.data.get("routing", {})
        if isinstance(configured_routing, dict):
            for policy_name in ("routes", "workflows", "use_cases"):
                configured_path = configured_routing.get(policy_name)
                if not isinstance(configured_path, str):
                    continue
                policy_path = Path(configured_path)
                if not policy_path.is_absolute():
                    policy_path = self.config.root / policy_path
                try:
                    policy = yaml.safe_load(policy_path.read_text(encoding="utf-8")) or {}
                except (OSError, yaml.YAMLError):
                    routing_summary[policy_name] = {"status": "unavailable"}
                else:
                    routing_summary[policy_name] = policy
        config_summary = {
            "routing": routing_summary,
            "controller": {
                key: value
                for key, value in self.config.data.get("controller", {}).items()
                if key
                in {
                    "request_timeout_seconds",
                    "telemetry_interval_seconds",
                    "maximum_image_bytes",
                }
            },
            "fleet": {
                key: value
                for key, value in self.config.data.get("fleet", {}).items()
                if key in {"queue_capacity", "require_gpu_telemetry"}
            },
            "repetition": copy.deepcopy(self.config.data.get("repetition", {})),
        }
        return {
            "schema": "sparse-network-dashboard-snapshot.v1",
            "generated_at": datetime.now(UTC).isoformat(),
            "session": session
            or {
                "role": "viewer",
                "permissions": ["dashboard:read"],
                "read_only": True,
            },
            "operations": {
                "profile_switch_in_progress": self._switching,
                "confirmation_patterns": {
                    "fleet": "CONFIRM {operation} {endpoint-or-fleet}",
                    "profile": "APPLY {profile-id}",
                    "configuration": "APPLY CONFIG",
                },
            },
            "fleet": fleet_status,
            "profiles": _redact_dashboard(catalog["profiles"]),
            "models": _redact_dashboard(catalog["models"]),
            "requests": [
                _redact_dashboard(record.to_dict()) for record in self.requests.values()
            ],
            "execution_traces": self._execution_traces(),
            "artifacts": artifacts,
            "evaluations": list(self.evaluations.values()),
            "use_cases": {
                "catalog": _redact_dashboard(self.use_cases.catalog()),
                "runs": [
                    _redact_dashboard(value) for value in self.use_cases.recent(limit=50)
                ],
                "jobs": [
                    _redact_dashboard(value.to_dict())
                    for value in self.use_case_jobs.values()
                ],
            },
            "runs": self._recent_runs(),
            "antidoom": self.antidoom_status(),
            "configuration": _redact_dashboard(config_summary),
            "events": [
                _redact_dashboard(event) for event in raw_events[-200:]
            ],
        }

    def run_swap(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.fleet.exclusive_swap_endpoint_ids:
            raise ValueError(
                "the selected hardware profile does not provide a 27B escalation model"
            )
        return self.swap.run(
            SwapRequest(
                prompt=str(payload.get("prompt", "")),
                candidate=(str(payload["candidate"]) if payload.get("candidate") else None),
                evidence_references=tuple(
                    str(value) for value in payload.get("evidence_references", [])
                ),
                expected_contains=(
                    str(payload["expected_contains"])
                    if payload.get("expected_contains")
                    else None
                ),
            )
        ).to_dict()

    def run_workflow(self, payload: dict[str, Any]) -> dict[str, Any]:
        mode = str(payload.get("mode", "top-1"))
        prompt = str(payload.get("prompt", ""))
        if not prompt.strip():
            raise ValueError("workflow prompt must not be empty")
        router = StaticRouter(
            self.config,
            self.registry,
            resident_endpoint_ids=set(self.fleet.resident_endpoint_ids),
        )
        execution = ExecutionRequest(
            route=RouteRequest(
                prompt=prompt,
                images=tuple(Path(str(value)) for value in payload.get("images", [])),
                explicit_lane=(str(payload["lane"]) if payload.get("lane") else None),
                high_risk=bool(payload.get("high_risk", False)),
            ),
            evidence_references=tuple(
                str(value) for value in payload.get("evidence_references", [])
            ),
            expected_contains=(
                str(payload["expected_contains"])
                if payload.get("expected_contains")
                else None
            ),
            timeout_seconds=(
                float(payload["timeout_seconds"])
                if payload.get("timeout_seconds") is not None
                else None
            ),
        )
        if mode == "top-1":
            return Top1Executor(self.fleet, router).run(execution).to_dict()
        trigger = str(payload.get("trigger", ""))
        return WorkflowExecutor(self.fleet, router).run(
            WorkflowRequest(execution=execution, mode=mode, trigger=trigger)
        ).to_dict()

    def run_use_case(
        self,
        payload: dict[str, Any],
        *,
        actor: str,
        permissions: set[str],
    ) -> dict[str, Any]:
        return self.use_cases.run(payload, actor=actor, permissions=permissions)

    def use_case_readiness(
        self,
        payload: dict[str, Any],
        *,
        permissions: set[str],
    ) -> dict[str, Any]:
        result = self.use_cases.readiness(payload)
        result["preparation"]["permitted"] = "fleet:operate" in permissions
        return result

    def submit_use_case(
        self,
        payload: dict[str, Any],
        *,
        actor: str,
        permissions: set[str],
    ) -> dict[str, Any]:
        template_id = str(payload.get("template_id", ""))
        if template_id not in self.use_cases.templates:
            raise ValueError(f"unknown use-case template: {template_id}")
        readiness = self.use_cases.readiness(payload)
        if readiness["state"] == "blocked":
            reasons = "; ".join(readiness["blocking_reasons"])
            raise ValueError(reasons or "a required capability is unavailable")
        if readiness["state"] == "needs_preparation" and "fleet:operate" not in permissions:
            raise PermissionError("this task needs installed models prepared by an authorized role")
        job = UseCaseJob(
            id=f"usecase-{uuid4()}",
            template_id=template_id,
            title=str(payload.get("title", self.use_cases.templates[template_id].get("title"))),
            actor=actor,
            parent_run_id=str(payload.get("parent_run_id") or "") or None,
        )
        with self._lock:
            self.use_case_jobs[job.id] = job

        manager = self.use_cases

        def worker() -> None:
            if readiness["state"] == "needs_preparation":
                job.state = "preparing_models"
                manager.fleet.event_log.emit(
                    event="use_case_preparing_models",
                    endpoint="controller",
                    request_id=job.id,
                    details={
                        "template_id": template_id,
                        "endpoints": readiness["preparation"]["endpoints"],
                    },
                )
                try:
                    with self._preparation_lock:
                        refreshed = manager.readiness(payload)
                        endpoints = tuple(refreshed["preparation"]["endpoints"])
                        if refreshed["state"] == "blocked":
                            raise ValueError(
                                "; ".join(refreshed["blocking_reasons"])
                                or "a required capability is unavailable"
                            )
                        if endpoints:
                            manager.fleet.load_endpoints(endpoints)
                    if job.cancellation.is_set():
                        raise RuntimeError("use-case run cancelled during model preparation")
                except Exception as exc:
                    job.error_type = type(exc).__name__
                    job.error = str(exc)
                    job.state = "cancelled" if job.cancellation.is_set() else "failed"
                    manager.fleet.event_log.emit(
                        event=(
                            "use_case_cancelled"
                            if job.state == "cancelled"
                            else "use_case_preparation_failed"
                        ),
                        endpoint="controller",
                        request_id=job.id,
                        details={"template_id": template_id, "error_type": job.error_type},
                    )
                    job.completed_at = datetime.now(UTC).isoformat()
                    return
            job.state = "running"
            manager.fleet.event_log.emit(
                event="use_case_started",
                endpoint="controller",
                request_id=job.id,
                details={"template_id": template_id, "actor": actor},
            )
            def update_progress(value: dict[str, Any]) -> None:
                job.progress = copy.deepcopy(value)
                manager.fleet.event_log.emit(
                    event="use_case_stage_changed",
                    endpoint="controller",
                    request_id=job.id,
                    details={"template_id": template_id, **value},
                )

            try:
                job.result = manager.run(
                    payload,
                    actor=actor,
                    permissions=permissions,
                    run_id=job.id,
                    cancellation=job.cancellation,
                    progress=update_progress,
                )
                job.state = "completed"
                job.progress = {"stage_id": "ready", "label": "Ready"}
            except Exception as exc:
                job.error_type = type(exc).__name__
                job.error = str(exc)
                job.state = "cancelled" if job.cancellation.is_set() else "failed"
                manager.fleet.event_log.emit(
                    event="use_case_cancelled" if job.state == "cancelled" else "use_case_failed",
                    endpoint="controller",
                    request_id=job.id,
                    details={"template_id": template_id, "error_type": job.error_type},
                )
            finally:
                job.completed_at = datetime.now(UTC).isoformat()

        threading.Thread(target=worker, name=f"use-case-{job.id}", daemon=True).start()
        return job.to_dict()

    def use_case_run(self, run_id: str) -> dict[str, Any]:
        job = self.use_case_jobs.get(run_id)
        if job is not None and job.state != "completed":
            return job.to_dict()
        if job is not None and job.result is not None:
            return job.result
        return self.use_cases.get(run_id)

    def correct_use_case_result(
        self,
        run_id: str,
        payload: dict[str, Any],
        *,
        actor: str,
    ) -> dict[str, Any]:
        result = self.use_cases.correct_result(run_id, payload, actor=actor)
        job = self.use_case_jobs.get(run_id)
        if job is not None:
            job.result = result
        return result

    def export_use_case_result(
        self,
        run_id: str,
        payload: dict[str, Any],
        *,
        actor: str,
    ) -> dict[str, Any]:
        return self.use_cases.export_result(run_id, payload, actor=actor)

    def cancel_use_case(
        self,
        run_id: str,
        *,
        confirmation: str,
        actor: str,
    ) -> dict[str, Any]:
        try:
            job = self.use_case_jobs[run_id]
        except KeyError as exc:
            raise KeyError(f"unknown use-case job: {run_id}") from exc
        if confirmation != f"CANCEL {run_id}":
            raise ValueError(f"use-case cancellation requires confirmation: CANCEL {run_id}")
        if job.state not in {"queued", "preparing_models", "running"}:
            raise ValueError(f"use-case job cannot be cancelled from {job.state}")
        job.cancellation.set()
        job.state = "cancelling"
        self.fleet.event_log.emit(
            event="use_case_cancellation_requested",
            endpoint="controller",
            request_id=run_id,
            details={"actor": actor},
        )
        return job.to_dict()

    def replay_use_case(
        self,
        run_id: str,
        payload: dict[str, Any],
        *,
        actor: str,
        permissions: set[str],
    ) -> dict[str, Any]:
        return self.use_cases.replay_batch(
            run_id, payload, actor=actor, permissions=permissions
        )

    def decide_use_case_admission(
        self,
        run_id: str,
        payload: dict[str, Any],
        *,
        actor: str,
    ) -> dict[str, Any]:
        return self.use_cases.admission_decision(run_id, payload, actor=actor)

    def write_status(self, path: Path) -> None:
        path.write_text(json.dumps(self.state_snapshot(), indent=2), encoding="utf-8")
