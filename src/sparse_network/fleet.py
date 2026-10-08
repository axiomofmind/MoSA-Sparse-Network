"""Resident endpoint fleet, persisted state machine, and bounded scheduler."""

from __future__ import annotations

import heapq
import json
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from time import monotonic
from typing import Any

import yaml

from .config import AppConfig
from .contracts import SmokeResult
from .controller import Controller
from .errors import (
    EndpointStateError,
    QueueCapacityError,
    RequestCancelledError,
    RequestFailedError,
    ResourceAdmissionError,
)
from .models import EndpointDefinition, ModelRegistry
from .runtimes import RuntimeAdapter, RuntimeState
from .telemetry import gpu_memory_used_bytes, process_tree_rss_bytes, query_nvidia_gpus
from .validation import validate_endpoint_artifacts


class FleetState(StrEnum):
    UNAVAILABLE = "unavailable"
    LOADING = "loading"
    READY = "ready"
    BUSY = "busy"
    DRAINING = "draining"
    UNLOADING = "unloading"
    FAILED = "failed"
    QUARANTINED = "quarantined"
    RESTORING = "restoring"


TRANSITIONAL_STATES = {
    FleetState.LOADING,
    FleetState.BUSY,
    FleetState.DRAINING,
    FleetState.UNLOADING,
    FleetState.RESTORING,
}

LEGAL_TRANSITIONS: dict[FleetState, set[FleetState]] = {
    FleetState.UNAVAILABLE: {FleetState.LOADING, FleetState.QUARANTINED},
    FleetState.LOADING: {FleetState.READY, FleetState.FAILED},
    FleetState.READY: {FleetState.BUSY, FleetState.DRAINING, FleetState.FAILED},
    FleetState.BUSY: {FleetState.READY, FleetState.DRAINING, FleetState.FAILED},
    FleetState.DRAINING: {FleetState.READY, FleetState.UNLOADING, FleetState.FAILED},
    FleetState.UNLOADING: {FleetState.UNAVAILABLE, FleetState.FAILED},
    FleetState.FAILED: {
        FleetState.UNAVAILABLE,
        FleetState.QUARANTINED,
        FleetState.RESTORING,
    },
    FleetState.QUARANTINED: {FleetState.UNAVAILABLE, FleetState.RESTORING},
    FleetState.RESTORING: {FleetState.LOADING, FleetState.FAILED},
}


def validate_transition(source: FleetState, target: FleetState) -> None:
    if target not in LEGAL_TRANSITIONS[source]:
        raise EndpointStateError(f"Illegal fleet transition: {source.value} -> {target.value}")


class ControllerEventLog:
    """Durable, ordered controller events with bounded replay support."""

    def __init__(self, path: Path, *, retention: int = 10000) -> None:
        self.path = path.resolve(strict=False)
        self._lock = threading.Lock()
        self._sequence = 0
        self.retention = max(100, retention)
        if self.path.exists():
            for event in self.read():
                identifier = event.get("event_id", event.get("details", {}).get("sequence", 0))
                self._sequence = max(self._sequence, int(identifier))

    def read(self) -> tuple[dict[str, Any], ...]:
        if not self.path.exists():
            return ()
        events: list[dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                events.append(value)
        return tuple(events)

    def emit(
        self,
        *,
        event: str,
        endpoint: str,
        request_id: str | None = None,
        execution_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            self._sequence += 1
            value = {
                "schema": "sparse-network-controller-event.v1",
                "event_id": self._sequence,
                "timestamp": datetime.now(UTC).isoformat(),
                "event": event,
                "request_id": request_id,
                "execution_id": execution_id,
                "endpoint": endpoint,
                "details": {"sequence": self._sequence, **(details or {})},
            }
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(value, sort_keys=True) + "\n")
                handle.flush()
            self._compact_if_needed()
            return value

    def _compact_if_needed(self) -> None:
        lines = self.path.read_text(encoding="utf-8", errors="replace").splitlines()
        if len(lines) <= self.retention * 2:
            return
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text("\n".join(lines[-self.retention :]) + "\n", encoding="utf-8")
        temporary.replace(self.path)

    def read_after(self, event_id: int, *, limit: int = 1000) -> tuple[dict[str, Any], ...]:
        """Replay retained events after an identifier, with bounded backpressure."""

        if event_id < 0 or limit <= 0 or limit > 1000:
            raise ValueError("event_id must be non-negative and limit must be 1..1000")
        return tuple(
            event
            for event in self.read()
            if int(event.get("event_id", event.get("details", {}).get("sequence", 0)))
            > event_id
        )[:limit]

    def last_states(self) -> dict[str, FleetState]:
        states: dict[str, FleetState] = {}
        for event in self.read():
            if event.get("event") != "state_transition":
                continue
            endpoint = event.get("endpoint")
            state = event.get("details", {}).get("to")
            try:
                if isinstance(endpoint, str):
                    states[endpoint] = FleetState(str(state))
            except ValueError:
                continue
        return states


@dataclass
class FleetEntry:
    endpoint: EndpointDefinition
    state: FleetState = FleetState.UNAVAILABLE
    runtime: RuntimeAdapter | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)
    active_requests: int = 0
    pending_requests: int = 0
    load_ms: int = 0
    loaded_vram_bytes: int = 0
    incremental_vram_bytes: int = 0
    loaded_process_ram_bytes: int = 0
    failure: str | None = None
    request_queue: list[tuple[int, int, ScheduledRequest]] = field(default_factory=list)


@dataclass
class ScheduledRequest:
    endpoint_id: str
    prompt: str
    original_prompt: str | None
    images: tuple[Path, ...]
    evidence_references: tuple[str, ...]
    cancellation: threading.Event
    timeout_seconds: float | None
    priority: int
    completed: threading.Event = field(default_factory=threading.Event)
    result: SmokeResult | None = None
    error: BaseException | None = None


class FleetManager:
    def __init__(
        self,
        config: AppConfig,
        registry: ModelRegistry,
        *,
        roster_path: Path | None = None,
        maximum_parallel_generations: int | None = None,
    ) -> None:
        self.config = config
        self.registry = registry
        self.controller = Controller(config, registry)
        fleet_config = config.data.get("fleet", {})
        selected_roster = roster_path or Path(
            str(
                fleet_config.get(
                    "roster", "configs/rosters/reference-32gb-gemma12.yaml"
                )
            )
        )
        if not selected_roster.is_absolute():
            selected_roster = config.root / selected_roster
        self.roster_path = selected_roster.resolve(strict=True)
        roster = yaml.safe_load(self.roster_path.read_text(encoding="utf-8"))
        if not isinstance(roster, dict) or not isinstance(roster.get("resident"), list):
            raise ValueError("Fleet roster needs a resident endpoint list")
        self.roster_id = str(roster.get("id", self.roster_path.stem))
        self.roster_display_name = str(roster.get("display_name", self.roster_id))
        self.roster_variant = str(roster.get("variant", "unspecified"))
        self.hardware_profile_id = str(roster.get("hardware_profile", ""))
        self.exclusive_swap_endpoint_ids = tuple(
            str(endpoint_id) for endpoint_id in roster.get("exclusive_swap", [])
        )
        self.resident_endpoint_ids = tuple(str(value) for value in roster["resident"])
        known_endpoint_ids = {endpoint.id for endpoint in registry.all()}
        self.optional_endpoint_ids = tuple(
            str(value)
            for value in roster.get("optional", [])
            if str(value) in known_endpoint_ids
        )
        for endpoint_id in self.exclusive_swap_endpoint_ids:
            registry.get(endpoint_id)
        hardware_path = config.root / "configs" / "hardware" / f"{self.hardware_profile_id}.yaml"
        hardware = yaml.safe_load(hardware_path.read_text(encoding="utf-8"))
        scheduler = hardware.get("scheduler", {})
        budgets = hardware.get("budgets", {})
        configured_parallel = int(scheduler.get("initial_parallel_generations", 1))
        self.maximum_parallel_generations = (
            maximum_parallel_generations
            if maximum_parallel_generations is not None
            else configured_parallel
        )
        if self.maximum_parallel_generations <= 0:
            raise ValueError("maximum_parallel_generations must be positive")
        self.queue_capacity = int(fleet_config.get("queue_capacity", 32))
        self.minimum_vram_reserve_bytes = int(
            float(budgets.get("minimum_vram_reserve_gb", 2)) * 1024**3
        )
        self.total_kv_cache_bytes = int(float(budgets.get("total_kv_cache_gb", 8)) * 1024**3)
        self.telemetry_required = bool(fleet_config.get("require_gpu_telemetry", True))
        self.entries = {
            str(endpoint_id): FleetEntry(registry.get(str(endpoint_id)))
            for endpoint_id in dict.fromkeys(
                (*self.resident_endpoint_ids, *self.optional_endpoint_ids)
            )
        }
        runs = config.paths["runs"]
        if runs is None:
            raise ValueError("paths.runs must be configured")
        self.event_log = ControllerEventLog(runs / "fleet-events.jsonl")
        recovered = self.event_log.last_states()
        for endpoint_id, entry in self.entries.items():
            if recovered.get(endpoint_id) in TRANSITIONAL_STATES:
                entry.state = FleetState.QUARANTINED
                entry.failure = "previous process ended in a transitional state"
                self.event_log.emit(
                    event="recovery_quarantine",
                    endpoint=endpoint_id,
                    details={"previous_state": recovered[endpoint_id].value},
                )
        self._condition = threading.Condition()
        self._load_lock = threading.Lock()
        self._active_total = 0
        self._accepting = True
        self._sequence = 0
        self._queued_total = 0
        self._stopping_workers = False
        self._workers = [
            threading.Thread(target=self._worker, name=f"fleet-worker-{index}", daemon=True)
            for index in range(self.maximum_parallel_generations)
        ]
        for worker in self._workers:
            worker.start()
        self.initial_vram_bytes = 0
        self.peak_vram_bytes = 0
        self.total_vram_bytes = self._detect_total_vram()

    def _detect_total_vram(self) -> int:
        gpu_indexes = {
            entry.endpoint.telemetry_gpu_index
            for entry in self.entries.values()
            if entry.endpoint.telemetry_gpu_index is not None
        }
        if len(gpu_indexes) > 1:
            raise ResourceAdmissionError("Resident endpoints must use one telemetry GPU")
        if not gpu_indexes:
            return 0
        selected = next(iter(gpu_indexes))
        for gpu in query_nvidia_gpus():
            if int(gpu["index"]) == selected:
                return int(gpu["memory_total_mib"]) * 1024**2
        if self.telemetry_required:
            raise ResourceAdmissionError("Required GPU telemetry is unavailable")
        return 0

    def _transition(self, entry: FleetEntry, target: FleetState, *, reason: str) -> None:
        source = entry.state
        validate_transition(source, target)
        entry.state = target
        self.event_log.emit(
            event="state_transition",
            endpoint=entry.endpoint.id,
            details={"from": source.value, "to": target.value, "reason": reason},
        )

    def _planned_budget_check(self, requested: tuple[str, ...]) -> None:
        planned = set(requested) | {
            endpoint_id
            for endpoint_id, entry in self.entries.items()
            if entry.state in {FleetState.LOADING, FleetState.READY, FleetState.BUSY}
        }
        proposed_vram = sum(
            int(self.entries[endpoint_id].endpoint.proposed_budget.get("incremental_vram_bytes", 0))
            for endpoint_id in planned
        )
        proposed_kv = sum(
            int(self.entries[endpoint_id].endpoint.proposed_budget.get("kv_cache_bytes", 0))
            for endpoint_id in planned
        )
        if self.total_vram_bytes and (
            proposed_vram + self.minimum_vram_reserve_bytes > self.total_vram_bytes
        ):
            raise ResourceAdmissionError(
                "Planned resident VRAM exceeds capacity after the required reserve"
            )
        if proposed_kv > self.total_kv_cache_bytes:
            raise ResourceAdmissionError("Planned KV caches exceed the fleet KV budget")

    def load_all(self) -> dict[str, Any]:
        return self.load_endpoints(self.resident_endpoint_ids)

    def residents_ready(self) -> bool:
        """Check required residents without treating optional endpoints as required."""

        return all(
            self.entries[endpoint_id].state == FleetState.READY
            and self.entries[endpoint_id].runtime is not None
            for endpoint_id in self.resident_endpoint_ids
        )

    def load_endpoints(self, endpoint_ids: tuple[str, ...] | list[str]) -> dict[str, Any]:
        """Load only installed resident endpoints, serializing concurrent cold starts."""

        requested = tuple(dict.fromkeys(str(value) for value in endpoint_ids))
        unknown = [value for value in requested if value not in self.entries]
        if unknown:
            raise ValueError(f"endpoints are not resident in this fleet: {', '.join(unknown)}")
        with self._load_lock:
            return self._load_endpoints_locked(requested)

    def _load_endpoints_locked(self, endpoint_ids: tuple[str, ...]) -> dict[str, Any]:
        self._planned_budget_check(endpoint_ids)
        telemetry_index = next(
            (
                entry.endpoint.telemetry_gpu_index
                for entry in self.entries.values()
                if entry.endpoint.telemetry_gpu_index is not None
            ),
            None,
        )
        self.initial_vram_bytes = gpu_memory_used_bytes(telemetry_index)
        self.peak_vram_bytes = self.initial_vram_bytes
        loaded: list[FleetEntry] = []
        try:
            for endpoint_id in endpoint_ids:
                entry = self.entries[endpoint_id]
                if entry.state in {FleetState.READY, FleetState.BUSY}:
                    continue
                if entry.state not in {
                    FleetState.UNAVAILABLE,
                    FleetState.QUARANTINED,
                    FleetState.FAILED,
                }:
                    raise ValueError(
                        f"endpoint {endpoint_id} cannot be prepared from {entry.state.value}"
                    )
                validate_endpoint_artifacts(entry.endpoint, self.config.paths["model_cache"])
                if entry.state in {FleetState.QUARANTINED, FleetState.FAILED}:
                    self._transition(entry, FleetState.RESTORING, reason="explicit fleet load")
                    self._transition(entry, FleetState.LOADING, reason="restoration started")
                else:
                    self._transition(entry, FleetState.LOADING, reason="fleet load")
                try:
                    runtime = self.controller.create_runtime(entry.endpoint)
                except Exception as exc:
                    entry.failure = str(exc)
                    self._transition(entry, FleetState.FAILED, reason="runtime creation failed")
                    raise
                before_vram = gpu_memory_used_bytes(entry.endpoint.telemetry_gpu_index)
                started = monotonic()
                try:
                    runtime.start()
                except Exception as exc:
                    runtime.stop()
                    entry.failure = str(exc)
                    self._transition(entry, FleetState.FAILED, reason="runtime load failed")
                    raise
                entry.runtime = runtime
                entry.load_ms = round((monotonic() - started) * 1000)
                entry.loaded_process_ram_bytes = process_tree_rss_bytes(runtime.pid)
                current_vram = gpu_memory_used_bytes(entry.endpoint.telemetry_gpu_index)
                entry.loaded_vram_bytes = current_vram
                entry.incremental_vram_bytes = max(0, current_vram - before_vram)
                self.peak_vram_bytes = max(self.peak_vram_bytes, current_vram)
                if self.total_vram_bytes and (
                    current_vram + self.minimum_vram_reserve_bytes > self.total_vram_bytes
                ):
                    self.event_log.emit(
                        event="resource_admission_failed",
                        endpoint=entry.endpoint.id,
                        details={
                            "used_vram_bytes": current_vram,
                            "total_vram_bytes": self.total_vram_bytes,
                            "reserve_bytes": self.minimum_vram_reserve_bytes,
                        },
                    )
                    runtime.stop()
                    entry.runtime = None
                    entry.failure = "fleet load breached the required VRAM reserve"
                    self._transition(entry, FleetState.FAILED, reason=entry.failure)
                    raise ResourceAdmissionError("Fleet load breached the required VRAM reserve")
                self._transition(entry, FleetState.READY, reason="runtime healthy")
                loaded.append(entry)
        except Exception:
            for entry in reversed(loaded):
                self._force_unload(entry, reason="rollback after fleet load failure")
            raise
        return self.status()

    def select_endpoint(
        self,
        *,
        capability: str,
        required_modalities: tuple[str, ...] = ("text",),
    ) -> str:
        """Choose the cheapest compatible resident endpoint deterministically."""
        required = set(required_modalities)
        candidates = [
            entry
            for entry in self.entries.values()
            if entry.state in {FleetState.READY, FleetState.BUSY}
            and capability in entry.endpoint.capabilities
            and required.issubset(entry.endpoint.modalities)
        ]
        if not candidates:
            raise EndpointStateError(
                f"No resident endpoint can serve {capability!r} with modalities "
                f"{sorted(required)!r}"
            )
        candidates.sort(
            key=lambda entry: (
                int(entry.endpoint.proposed_budget.get("request_workspace_bytes", 0)),
                int(entry.endpoint.proposed_budget.get("incremental_vram_bytes", 0)),
                entry.endpoint.id,
            )
        )
        return candidates[0].endpoint.id

    def _request_headroom_check(self, entry: FleetEntry) -> None:
        if self.total_vram_bytes == 0:
            if self.telemetry_required and entry.endpoint.telemetry_gpu_index is not None:
                raise ResourceAdmissionError("GPU telemetry unavailable; request rejected")
            return
        used = gpu_memory_used_bytes(entry.endpoint.telemetry_gpu_index)
        workspace = int(entry.endpoint.proposed_budget.get("request_workspace_bytes", 0))
        if used + workspace + self.minimum_vram_reserve_bytes > self.total_vram_bytes:
            self.event_log.emit(
                event="resource_admission_failed",
                endpoint=entry.endpoint.id,
                details={
                    "used_vram_bytes": used,
                    "workspace_bytes": workspace,
                    "reserve_bytes": self.minimum_vram_reserve_bytes,
                },
            )
            raise ResourceAdmissionError("Request would breach the required VRAM reserve")

    def submit(
        self,
        *,
        endpoint_id: str,
        prompt: str,
        original_prompt: str | None = None,
        images: tuple[Path, ...] = (),
        evidence_references: tuple[str, ...] = (),
        cancellation: threading.Event | None = None,
        timeout_seconds: float | None = None,
        priority: int = 10,
        wait_timeout_seconds: float | None = None,
    ) -> SmokeResult:
        entry = self.entries.get(endpoint_id)
        if entry is None:
            raise EndpointStateError(f"Endpoint is not in the resident roster: {endpoint_id}")
        with self._condition:
            if not self._accepting:
                raise EndpointStateError("Fleet is shutting down")
            if entry.state not in {FleetState.READY, FleetState.BUSY}:
                raise EndpointStateError(
                    f"Endpoint {endpoint_id} cannot accept work from {entry.state.value}"
                )
            entry.pending_requests += 1
            self._sequence += 1
            sequence = self._sequence
        request = ScheduledRequest(
            endpoint_id=endpoint_id,
            prompt=prompt,
            original_prompt=original_prompt,
            images=images,
            evidence_references=evidence_references,
            cancellation=cancellation or threading.Event(),
            timeout_seconds=timeout_seconds,
            priority=priority,
        )
        with self._condition:
            if self._queued_total >= self.queue_capacity:
                entry.pending_requests -= 1
                raise QueueCapacityError("Fleet request queue is full")
            heapq.heappush(entry.request_queue, (priority, sequence, request))
            self._queued_total += 1
            self._condition.notify()
        self.event_log.emit(
            event="request_queued",
            endpoint=endpoint_id,
            details={"priority": priority, "queue_sequence": sequence},
        )
        if not request.completed.wait(wait_timeout_seconds):
            request.cancellation.set()
            raise RequestFailedError("Timed out waiting for the scheduled request")
        if request.error is not None:
            raise request.error
        if request.result is None:
            raise RequestFailedError("Fleet request completed without a result")
        return request.result

    def _take_request(self) -> tuple[FleetEntry, ScheduledRequest] | None:
        with self._condition:
            while True:
                candidates = [
                    (entry.request_queue[0][0], entry.request_queue[0][1], endpoint_id)
                    for endpoint_id, entry in self.entries.items()
                    if entry.request_queue and entry.state == FleetState.READY
                ]
                if candidates:
                    _priority, _sequence, endpoint_id = min(candidates)
                    entry = self.entries[endpoint_id]
                    _priority, _sequence, request = heapq.heappop(entry.request_queue)
                    self._queued_total -= 1
                    return entry, request
                if self._stopping_workers and self._queued_total == 0:
                    return None
                self._condition.wait(timeout=0.1)

    def _worker(self) -> None:
        while True:
            selected = self._take_request()
            if selected is None:
                return
            entry, request = selected
            entry.lock.acquire()
            try:
                if request.cancellation.is_set():
                    raise RequestCancelledError("Request cancelled while queued")
                with self._condition:
                    if entry.state != FleetState.READY:
                        raise EndpointStateError(
                            f"Endpoint became unavailable while queued: {entry.state.value}"
                        )
                    self._request_headroom_check(entry)
                    self._transition(entry, FleetState.BUSY, reason="scheduled request started")
                    entry.active_requests += 1
                    self._active_total += 1
                runtime = entry.runtime
                if runtime is None:
                    raise EndpointStateError("Resident endpoint has no runtime")
                request.result = self.controller.smoke(
                    endpoint_id=entry.endpoint.id,
                    prompt=request.prompt,
                    original_prompt=request.original_prompt,
                    images=request.images,
                    evidence_references=request.evidence_references,
                    runtime=runtime,
                    keep_loaded=True,
                    cancellation=request.cancellation,
                    timeout_seconds=request.timeout_seconds,
                )
                observed_peak = int(request.result.lifecycle["peak_vram_bytes"])
                self.peak_vram_bytes = max(self.peak_vram_bytes, observed_peak)
                if self.total_vram_bytes and (
                    observed_peak + self.minimum_vram_reserve_bytes > self.total_vram_bytes
                ):
                    self.event_log.emit(
                        event="resource_reserve_breached",
                        endpoint=entry.endpoint.id,
                        request_id=request.result.envelope.request_id,
                        details={
                            "peak_vram_bytes": observed_peak,
                            "reserve_bytes": self.minimum_vram_reserve_bytes,
                        },
                    )
                self.event_log.emit(
                    event="request_completed",
                    endpoint=entry.endpoint.id,
                    request_id=request.result.envelope.request_id,
                    execution_id=request.result.envelope.execution_id,
                    details={"status": request.result.envelope.status},
                )
                if request.result.envelope.repetition.get("detected", False):
                    self.event_log.emit(
                        event="repetition_detected",
                        endpoint=entry.endpoint.id,
                        request_id=request.result.envelope.request_id,
                        execution_id=request.result.envelope.execution_id,
                        details={
                            "model_revision": entry.endpoint.model_revision,
                            "route": request.result.envelope.capability,
                            "sampling": {"temperature": 0},
                            **request.result.envelope.repetition,
                        },
                    )
            except BaseException as exc:
                request.error = exc
                self.event_log.emit(
                    event="request_failed",
                    endpoint=entry.endpoint.id,
                    details={"error": str(exc), "type": type(exc).__name__},
                )
            finally:
                with self._condition:
                    if entry.active_requests:
                        entry.active_requests -= 1
                        self._active_total -= 1
                    entry.pending_requests = max(0, entry.pending_requests - 1)
                    runtime_healthy = (
                        entry.runtime is not None
                        and entry.runtime.state == RuntimeState.READY
                    )
                    if entry.state == FleetState.BUSY and runtime_healthy:
                        self._transition(entry, FleetState.READY, reason="request finished")
                    elif (
                        entry.state in {FleetState.BUSY, FleetState.DRAINING}
                        and not runtime_healthy
                    ):
                        entry.failure = "runtime did not return to ready after request"
                        self._transition(entry, FleetState.FAILED, reason=entry.failure)
                    current = gpu_memory_used_bytes(entry.endpoint.telemetry_gpu_index)
                    self.peak_vram_bytes = max(self.peak_vram_bytes, current)
                    self._condition.notify_all()
                entry.lock.release()
                request.completed.set()

    def drain_endpoint(self, endpoint_id: str, *, timeout_seconds: float = 120) -> None:
        entry = self.entries[endpoint_id]
        with self._condition:
            if entry.state == FleetState.READY:
                self._transition(entry, FleetState.DRAINING, reason="operator drain")
            elif entry.state == FleetState.BUSY:
                self._transition(entry, FleetState.DRAINING, reason="drain after active request")
            elif entry.state != FleetState.DRAINING:
                raise EndpointStateError(f"Cannot drain endpoint from {entry.state.value}")
            deadline = monotonic() + timeout_seconds
            while (entry.active_requests or entry.pending_requests) and monotonic() < deadline:
                while entry.request_queue:
                    _priority, _sequence, request = heapq.heappop(entry.request_queue)
                    self._queued_total -= 1
                    entry.pending_requests = max(0, entry.pending_requests - 1)
                    request.error = EndpointStateError(
                        f"Endpoint {endpoint_id} drained while request was queued"
                    )
                    request.completed.set()
                self._condition.wait(timeout=0.1)
            if entry.active_requests or entry.pending_requests:
                raise RequestFailedError(f"Timed out draining endpoint {endpoint_id}")

    def _force_unload(self, entry: FleetEntry, *, reason: str) -> None:
        if entry.state == FleetState.READY:
            self._transition(entry, FleetState.DRAINING, reason=reason)
        if entry.state == FleetState.FAILED:
            runtime = entry.runtime
            if runtime is not None:
                runtime.stop()
            entry.runtime = None
            self._transition(entry, FleetState.UNAVAILABLE, reason=reason)
            return
        if entry.state == FleetState.DRAINING:
            self._transition(entry, FleetState.UNLOADING, reason=reason)
        runtime = entry.runtime
        if runtime is not None:
            runtime.stop()
        entry.runtime = None
        if entry.state == FleetState.UNLOADING:
            self._transition(entry, FleetState.UNAVAILABLE, reason="runtime released")

    def unload_endpoint(self, endpoint_id: str, *, timeout_seconds: float = 120) -> None:
        self.drain_endpoint(endpoint_id, timeout_seconds=timeout_seconds)
        self._force_unload(self.entries[endpoint_id], reason="drain complete")

    def quarantine_endpoint(self, endpoint_id: str, *, timeout_seconds: float = 120) -> None:
        entry = self.entries[endpoint_id]
        if entry.state in {FleetState.READY, FleetState.BUSY, FleetState.DRAINING}:
            self.unload_endpoint(endpoint_id, timeout_seconds=timeout_seconds)
        elif entry.runtime is not None:
            self._force_unload(entry, reason="operator quarantine")
        if entry.state in {FleetState.FAILED, FleetState.UNAVAILABLE}:
            self._transition(entry, FleetState.QUARANTINED, reason="operator quarantine")
        elif entry.state != FleetState.QUARANTINED:
            raise EndpointStateError(f"Cannot quarantine endpoint from {entry.state.value}")

    def pause_admission(self, *, reason: str) -> None:
        with self._condition:
            self._accepting = False
        self.event_log.emit(
            event="fleet_admission_paused",
            endpoint="controller",
            details={"reason": reason},
        )

    def resume_admission(self, *, reason: str) -> None:
        with self._condition:
            if self._stopping_workers:
                raise EndpointStateError("Cannot resume a stopped fleet")
            self._accepting = True
            self._condition.notify_all()
        self.event_log.emit(
            event="fleet_admission_resumed",
            endpoint="controller",
            details={"reason": reason},
        )

    def unload_all(self, *, timeout_seconds: float = 120) -> dict[str, Any]:
        for endpoint_id in reversed(tuple(self.entries)):
            entry = self.entries[endpoint_id]
            if entry.state in {FleetState.READY, FleetState.BUSY, FleetState.DRAINING}:
                self.unload_endpoint(endpoint_id, timeout_seconds=timeout_seconds)
            elif entry.runtime is not None:
                self._force_unload(entry, reason="fleet unload all")
        return self.status()

    def restore_endpoint(self, endpoint_id: str) -> None:
        entry = self.entries[endpoint_id]
        if entry.state not in {FleetState.FAILED, FleetState.QUARANTINED}:
            raise EndpointStateError(f"Cannot restore endpoint from {entry.state.value}")
        if entry.runtime is not None:
            entry.runtime.stop()
            entry.runtime = None
        self._transition(entry, FleetState.RESTORING, reason="explicit endpoint restore")
        self._transition(entry, FleetState.LOADING, reason="restoration runtime load")
        runtime: RuntimeAdapter | None = None
        try:
            runtime = self.controller.create_runtime(entry.endpoint)
            before_vram = gpu_memory_used_bytes(entry.endpoint.telemetry_gpu_index)
            started = monotonic()
            runtime.start()
        except Exception as exc:
            if runtime is not None:
                runtime.stop()
            entry.failure = str(exc)
            self._transition(entry, FleetState.FAILED, reason="restoration failed")
            raise
        entry.runtime = runtime
        entry.load_ms = round((monotonic() - started) * 1000)
        entry.loaded_process_ram_bytes = process_tree_rss_bytes(runtime.pid)
        current_vram = gpu_memory_used_bytes(entry.endpoint.telemetry_gpu_index)
        entry.loaded_vram_bytes = current_vram
        entry.incremental_vram_bytes = max(0, current_vram - before_vram)
        self.peak_vram_bytes = max(self.peak_vram_bytes, current_vram)
        if self.total_vram_bytes and (
            current_vram + self.minimum_vram_reserve_bytes > self.total_vram_bytes
        ):
            runtime.stop()
            entry.runtime = None
            entry.failure = "restoration breached VRAM reserve"
            self._transition(entry, FleetState.FAILED, reason=entry.failure)
            raise ResourceAdmissionError(entry.failure)
        entry.failure = None
        self._transition(entry, FleetState.READY, reason="restoration healthy")

    def shutdown(self, *, timeout_seconds: float = 120) -> dict[str, Any]:
        with self._condition:
            self._accepting = False
        errors: list[str] = []
        for endpoint_id in reversed(tuple(self.entries)):
            entry = self.entries[endpoint_id]
            if entry.state in {FleetState.READY, FleetState.BUSY, FleetState.DRAINING}:
                try:
                    self.unload_endpoint(endpoint_id, timeout_seconds=timeout_seconds)
                except Exception as exc:
                    errors.append(f"{endpoint_id}: {exc}")
                    if entry.runtime is not None:
                        entry.runtime.stop()
                        entry.runtime = None
            elif entry.runtime is not None:
                self._force_unload(entry, reason="fleet shutdown")
        with self._condition:
            self._stopping_workers = True
            self._condition.notify_all()
        for worker in self._workers:
            worker.join(timeout=5)
        if errors:
            raise RequestFailedError("Fleet shutdown errors: " + "; ".join(errors))
        return self.status()

    def status(self) -> dict[str, Any]:
        telemetry_index = next(
            (
                entry.endpoint.telemetry_gpu_index
                for entry in self.entries.values()
                if entry.endpoint.telemetry_gpu_index is not None
            ),
            None,
        )
        current_vram = gpu_memory_used_bytes(telemetry_index)
        return {
            "schema": "sparse-network-fleet-status.v1",
            "roster": self.roster_id,
            "roster_display_name": self.roster_display_name,
            "roster_variant": self.roster_variant,
            "exclusive_swap_endpoints": list(self.exclusive_swap_endpoint_ids),
            "active_escalation_endpoint": (
                self.exclusive_swap_endpoint_ids[0]
                if self.exclusive_swap_endpoint_ids
                else None
            ),
            "maximum_parallel_generations": self.maximum_parallel_generations,
            "queue_capacity": self.queue_capacity,
            "queue_depth": self._queued_total,
            "active_requests": self._active_total,
            "initial_vram_bytes": self.initial_vram_bytes,
            "current_vram_bytes": current_vram,
            "peak_vram_bytes": self.peak_vram_bytes,
            "total_vram_bytes": self.total_vram_bytes,
            "minimum_vram_reserve_bytes": self.minimum_vram_reserve_bytes,
            "headroom_bytes": max(0, self.total_vram_bytes - current_vram),
            "endpoints": {
                endpoint_id: {
                    "state": entry.state.value,
                    "pid": entry.runtime.pid if entry.runtime is not None else None,
                    "active_requests": entry.active_requests,
                    "pending_requests": entry.pending_requests,
                    "queue_depth": len(entry.request_queue),
                    "load_ms": entry.load_ms,
                    "loaded_vram_bytes": entry.loaded_vram_bytes,
                    "incremental_vram_bytes": entry.incremental_vram_bytes,
                    "loaded_process_ram_bytes": entry.loaded_process_ram_bytes,
                    "proposed_vram_bytes": int(
                        entry.endpoint.proposed_budget.get("incremental_vram_bytes", 0)
                    ),
                    "kv_cache_bytes": int(
                        entry.endpoint.proposed_budget.get("kv_cache_bytes", 0)
                    ),
                    "failure": entry.failure,
                }
                for endpoint_id, entry in self.entries.items()
            },
        }
