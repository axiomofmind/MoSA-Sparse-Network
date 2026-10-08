"""Exclusive Qwen3.8-27B escalation swap with durable recovery phases."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Any
from uuid import uuid4

import psutil

from .contracts import SmokeResult
from .errors import (
    RequestCancelledError,
    RequestFailedError,
    ResourceAdmissionError,
)
from .fleet import FleetManager, FleetState
from .runtimes import RuntimeAdapter
from .telemetry import gpu_memory_used_bytes
from .validation import validate_endpoint_artifacts

SWAP_PHASES = (
    "preserving",
    "draining",
    "reclaiming",
    "loading_large",
    "solving",
    "verifying",
    "unloading_large",
    "restoring",
    "healthy",
)
TERMINAL_SWAP_PHASES = {"healthy", "failed"}


@dataclass(frozen=True)
class SwapRequest:
    prompt: str
    candidate: str | None = None
    images: tuple[Path, ...] = ()
    evidence_references: tuple[str, ...] = ()
    expected_contains: str | None = None
    timeout_seconds: float | None = None


@dataclass(frozen=True)
class SwapResult:
    swap_id: str
    status: str
    result: SmokeResult | None
    adjudication: SmokeResult | None
    preserved_artifact: str
    phase_timings_ms: dict[str, int]
    fleet_restored: bool
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "sparse-network-swap-result.v1",
            "swap_id": self.swap_id,
            "status": self.status,
            "result": self.result.to_dict() if self.result else None,
            "adjudication": self.adjudication.to_dict() if self.adjudication else None,
            "preserved_artifact": self.preserved_artifact,
            "phase_timings_ms": self.phase_timings_ms,
            "fleet_restored": self.fleet_restored,
            "error": self.error,
        }


class SwapCoordinator:
    """Own the one-at-a-time transition between resident and large-model modes."""

    def __init__(self, fleet: FleetManager, *, endpoint_id: str | None = None) -> None:
        self.fleet = fleet
        declared = fleet.exclusive_swap_endpoint_ids
        self.endpoint_id = endpoint_id or (declared[0] if declared else "qwen38-27b")
        self._lock = threading.Lock()

    def _phase(
        self,
        swap_id: str,
        phase: str,
        *,
        started: float,
        details: dict[str, Any] | None = None,
    ) -> int:
        elapsed = round((monotonic() - started) * 1000)
        self.fleet.event_log.emit(
            event="swap_phase",
            endpoint=self.endpoint_id,
            execution_id=swap_id,
            details={"phase": phase, "elapsed_ms": elapsed, **(details or {})},
        )
        return elapsed

    def recover_if_interrupted(self) -> bool:
        """Restore residents when the durable log ends inside a swap."""

        last = None
        for event in self.fleet.event_log.read():
            if event.get("event") == "swap_phase":
                last = event
        if last is None or last.get("details", {}).get("phase") in TERMINAL_SWAP_PHASES:
            return False
        self.fleet.pause_admission(reason="recover interrupted exclusive swap")
        try:
            if any(entry.runtime is not None for entry in self.fleet.entries.values()):
                self.fleet.unload_all()
            self.fleet.load_all()
            self.fleet.event_log.emit(
                event="swap_recovered",
                endpoint=self.endpoint_id,
                execution_id=last.get("execution_id"),
                details={"interrupted_phase": last.get("details", {}).get("phase")},
            )
            self.fleet.event_log.emit(
                event="swap_phase",
                endpoint=self.endpoint_id,
                execution_id=last.get("execution_id"),
                details={"phase": "healthy", "elapsed_ms": 0, "recovered": True},
            )
        finally:
            healthy = self.fleet.residents_ready()
            if healthy:
                self.fleet.resume_admission(reason="interrupted swap recovered")
        return True

    def _restore_residents(self) -> bool:
        if self.fleet.residents_ready():
            return True
        # A failed drain may leave a mixed roster. Normalize it to fully unloaded
        # before invoking the all-or-nothing fleet loader.
        for entry in reversed(tuple(self.fleet.entries.values())):
            if entry.runtime is None:
                continue
            if entry.state in {FleetState.READY, FleetState.BUSY, FleetState.DRAINING}:
                self.fleet.unload_endpoint(entry.endpoint.id, timeout_seconds=120)
            else:
                self.fleet._force_unload(entry, reason="swap restoration normalization")
        self.fleet.load_all()
        return self.fleet.residents_ready()

    def run(
        self,
        request: SwapRequest,
        *,
        cancellation: threading.Event | None = None,
    ) -> SwapResult:
        if not request.prompt.strip():
            raise ValueError("swap prompt must not be empty")
        if not self._lock.acquire(blocking=False):
            raise RequestFailedError("An exclusive escalation swap is already active")
        cancellation = cancellation or threading.Event()
        swap_id = f"swap-{uuid4()}"
        phase_timings: dict[str, int] = {}
        large_runtime: RuntimeAdapter | None = None
        answer: SmokeResult | None = None
        adjudication: SmokeResult | None = None
        error: str | None = None
        restored = False
        preserving_started = monotonic()
        preserved = self.fleet.controller.artifacts.put_text(
            request_id=swap_id,
            kind="swap_input",
            text=request.prompt,
            provenance={
                "type": "exclusive_swap_input",
                "parents": list(request.evidence_references),
            },
            metadata={
                "endpoint": self.endpoint_id,
                "candidate": request.candidate,
                "images": [str(path.resolve(strict=False)) for path in request.images],
                "evidence_references": list(request.evidence_references),
            },
        )
        phase_timings["preserving"] = self._phase(
            swap_id, "preserving", started=preserving_started, details={"artifact": preserved.id}
        )
        baseline_vram = self.fleet.status()["current_vram_bytes"]
        self.fleet.pause_admission(reason=f"exclusive escalation {swap_id}")
        try:
            started = monotonic()
            if cancellation.is_set():
                raise RequestCancelledError("Swap cancelled before drain")
            self.fleet.unload_all(timeout_seconds=120)
            phase_timings["draining"] = self._phase(swap_id, "draining", started=started)

            endpoint = self.fleet.registry.get(self.endpoint_id)
            validate_endpoint_artifacts(endpoint, self.fleet.config.paths["model_cache"])
            started = monotonic()
            reclaimed_vram = gpu_memory_used_bytes(endpoint.telemetry_gpu_index)
            tolerance = int(
                endpoint.proposed_budget.get(
                    "vram_reclaim_tolerance_bytes", 512 * 1024**2
                )
            )
            if reclaimed_vram > self.fleet.initial_vram_bytes + tolerance:
                raise ResourceAdmissionError(
                    f"VRAM was not reclaimed: {reclaimed_vram} bytes remain in use"
                )
            required_ram = int(endpoint.proposed_budget.get("process_ram_bytes", 0))
            if psutil.virtual_memory().available < required_ram:
                raise ResourceAdmissionError("Insufficient available system RAM for large model")
            phase_timings["reclaiming"] = self._phase(
                swap_id,
                "reclaiming",
                started=started,
                details={
                    "resident_vram_before_bytes": baseline_vram,
                    "reclaimed_vram_bytes": reclaimed_vram,
                    "available_ram_bytes": psutil.virtual_memory().available,
                },
            )

            started = monotonic()
            large_runtime = self.fleet.controller.create_runtime(endpoint)
            large_runtime.start()
            phase_timings["loading_large"] = self._phase(
                swap_id, "loading_large", started=started, details={"pid": large_runtime.pid}
            )

            if cancellation.is_set():
                raise RequestCancelledError("Swap generation cancelled")
            started = monotonic()
            answer = self.fleet.controller.smoke(
                endpoint_id=self.endpoint_id,
                prompt=request.prompt,
                original_prompt=request.prompt,
                images=request.images,
                evidence_references=request.evidence_references,
                runtime=large_runtime,
                keep_loaded=True,
                cancellation=cancellation,
                timeout_seconds=request.timeout_seconds,
            )
            phase_timings["solving"] = self._phase(
                swap_id,
                "solving",
                started=started,
                details={"request_id": answer.envelope.request_id},
            )
            if request.candidate is not None:
                # Candidate text is revealed only after the independent solve exists.
                adjudication = self.fleet.controller.smoke(
                    endpoint_id=self.endpoint_id,
                    prompt=(
                        "Compare the independent solution and candidate. Preserve controller "
                        "policy and permissions. Identify concrete errors, then choose the more "
                        "accurate answer.\n\nIndependent solution:\n"
                        f"{answer.answer}\n\nCandidate:\n{request.candidate}"
                    ),
                    original_prompt=request.prompt,
                    runtime=large_runtime,
                    keep_loaded=True,
                    cancellation=cancellation,
                    timeout_seconds=request.timeout_seconds,
                )
            started = monotonic()
            selected = adjudication or answer
            if selected.envelope.status != "answer" or selected.answer is None:
                raise RequestFailedError("Large-model response failed controller verification")
            if request.expected_contains and request.expected_contains not in selected.answer:
                raise RequestFailedError(
                    "Large-model response failed expected-content verification"
                )
            phase_timings["verifying"] = self._phase(
                swap_id,
                "verifying",
                started=started,
                details={"status": selected.envelope.status, "expected_contains_passed": True},
            )
        except BaseException as exc:
            error = str(exc)
        finally:
            try:
                started = monotonic()
                if large_runtime is not None:
                    large_runtime.stop()
                phase_timings["unloading_large"] = self._phase(
                    swap_id, "unloading_large", started=started
                )
                started = monotonic()
                restored = self._restore_residents()
                if not restored:
                    raise RequestFailedError("Resident fleet restoration was incomplete")
                phase_timings["restoring"] = self._phase(
                    swap_id, "restoring", started=started, details={"healthy": True}
                )
            except BaseException as restore_exc:
                error = f"{error}; restoration failed: {restore_exc}" if error else str(restore_exc)
                restored = False
            if restored:
                self.fleet.resume_admission(reason=f"exclusive escalation {swap_id} restored")
            terminal = "healthy" if error is None and restored else "failed"
            phase_timings[terminal] = self._phase(
                swap_id, terminal, started=preserving_started, details={"error": error}
            )
            self._lock.release()
        return SwapResult(
            swap_id=swap_id,
            status="answer" if error is None else "failed",
            result=answer,
            adjudication=adjudication,
            preserved_artifact=preserved.id,
            phase_timings_ms=phase_timings,
            fleet_restored=restored,
            error=error,
        )
