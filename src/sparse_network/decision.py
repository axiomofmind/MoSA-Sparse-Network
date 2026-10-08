"""Typed decision endpoints used for bounded routing and evaluation."""

from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from time import monotonic
from typing import Any, TextIO

from .config import AppConfig
from .errors import ConfigurationError, RequestFailedError, RuntimeUnavailableError
from .models import EndpointDefinition, ModelRegistry
from .validation import validate_endpoint_artifacts


@dataclass(frozen=True)
class DecisionResult:
    endpoint: str
    revision: str
    answers: dict[str, Any]
    usage: dict[str, Any]
    elapsed_ms: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "sparse-network-decision-result.v1",
            **asdict(self),
        }


def choice_probabilities(result: DecisionResult, question: str) -> dict[str, float]:
    answer = result.answers.get(question)
    if not isinstance(answer, dict) or answer.get("type") != "choice":
        raise RequestFailedError(f"Decision answer {question!r} is not a choice")
    raw = answer.get("probabilities")
    if not isinstance(raw, dict) or not raw:
        raise RequestFailedError(f"Decision answer {question!r} has no probabilities")
    probabilities: dict[str, float] = {}
    for name, value in raw.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RequestFailedError("Decision probability is not numeric")
        score = float(value)
        if not 0 <= score <= 1:
            raise RequestFailedError("Decision probability is outside [0, 1]")
        probabilities[str(name)] = score
    return probabilities


class WorkerDecisionBackend:
    def __init__(
        self,
        *,
        endpoint: EndpointDefinition,
        executable: str,
        model_path: Path,
        worker_path: Path,
        log_path: Path,
        startup_timeout_seconds: float,
    ) -> None:
        self.endpoint = endpoint
        self.endpoint_id = endpoint.id
        self.revision = endpoint.model_revision or "unversioned"
        self.executable = executable
        self.model_path = model_path
        self.worker_path = worker_path
        self.log_path = log_path
        self.startup_timeout_seconds = startup_timeout_seconds
        self._process: subprocess.Popen[str] | None = None
        self._log_handle: TextIO | None = None
        self._messages: queue.Queue[dict[str, Any]] = queue.Queue()
        self._reader: threading.Thread | None = None

    @property
    def pid(self) -> int | None:
        return self._process.pid if self._process is not None else None

    def _resolved_executable(self) -> str:
        candidate = Path(self.executable)
        if candidate.exists():
            return str(candidate.resolve())
        located = shutil.which(self.executable)
        if located:
            return located
        raise RuntimeUnavailableError(f"Decision Python executable not found: {self.executable}")

    def _read_messages(self) -> None:
        process = self._process
        if process is None or process.stdout is None:
            return
        for line in process.stdout:
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                message = {"event": "protocol_error", "error": line.rstrip()}
            if isinstance(message, dict):
                self._messages.put(message)

    def _next_message(self, timeout: float) -> dict[str, Any]:
        try:
            return self._messages.get(timeout=max(0.05, timeout))
        except queue.Empty as exc:
            raise TimeoutError from exc

    def _log_tail(self) -> str:
        try:
            content = self.log_path.read_text(encoding="utf-8", errors="replace")
            return "\n".join(content.splitlines()[-60:])
        except OSError:
            return ""

    def start(self) -> None:
        if self._process is not None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = self.log_path.open("a", encoding="utf-8")
        runtime = self.endpoint.runtime
        command = [
            self._resolved_executable(),
            str(self.worker_path),
            "--model",
            str(self.model_path),
            "--device",
            str(runtime.get("device", "cuda:0")),
            "--memory-fraction",
            str(float(runtime.get("gpu_memory_fraction", 0.4))),
        ]
        if bool(runtime.get("trust_remote_code", False)):
            command.append("--trust-remote-code")
        environment = os.environ.copy()
        environment.update(self.endpoint.environment)
        environment["PYTHONNOUSERSITE"] = "1"
        creation_flags = int(getattr(subprocess, "CREATE_NO_WINDOW", 0)) if os.name == "nt" else 0
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._log_handle,
            text=True,
            encoding="utf-8",
            bufsize=1,
            env=environment,
            creationflags=creation_flags,
        )
        self._reader = threading.Thread(target=self._read_messages, daemon=True)
        self._reader.start()
        deadline = monotonic() + self.startup_timeout_seconds
        while monotonic() < deadline:
            process = self._process
            if process.poll() is not None and self._messages.empty():
                self.stop()
                raise RuntimeUnavailableError(f"Decision worker exited early: {self._log_tail()}")
            try:
                message = self._next_message(min(0.25, deadline - monotonic()))
            except TimeoutError:
                continue
            if message.get("event") == "ready":
                return
            if message.get("event") in {"startup_error", "protocol_error"}:
                self.stop()
                raise RuntimeUnavailableError(f"Decision worker failed: {message.get('error')}")
        self.stop()
        raise RuntimeUnavailableError("Decision worker startup timed out")

    def decide(
        self,
        state: Any,
        questions: dict[str, Any],
        *,
        images: tuple[Path, ...] = (),
    ) -> DecisionResult:
        process = self._process
        if process is None or process.stdin is None:
            raise RequestFailedError("Decision worker is not running")
        if not questions:
            raise ValueError("questions must not be empty")
        resolved_images = [str(path.resolve(strict=True)) for path in images]
        process.stdin.write(
            json.dumps(
                {
                    "command": "decide",
                    "state": state,
                    "questions": questions,
                    "images": resolved_images,
                }
            )
            + "\n"
        )
        process.stdin.flush()
        deadline = monotonic() + 300
        while monotonic() < deadline:
            if process.poll() is not None and self._messages.empty():
                raise RequestFailedError(f"Decision worker exited: {self._log_tail()}")
            try:
                message = self._next_message(min(0.25, deadline - monotonic()))
            except TimeoutError:
                continue
            if message.get("event") == "error":
                raise RequestFailedError(f"Decision failed: {message.get('error')}")
            if message.get("event") == "decision":
                raw = message.get("result")
                if not isinstance(raw, dict) or not isinstance(raw.get("answers"), dict):
                    raise RequestFailedError("Decision worker returned an invalid result")
                usage = raw.get("usage", {})
                return DecisionResult(
                    endpoint=self.endpoint_id,
                    revision=self.revision,
                    answers=raw["answers"],
                    usage=usage if isinstance(usage, dict) else {},
                    elapsed_ms=int(message.get("elapsed_ms", 0)),
                )
        raise RequestFailedError("Decision request timed out")

    def stop(self) -> None:
        process = self._process
        if process is not None and process.poll() is None and process.stdin is not None:
            try:
                process.stdin.write('{"command":"shutdown"}\n')
                process.stdin.flush()
                process.wait(timeout=10)
            except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                process.kill()
                process.wait(timeout=5)
        self._process = None
        if self._reader is not None:
            self._reader.join(timeout=1)
            self._reader = None
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None


def create_decision_backend(
    config: AppConfig,
    registry: ModelRegistry,
    endpoint_id: str,
) -> WorkerDecisionBackend:
    endpoint = registry.get(endpoint_id)
    if endpoint.adapter != "decision_transformers":
        raise ConfigurationError(f"Endpoint {endpoint_id} is not a decision endpoint")
    validate_endpoint_artifacts(endpoint, config.paths["model_cache"])
    paths = endpoint.artifact_paths(config.paths["model_cache"])
    config_path = paths.get("config")
    if config_path is None:
        raise ConfigurationError(f"Decision endpoint {endpoint_id} has no config artifact")
    logs = config.paths["logs"]
    if logs is None:
        raise ConfigurationError("paths.logs must be configured")
    controller = config.data.get("controller", {})
    return WorkerDecisionBackend(
        endpoint=endpoint,
        executable=config.runtime_executable("decisions"),
        model_path=config_path.parent,
        worker_path=config.root / "src" / "sparse_network" / "decision_worker.py",
        log_path=logs / f"{endpoint.id}-decision.log",
        startup_timeout_seconds=float(controller.get("startup_timeout_seconds", 120)),
    )
