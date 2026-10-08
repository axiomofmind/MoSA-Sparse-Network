"""Lifecycle adapter for the isolated Transformers JSON-lines worker."""

from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import threading
from pathlib import Path
from time import monotonic
from typing import Any, TextIO

from sparse_network.errors import (
    EndpointStateError,
    RequestCancelledError,
    RequestFailedError,
    RuntimeUnavailableError,
)
from sparse_network.models import EndpointDefinition

from .base import RuntimeAdapter, RuntimeResult, RuntimeState


class TransformersAdapter(RuntimeAdapter):
    def __init__(
        self,
        endpoint: EndpointDefinition,
        *,
        executable: str,
        model_path: Path,
        worker_path: Path,
        log_path: Path,
        startup_timeout_seconds: float,
        shutdown_grace_seconds: float,
    ) -> None:
        super().__init__(endpoint)
        self.executable = executable
        self.model_path = model_path
        self.worker_path = worker_path
        self.log_path = log_path
        self.startup_timeout_seconds = startup_timeout_seconds
        self.shutdown_grace_seconds = shutdown_grace_seconds
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
        raise RuntimeUnavailableError(
            f"Transformers Python executable not found: {self.executable}"
        )

    def _command(self) -> list[str]:
        runtime = self.endpoint.runtime
        command = [
            self._resolved_executable(),
            str(self.worker_path),
            "--model",
            str(self.model_path),
            "--device",
            str(runtime.get("device", "cuda:0")),
            "--memory-fraction",
            str(float(runtime.get("gpu_memory_fraction", 0.75))),
        ]
        if bool(runtime.get("thinking", False)):
            command.append("--thinking")
        if "image" in self.endpoint.modalities:
            command.append("--vision")
        command.extend(
            [
                "--attention-implementation",
                str(runtime.get("attention_implementation", "eager")),
            ]
        )
        return command

    def _read_messages(self) -> None:
        process = self._process
        if process is None or process.stdout is None:
            return
        for line in process.stdout:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                value = {"event": "protocol_error", "error": line.rstrip()}
            if isinstance(value, dict):
                self._messages.put(value)

    def _next_message(self, timeout: float) -> dict[str, Any]:
        try:
            return self._messages.get(timeout=max(0.05, timeout))
        except queue.Empty as exc:
            raise TimeoutError from exc

    def _log_tail(self, lines: int = 60) -> str:
        try:
            content = self.log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return ""
        return "\n".join(content[-lines:])

    def start(self) -> None:
        if self.state not in {RuntimeState.STOPPED, RuntimeState.FAILED}:
            raise EndpointStateError(f"Cannot start {self.endpoint.id} from {self.state}")
        self.state = RuntimeState.STARTING
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = self.log_path.open("a", encoding="utf-8")
        environment = os.environ.copy()
        environment.update(self.endpoint.environment)
        environment["PYTHONNOUSERSITE"] = "1"
        creation_flags = int(getattr(subprocess, "CREATE_NO_WINDOW", 0)) if os.name == "nt" else 0
        try:
            self._process = subprocess.Popen(
                self._command(),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self._log_handle,
                text=True,
                encoding="utf-8",
                bufsize=1,
                env=environment,
                creationflags=creation_flags,
            )
        except OSError as exc:
            self.state = RuntimeState.FAILED
            self._close_log()
            raise RuntimeUnavailableError(f"Unable to launch Transformers worker: {exc}") from exc
        self._reader = threading.Thread(target=self._read_messages, daemon=True)
        self._reader.start()
        deadline = monotonic() + self.startup_timeout_seconds
        while monotonic() < deadline:
            if self._process.poll() is not None and self._messages.empty():
                self.state = RuntimeState.FAILED
                raise RuntimeUnavailableError(
                    f"Transformers worker exited with {self._process.returncode}: "
                    f"{self._log_tail()}"
                )
            try:
                message = self._next_message(min(0.25, deadline - monotonic()))
            except TimeoutError:
                continue
            if message.get("event") == "ready":
                self.state = RuntimeState.READY
                return
            if message.get("event") in {"startup_error", "protocol_error"}:
                self.state = RuntimeState.FAILED
                raise RuntimeUnavailableError(
                    f"Transformers worker failed: {message.get('error')}\n"
                    f"{message.get('traceback', '')}"
                )
        self.state = RuntimeState.FAILED
        self.stop()
        raise RuntimeUnavailableError("Transformers worker startup timed out")

    def generate(
        self,
        *,
        prompt: str,
        images: tuple[Path, ...],
        cancellation: threading.Event,
        timeout_seconds: float,
    ) -> RuntimeResult:
        if images and "image" not in self.endpoint.modalities:
            raise RequestFailedError("Transformers text worker does not accept images")
        if self.state != RuntimeState.READY:
            raise EndpointStateError(f"Endpoint is not ready: {self.state}")
        process = self._process
        if process is None or process.stdin is None:
            raise EndpointStateError("Transformers worker has no input stream")
        if cancellation.is_set():
            raise RequestCancelledError("Request cancelled before submission")
        self.state = RuntimeState.BUSY
        request = {
            "command": "generate",
            "prompt": prompt,
            "images": [str(image) for image in images],
            "max_tokens": self.endpoint.max_output_tokens,
        }
        process.stdin.write(json.dumps(request) + "\n")
        process.stdin.flush()
        deadline = monotonic() + timeout_seconds
        try:
            while monotonic() < deadline:
                if cancellation.is_set():
                    self.cancel()
                    raise RequestCancelledError("Transformers request cancelled")
                if process.poll() is not None and self._messages.empty():
                    raise RequestFailedError(
                        f"Transformers worker exited with {process.returncode}: {self._log_tail()}"
                    )
                try:
                    message = self._next_message(min(0.1, deadline - monotonic()))
                except TimeoutError:
                    continue
                if message.get("event") == "error":
                    raise RequestFailedError(
                        f"Transformers generation failed: {message.get('error')}\n"
                        f"{message.get('traceback', '')}"
                    )
                if message.get("event") == "result":
                    return RuntimeResult(
                        text=str(message.get("text", "")),
                        input_tokens=int(message.get("input_tokens", 0)),
                        output_tokens=int(message.get("output_tokens", 0)),
                        finish_reason=str(message.get("finish_reason", "stop")),
                        raw=message,
                    )
            self.cancel()
            raise RequestFailedError("Transformers request timed out")
        finally:
            if self.state == RuntimeState.BUSY:
                self.state = RuntimeState.READY

    def count_input_tokens(
        self,
        *,
        prompt: str,
        images: tuple[Path, ...],
        timeout_seconds: float,
    ) -> int | None:
        if self.state != RuntimeState.READY:
            raise EndpointStateError(f"Endpoint is not ready: {self.state}")
        process = self._process
        if process is None or process.stdin is None:
            raise EndpointStateError("Transformers worker has no input stream")
        process.stdin.write(
            json.dumps(
                {
                    "command": "count_tokens",
                    "prompt": prompt,
                    "images": [str(image) for image in images],
                }
            )
            + "\n"
        )
        process.stdin.flush()
        deadline = monotonic() + timeout_seconds
        while monotonic() < deadline:
            if process.poll() is not None and self._messages.empty():
                raise RequestFailedError(
                    f"Transformers worker exited with {process.returncode}: {self._log_tail()}"
                )
            try:
                message = self._next_message(min(0.1, deadline - monotonic()))
            except TimeoutError:
                continue
            if message.get("event") == "error":
                raise RequestFailedError(
                    f"Transformers tokenization failed: {message.get('error')}\n"
                    f"{message.get('traceback', '')}"
                )
            if message.get("event") == "token_count":
                return int(message["input_tokens"])
        raise RequestFailedError("Transformers tokenization timed out")

    def cancel(self) -> None:
        process = self._process
        if process is not None and process.poll() is None:
            process.terminate()
            self.state = RuntimeState.FAILED

    def _close_log(self) -> None:
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None

    def stop(self) -> None:
        process = self._process
        if process is None:
            self.state = RuntimeState.STOPPED
            self._close_log()
            return
        self.state = RuntimeState.DRAINING
        if process.poll() is None and process.stdin is not None:
            try:
                process.stdin.write('{"command":"shutdown"}\n')
                process.stdin.flush()
                process.wait(timeout=self.shutdown_grace_seconds)
            except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                process.terminate()
                try:
                    process.wait(timeout=self.shutdown_grace_seconds)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
        self._process = None
        if self._reader is not None:
            self._reader.join(timeout=1)
            self._reader = None
        self._close_log()
        self.state = RuntimeState.STOPPED
