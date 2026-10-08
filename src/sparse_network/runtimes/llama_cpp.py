"""Lifecycle adapter for a local llama.cpp OpenAI-compatible server."""

from __future__ import annotations

import base64
import http.client
import json
import mimetypes
import os
import shutil
import socket
import subprocess
import threading
from contextlib import suppress
from pathlib import Path
from time import monotonic, sleep
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from sparse_network.errors import (
    EndpointStateError,
    RequestCancelledError,
    RequestFailedError,
    RuntimeUnavailableError,
)
from sparse_network.models import EndpointDefinition
from sparse_network.repetition import (
    RepetitionDetectedError,
    RepetitionPolicy,
    detect_repetition,
)

from .base import RuntimeAdapter, RuntimeResult, RuntimeState


def _free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


class LlamaCppAdapter(RuntimeAdapter):
    def __init__(
        self,
        endpoint: EndpointDefinition,
        *,
        executable: str,
        model_path: Path,
        projector_path: Path | None,
        log_path: Path,
        host: str,
        startup_timeout_seconds: float,
        shutdown_grace_seconds: float,
        repetition_policy: RepetitionPolicy,
    ) -> None:
        super().__init__(endpoint)
        self.executable = executable
        self.model_path = model_path
        self.projector_path = projector_path
        self.log_path = log_path
        self.host = host
        self.port = 0
        self.startup_timeout_seconds = startup_timeout_seconds
        self.shutdown_grace_seconds = shutdown_grace_seconds
        self.repetition_policy = repetition_policy
        self._process: subprocess.Popen[bytes] | None = None
        self._log_handle: Any | None = None
        self._active_connection: http.client.HTTPConnection | None = None
        self._connection_lock = threading.Lock()

    @property
    def pid(self) -> int | None:
        return self._process.pid if self._process is not None else None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def _resolved_executable(self) -> str:
        candidate = Path(self.executable)
        if candidate.exists():
            return str(candidate.resolve())
        located = shutil.which(self.executable)
        if located:
            return located
        raise RuntimeUnavailableError(f"llama.cpp executable not found: {self.executable}")

    def _command(self) -> list[str]:
        runtime = self.endpoint.runtime
        command = [
            self._resolved_executable(),
            "--model",
            str(self.model_path),
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--ctx-size",
            str(self.endpoint.context_size),
            "--n-predict",
            str(self.endpoint.max_output_tokens),
            "--n-gpu-layers",
            str(int(runtime.get("gpu_layers", 0))),
            "--parallel",
            str(int(runtime.get("parallel_slots", 1))),
            "--batch-size",
            str(int(runtime.get("batch_size", 512))),
            "--ubatch-size",
            str(int(runtime.get("micro_batch_size", 128))),
            "--metrics",
            "--no-webui",
        ]
        if self.projector_path is not None:
            command.extend(["--mmproj", str(self.projector_path)])
        return command

    def _log_tail(self, lines: int = 40) -> str:
        try:
            content = self.log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return ""
        return "\n".join(content[-lines:])

    def start(self) -> None:
        if self.state not in {RuntimeState.STOPPED, RuntimeState.FAILED}:
            raise EndpointStateError(f"Cannot start {self.endpoint.id} from {self.state}")
        if not self.model_path.is_file():
            raise RuntimeUnavailableError(f"Model file not found: {self.model_path}")
        self.state = RuntimeState.STARTING
        self.port = _free_port(self.host)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = self.log_path.open("wb")
        environment = os.environ.copy()
        environment.update(self.endpoint.environment)
        creation_flags = int(getattr(subprocess, "CREATE_NO_WINDOW", 0)) if os.name == "nt" else 0
        try:
            self._process = subprocess.Popen(
                self._command(),
                stdout=self._log_handle,
                stderr=subprocess.STDOUT,
                env=environment,
                creationflags=creation_flags,
            )
        except OSError as exc:
            self.state = RuntimeState.FAILED
            self._close_log()
            raise RuntimeUnavailableError(f"Unable to launch llama.cpp: {exc}") from exc

        deadline = monotonic() + self.startup_timeout_seconds
        last_error = "server did not become ready"
        while monotonic() < deadline:
            if self._process.poll() is not None:
                self.state = RuntimeState.FAILED
                tail = self._log_tail()
                self._close_log()
                raise RuntimeUnavailableError(
                    f"llama.cpp exited with {self._process.returncode}: {tail}"
                )
            try:
                with urlopen(f"{self.base_url}/health", timeout=1) as response:
                    if response.status == 200:
                        self.state = RuntimeState.READY
                        return
            except (HTTPError, URLError, TimeoutError, OSError) as exc:
                last_error = str(exc)
            sleep(0.25)

        self.state = RuntimeState.FAILED
        tail = self._log_tail()
        self.stop()
        raise RuntimeUnavailableError(
            f"llama.cpp startup timed out ({last_error}). Log tail:\n{tail}"
        )

    def generate(
        self,
        *,
        prompt: str,
        images: tuple[Path, ...],
        cancellation: threading.Event,
        timeout_seconds: float,
    ) -> RuntimeResult:
        if self.state != RuntimeState.READY:
            raise EndpointStateError(f"Endpoint is not ready: {self.state}")
        if cancellation.is_set():
            raise RequestCancelledError("Request cancelled before submission")
        self.state = RuntimeState.BUSY
        connection = http.client.HTTPConnection(self.host, self.port, timeout=timeout_seconds)
        with self._connection_lock:
            self._active_connection = connection
        cancellation_done = threading.Event()

        def watch_cancellation() -> None:
            while not cancellation_done.wait(0.05):
                if cancellation.is_set():
                    with suppress(OSError):
                        connection.close()
                    return

        watcher = threading.Thread(target=watch_cancellation, daemon=True)
        watcher.start()
        message_content: str | list[dict[str, Any]] = prompt
        if images:
            content_parts: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
            for path in images:
                media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
                encoded = base64.b64encode(path.read_bytes()).decode("ascii")
                content_parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{media_type};base64,{encoded}"},
                    }
                )
            message_content = content_parts
        request_data: dict[str, Any] = {
                "model": self.endpoint.id,
                "messages": [{"role": "user", "content": message_content}],
                "temperature": 0,
                "max_tokens": self.endpoint.max_output_tokens,
                "cache_prompt": False,
                "stream": True,
                "stream_options": {"include_usage": True},
        }
        if "thinking" in self.endpoint.runtime:
            thinking = bool(self.endpoint.runtime["thinking"])
            request_data["chat_template_kwargs"] = {"enable_thinking": thinking}
            if not thinking:
                request_data["reasoning_effort"] = "none"
        payload = json.dumps(request_data).encode("utf-8")
        try:
            connection.request(
                "POST",
                "/v1/chat/completions",
                body=payload,
                headers={"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            if response.status != 200:
                body = response.read()
                raise RequestFailedError(
                    f"llama.cpp returned HTTP {response.status}: "
                    f"{body.decode('utf-8', errors='replace')[:1000]}"
                )
            text_parts: list[str] = []
            input_tokens = 0
            output_tokens = 0
            finish_reason: str | None = None
            chunks = 0
            last_repetition_check = 0
            current_length = 0
            while True:
                if cancellation.is_set():
                    raise RequestCancelledError("Request cancelled")
                line = response.readline()
                if not line:
                    break
                stripped = line.strip()
                if not stripped or not stripped.startswith(b"data:"):
                    continue
                data = stripped[5:].strip()
                if data == b"[DONE]":
                    break
                try:
                    decoded = json.loads(data)
                except json.JSONDecodeError as exc:
                    raise RequestFailedError(
                        "llama.cpp returned malformed streaming completion JSON"
                    ) from exc
                chunks += 1
                usage = decoded.get("usage") or {}
                input_tokens = max(input_tokens, int(usage.get("prompt_tokens", 0)))
                output_tokens = max(output_tokens, int(usage.get("completion_tokens", 0)))
                choices = decoded.get("choices") or []
                if choices:
                    choice = choices[0]
                    delta = choice.get("delta") or {}
                    if content := delta.get("content"):
                        content_text = str(content)
                        text_parts.append(content_text)
                        current_length += len(content_text)
                        if (
                            current_length - last_repetition_check
                            >= self.repetition_policy.streaming_check_interval_characters
                        ):
                            current_text = "".join(text_parts)
                            hit = detect_repetition(current_text, self.repetition_policy)
                            last_repetition_check = current_length
                            if hit is not None:
                                raise RepetitionDetectedError(
                                    current_text,
                                    hit,
                                    streaming=True,
                                )
                    if choice.get("finish_reason") is not None:
                        finish_reason = str(choice["finish_reason"])
            if cancellation.is_set():
                raise RequestCancelledError("Request cancelled")
            if chunks == 0:
                raise RequestFailedError("llama.cpp returned an empty completion stream")
            return RuntimeResult(
                text="".join(text_parts),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                finish_reason=finish_reason,
                raw={"streamed": True, "chunks": chunks},
            )
        except RequestCancelledError:
            raise
        except (OSError, TimeoutError, http.client.HTTPException, AttributeError) as exc:
            if cancellation.is_set():
                raise RequestCancelledError("Request cancelled") from exc
            raise RequestFailedError(f"llama.cpp request failed: {exc}") from exc
        finally:
            cancellation_done.set()
            watcher.join(timeout=1)
            connection.close()
            with self._connection_lock:
                self._active_connection = None
            if self.state == RuntimeState.BUSY:
                self.state = RuntimeState.READY

    def cancel(self) -> None:
        with self._connection_lock:
            connection = self._active_connection
        if connection is not None:
            with suppress(OSError):
                connection.close()

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
        self.cancel()
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=self.shutdown_grace_seconds)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        self._process = None
        self._close_log()
        self.state = RuntimeState.STOPPED
