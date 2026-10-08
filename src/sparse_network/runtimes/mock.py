"""Deterministic runtime fixture used by CI and controller development."""

from __future__ import annotations

import threading
from pathlib import Path
from time import monotonic, sleep

from sparse_network.errors import EndpointStateError, RequestCancelledError, RequestFailedError

from .base import RuntimeAdapter, RuntimeResult, RuntimeState


class MockAdapter(RuntimeAdapter):
    @property
    def pid(self) -> int | None:
        return None

    def start(self) -> None:
        if self.state not in {RuntimeState.STOPPED, RuntimeState.FAILED}:
            raise EndpointStateError(f"Cannot start mock endpoint from {self.state}")
        self.state = RuntimeState.STARTING
        self.state = RuntimeState.READY

    def generate(
        self,
        *,
        prompt: str,
        images: tuple[Path, ...],
        cancellation: threading.Event,
        timeout_seconds: float,
    ) -> RuntimeResult:
        if self.state != RuntimeState.READY:
            raise EndpointStateError(f"Mock endpoint is not ready: {self.state}")
        self.state = RuntimeState.BUSY
        started = monotonic()
        try:
            if prompt.startswith("[mock:failure]"):
                raise RequestFailedError("Simulated mock runtime failure")
            if prompt.startswith("[mock:slow]"):
                while monotonic() - started < min(timeout_seconds + 1, 10):
                    if cancellation.wait(0.02):
                        raise RequestCancelledError("Mock request cancelled")
                    sleep(0.01)
                raise RequestFailedError("Simulated mock request timeout")
            if cancellation.is_set():
                raise RequestCancelledError("Mock request cancelled")
            endpoint_loop = prompt.startswith(f"[mock:loop:{self.endpoint.id}]")
            if prompt.startswith("[mock:loop]") or endpoint_loop:
                text = "Useful prefix. " + ("Wait, reconsider. " * 20).strip()
            elif "Return only one JSON object with array keys errors" in prompt:
                text = '{"errors": [], "evidence": [], "recommendation": "accept"}'
            else:
                image_note = f" [{len(images)} image(s)]" if images else ""
                text = f"Mock response: {prompt}{image_note}"
            return RuntimeResult(
                text=text,
                input_tokens=max(1, len(prompt.split())),
                output_tokens=max(1, len(text.split())),
                finish_reason="stop",
                raw={"mock": True},
            )
        finally:
            if self.state == RuntimeState.BUSY:
                self.state = RuntimeState.READY

    def cancel(self) -> None:
        return None

    def stop(self) -> None:
        self.state = RuntimeState.STOPPED
