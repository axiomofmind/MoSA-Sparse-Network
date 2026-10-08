"""Lifecycle adapter for the isolated PaddleOCR JSON-lines worker."""

from __future__ import annotations

from pathlib import Path

from sparse_network.models import EndpointDefinition

from .transformers import TransformersAdapter


class PaddleOcrAdapter(TransformersAdapter):
    def __init__(
        self,
        endpoint: EndpointDefinition,
        *,
        executable: str,
        detection_model_path: Path,
        recognition_model_path: Path,
        worker_path: Path,
        log_path: Path,
        startup_timeout_seconds: float,
        shutdown_grace_seconds: float,
    ) -> None:
        super().__init__(
            endpoint,
            executable=executable,
            model_path=detection_model_path,
            worker_path=worker_path,
            log_path=log_path,
            startup_timeout_seconds=startup_timeout_seconds,
            shutdown_grace_seconds=shutdown_grace_seconds,
        )
        self.detection_model_path = detection_model_path
        self.recognition_model_path = recognition_model_path

    def _command(self) -> list[str]:
        return [
            self._resolved_executable(),
            str(self.worker_path),
            "--detection-model",
            str(self.detection_model_path),
            "--recognition-model",
            str(self.recognition_model_path),
            "--device",
            str(self.endpoint.runtime.get("device", "gpu:0")),
        ]

    def count_input_tokens(
        self,
        *,
        prompt: str,
        images: tuple[Path, ...],
        timeout_seconds: float,
    ) -> None:
        del prompt, images, timeout_seconds
        return None
