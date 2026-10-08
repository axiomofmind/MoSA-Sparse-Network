"""Runtime-neutral endpoint interface."""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from sparse_network.models import EndpointDefinition


class RuntimeState(StrEnum):
    STOPPED = "stopped"
    STARTING = "starting"
    READY = "ready"
    BUSY = "busy"
    DRAINING = "draining"
    FAILED = "failed"


@dataclass(frozen=True)
class RuntimeResult:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    finish_reason: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


class RuntimeAdapter(ABC):
    def __init__(self, endpoint: EndpointDefinition) -> None:
        self.endpoint = endpoint
        self.state = RuntimeState.STOPPED

    @property
    @abstractmethod
    def pid(self) -> int | None:
        raise NotImplementedError

    @abstractmethod
    def start(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def generate(
        self,
        *,
        prompt: str,
        images: tuple[Path, ...],
        cancellation: threading.Event,
        timeout_seconds: float,
    ) -> RuntimeResult:
        raise NotImplementedError

    def count_input_tokens(
        self,
        *,
        prompt: str,
        images: tuple[Path, ...],
        timeout_seconds: float,
    ) -> int | None:
        """Return the runtime's input-token count when its tokenizer is available."""

        return None

    @abstractmethod
    def cancel(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def stop(self) -> None:
        raise NotImplementedError
