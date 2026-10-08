"""Controller-owned request and response contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .errors import RequestFailedError

VALID_STATUSES = {
    "answer",
    "needs_tool",
    "needs_verification",
    "needs_escalation",
    "failed",
    "cancelled",
}


@dataclass(frozen=True)
class ResourceUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    elapsed_ms: int = 0
    peak_vram_bytes: int = 0
    peak_ram_bytes: int = 0
    cold_load_ms: int = 0


@dataclass(frozen=True)
class EndpointEnvelope:
    schema: str
    request_id: str
    execution_id: str
    endpoint: str
    model_revision: str | None
    status: str
    capability: str | None
    modalities_consumed: tuple[str, ...]
    answer_reference: str | None
    evidence_references: tuple[str, ...] = ()
    verification: dict[str, Any] = field(default_factory=dict)
    repetition: dict[str, Any] = field(default_factory=dict)
    escalation: dict[str, Any] = field(default_factory=dict)
    resource_usage: ResourceUsage = field(default_factory=ResourceUsage)

    def validate(self) -> None:
        if self.schema != "sparse-network-envelope.v2":
            raise RequestFailedError(f"Unsupported envelope schema: {self.schema}")
        for field_name in ("request_id", "execution_id", "endpoint"):
            if not getattr(self, field_name):
                raise RequestFailedError(f"Envelope field {field_name} cannot be empty")
        if self.status not in VALID_STATUSES:
            raise RequestFailedError(f"Invalid envelope status: {self.status}")
        if self.status == "answer" and not self.answer_reference:
            raise RequestFailedError("An answer envelope needs answer_reference")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        value = asdict(self)
        value["modalities_consumed"] = list(self.modalities_consumed)
        value["evidence_references"] = list(self.evidence_references)
        return value


@dataclass(frozen=True)
class SmokeResult:
    envelope: EndpointEnvelope
    answer: str | None
    lifecycle: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "envelope": self.envelope.to_dict(),
            "answer": self.answer,
            "lifecycle": self.lifecycle,
        }
