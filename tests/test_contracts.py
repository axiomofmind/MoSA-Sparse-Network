from __future__ import annotations

import pytest

from sparse_network.contracts import EndpointEnvelope, ResourceUsage
from sparse_network.errors import RequestFailedError


def envelope(**overrides: object) -> EndpointEnvelope:
    values = {
        "schema": "sparse-network-envelope.v2",
        "request_id": "request-1",
        "execution_id": "execution-1",
        "endpoint": "mock-echo",
        "model_revision": "builtin",
        "status": "answer",
        "capability": "text_generation",
        "modalities_consumed": ("text",),
        "answer_reference": "artifact-1",
        "verification": {},
        "repetition": {},
        "escalation": {},
        "resource_usage": ResourceUsage(),
    }
    values.update(overrides)
    return EndpointEnvelope(**values)  # type: ignore[arg-type]


def test_valid_envelope_serializes_lists() -> None:
    result = envelope().to_dict()
    assert result["schema"] == "sparse-network-envelope.v2"
    assert result["modalities_consumed"] == ["text"]


def test_answer_requires_artifact_reference() -> None:
    with pytest.raises(RequestFailedError):
        envelope(answer_reference=None).validate()


def test_failed_envelope_may_omit_answer() -> None:
    envelope(status="failed", answer_reference=None).validate()
