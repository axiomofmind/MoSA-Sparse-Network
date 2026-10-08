from __future__ import annotations

import sys
from pathlib import Path

import pytest

from sparse_network.decision import (
    DecisionResult,
    WorkerDecisionBackend,
    choice_probabilities,
)
from sparse_network.errors import RequestFailedError
from sparse_network.models import EndpointDefinition, ModelSource


def endpoint() -> EndpointDefinition:
    return EndpointDefinition(
        id="decision-fixture",
        display_name="Decision fixture",
        role="fixture",
        family="fixture",
        source=ModelSource(type="builtin"),
        runtime={
            "adapter": "decision_transformers",
            "device": "cpu",
            "trust_remote_code": True,
        },
        modalities=("text", "image"),
        capabilities=("route_scoring",),
        context_size=128,
        max_output_tokens=0,
        environment={},
        telemetry_gpu_index=None,
        admission={"state": "candidate"},
        license={},
    )


def fake_worker(path: Path) -> None:
    path.write_text(
        """import json, sys
print(json.dumps({'event': 'ready'}), flush=True)
for line in sys.stdin:
    request = json.loads(line)
    if request.get('command') == 'shutdown':
        raise SystemExit(0)
    result = {
        'answers': {
            'lane': {
                'type': 'choice',
                'choice': 'coding_repair',
                'confidence': 0.8,
                'probabilities': {'coding_repair': 0.8, 'general_generation': 0.2},
            }
        },
        'usage': {'input_tokens': 12, 'output_tokens': 0},
    }
    print(json.dumps({'event': 'decision', 'result': result, 'elapsed_ms': 7}), flush=True)
""",
        encoding="utf-8",
    )


def test_decision_worker_lifecycle_and_choice_probabilities(tmp_path: Path) -> None:
    worker = tmp_path / "worker.py"
    fake_worker(worker)
    backend = WorkerDecisionBackend(
        endpoint=endpoint(),
        executable=sys.executable,
        model_path=tmp_path,
        worker_path=worker,
        log_path=tmp_path / "worker.log",
        startup_timeout_seconds=3,
    )
    backend.start()
    result = backend.decide(
        "Fix this Python function",
        {"lane": {"type": "choice", "instructions": "Choose", "criteria": {}}},
    )
    assert choice_probabilities(result, "lane") == {
        "coding_repair": 0.8,
        "general_generation": 0.2,
    }
    assert result.usage["output_tokens"] == 0
    assert backend.pid is not None
    backend.stop()
    assert backend.pid is None


def test_choice_probabilities_rejects_untyped_answer() -> None:
    result = DecisionResult(
        endpoint="fixture",
        revision="v1",
        answers={"lane": {"type": "noul", "noul": 0.8}},
        usage={},
        elapsed_ms=1,
    )
    with pytest.raises(RequestFailedError, match="not a choice"):
        choice_probabilities(result, "lane")
