from __future__ import annotations

import sys
import threading
from pathlib import Path

from sparse_network.models import EndpointDefinition, ModelSource
from sparse_network.runtimes.transformers import TransformersAdapter


def endpoint() -> EndpointDefinition:
    return EndpointDefinition(
        id="transformers-fixture",
        display_name="Transformers fixture",
        role="fixture",
        family="fixture",
        source=ModelSource(type="builtin"),
        runtime={"adapter": "transformers", "device": "cpu"},
        modalities=("text",),
        capabilities=("text_generation",),
        context_size=128,
        max_output_tokens=16,
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
    print(json.dumps({'event': 'result', 'text': 'worker response', 'input_tokens': 2,
                      'output_tokens': 2, 'finish_reason': 'stop'}), flush=True)
""",
        encoding="utf-8",
    )


def test_transformers_worker_lifecycle(tmp_path: Path) -> None:
    worker = tmp_path / "worker.py"
    fake_worker(worker)
    adapter = TransformersAdapter(
        endpoint(),
        executable=sys.executable,
        model_path=tmp_path,
        worker_path=worker,
        log_path=tmp_path / "worker.log",
        startup_timeout_seconds=3,
        shutdown_grace_seconds=1,
    )
    adapter.start()
    result = adapter.generate(
        prompt="hello",
        images=(),
        cancellation=threading.Event(),
        timeout_seconds=3,
    )
    assert result.text == "worker response"
    assert adapter.pid is not None
    adapter.stop()
    assert adapter.pid is None
    assert adapter.state.value == "stopped"
