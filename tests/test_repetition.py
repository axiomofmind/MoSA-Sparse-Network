from __future__ import annotations

import json
import threading
from pathlib import Path

from conftest import make_test_config

from sparse_network.controller import Controller
from sparse_network.models import ModelRegistry
from sparse_network.repetition import (
    RepetitionDetectedError,
    RepetitionPolicy,
    find_inner_repetition,
)
from sparse_network.runtimes import LlamaCppAdapter, RuntimeState


def _mock_endpoint() -> dict:
    return {
        "id": "mock-loop",
        "definition": {
            "display_name": "Mock loop",
            "role": "fixture",
            "source": {"type": "builtin"},
            "runtime": {"adapter": "mock"},
            "modalities": ["text"],
            "capabilities": ["text_generation"],
            "context_size": 4096,
            "max_output_tokens": 512,
            "max_input_characters": 3584,
            "admission": {"state": "admitted"},
        },
    }


def test_antidoom_detector_finds_inner_loop_and_ignores_normal_text() -> None:
    normal = "A short response with no exact repeated span."
    assert find_inner_repetition(normal) == (False, None)
    text = "Useful prefix. " + "Wait, reconsider. " * 20
    found, hit = find_inner_repetition(text)
    assert found and hit is not None
    assert hit.period == len("Wait, reconsider. ")
    assert hit.repeats >= 4
    assert text[: hit.start].startswith("Useful prefix. ")


def test_controller_rejects_loop_and_preserves_prefix_as_diagnostic(tmp_path: Path) -> None:
    config = make_test_config(tmp_path, endpoint=_mock_endpoint())
    config.data["repetition"] = {
        "min_repeats": 4,
        "min_total_repeated": 60,
        "maximum_retries": 1,
    }
    registry = ModelRegistry.load(config)
    controller = Controller(config, registry)
    result = controller.smoke(endpoint_id="mock-loop", prompt="[mock:loop]")
    assert result.envelope.status == "needs_escalation"
    assert result.answer is None
    assert result.envelope.answer_reference is None
    assert result.envelope.repetition["detected"]
    diagnostic_id = str(result.envelope.repetition["diagnostic_prefix_reference"])
    assert controller.artifacts.read_text(diagnostic_id).startswith("Useful prefix. ")
    record = controller.artifacts.get(diagnostic_id)
    assert record.kind == "repetition_diagnostic"
    assert RepetitionPolicy.from_mapping(config.data["repetition"]).maximum_retries == 1


def test_llama_stream_is_cancelled_as_soon_as_exact_loop_is_detected(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    config = make_test_config(tmp_path, endpoint=_mock_endpoint())
    endpoint = ModelRegistry.load(config).get("mock-loop")
    repeated = "Useful prefix. " + "Wait, reconsider. " * 20
    payload = json.dumps(
        {"choices": [{"delta": {"content": repeated}, "finish_reason": None}]}
    ).encode()

    class Response:
        status = 200

        def __init__(self) -> None:
            self.lines = iter([b"data: " + payload + b"\n", b"data: [DONE]\n"])

        def readline(self) -> bytes:
            return next(self.lines, b"")

    class Connection:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            self.closed = False

        def request(self, *_args: object, **_kwargs: object) -> None:
            return None

        def getresponse(self) -> Response:
            return Response()

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(  # type: ignore[attr-defined]
        "sparse_network.runtimes.llama_cpp.http.client.HTTPConnection",
        Connection,
    )
    adapter = LlamaCppAdapter(
        endpoint,
        executable="llama-server",
        model_path=tmp_path / "unused.gguf",
        projector_path=None,
        log_path=tmp_path / "llama.log",
        host="127.0.0.1",
        startup_timeout_seconds=1,
        shutdown_grace_seconds=1,
        repetition_policy=RepetitionPolicy(streaming_check_interval_characters=32),
    )
    adapter.state = RuntimeState.READY
    try:
        adapter.generate(
            prompt="test",
            images=(),
            cancellation=threading.Event(),
            timeout_seconds=1,
        )
    except RepetitionDetectedError as exc:
        assert exc.streaming
        assert exc.diagnostic_prefix.startswith("Useful prefix. ")
    else:
        raise AssertionError("streaming repetition was not detected")
