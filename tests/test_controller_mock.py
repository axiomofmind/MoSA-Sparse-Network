from __future__ import annotations

import threading
from pathlib import Path
from time import sleep

import pytest
from conftest import make_test_config
from PIL import Image

from sparse_network.controller import Controller
from sparse_network.models import ModelRegistry


def mock_endpoint() -> dict:
    return {
        "id": "mock-echo",
        "definition": {
            "display_name": "Mock Echo",
            "role": "fixture",
            "source": {"type": "builtin"},
            "runtime": {"adapter": "mock"},
            "modalities": ["text"],
            "capabilities": ["text_generation"],
            "context_size": 128,
            "max_output_tokens": 32,
            "max_input_characters": 64,
            "admission": {"state": "admitted"},
        },
    }


def test_mock_smoke_persists_answer_and_stops(tmp_path: Path) -> None:
    config = make_test_config(tmp_path, endpoint=mock_endpoint())
    controller = Controller(config, ModelRegistry.load(config))
    result = controller.smoke(endpoint_id="mock-echo", prompt="hello")
    assert result.envelope.status == "answer"
    assert result.answer == "Mock response: hello"
    assert result.envelope.answer_reference is not None
    assert result.lifecycle["runtime_state"] == "stopped"
    manifests = list((tmp_path / "artifacts").rglob("*.json"))
    answers = list((tmp_path / "artifacts").rglob("*.txt"))
    assert len(manifests) == 2
    assert len(answers) == 2


def test_mock_failure_returns_failed_envelope(tmp_path: Path) -> None:
    config = make_test_config(tmp_path, endpoint=mock_endpoint())
    controller = Controller(config, ModelRegistry.load(config))
    result = controller.smoke(endpoint_id="mock-echo", prompt="[mock:failure]")
    assert result.envelope.status == "failed"
    assert result.envelope.answer_reference is None
    assert "Simulated" in result.envelope.verification["error"]


def test_oversized_prompt_is_rejected_before_runtime_start(tmp_path: Path) -> None:
    config = make_test_config(tmp_path, endpoint=mock_endpoint())
    controller = Controller(config, ModelRegistry.load(config))
    try:
        controller.smoke(endpoint_id="mock-echo", prompt="x" * 65)
    except ValueError as exc:
        assert "endpoint limit" in str(exc)
    else:
        raise AssertionError("oversized prompt was accepted")


def test_runtime_token_count_replaces_utf8_byte_estimate(tmp_path: Path) -> None:
    config = make_test_config(tmp_path, endpoint=mock_endpoint())
    controller = Controller(config, ModelRegistry.load(config))
    result = controller.smoke(endpoint_id="mock-echo", prompt="🙂" * 30)
    assert result.envelope.status == "answer"
    assert result.envelope.resource_usage.input_tokens == 1


def test_runtime_token_count_rejects_prompt_over_context_budget(tmp_path: Path) -> None:
    endpoint = mock_endpoint()
    endpoint["definition"]["max_input_characters"] = 1000
    config = make_test_config(tmp_path, endpoint=endpoint)
    controller = Controller(config, ModelRegistry.load(config))
    with pytest.raises(ValueError, match="needs 97 tokens.*budget is 96"):
        controller.smoke(endpoint_id="mock-echo", prompt=" ".join(["x"] * 97))


def test_mock_request_can_be_cancelled(tmp_path: Path) -> None:
    config = make_test_config(tmp_path, endpoint=mock_endpoint())
    controller = Controller(config, ModelRegistry.load(config))
    cancellation = threading.Event()
    output: list[object] = []

    def run() -> None:
        output.append(
            controller.smoke(
                endpoint_id="mock-echo",
                prompt="[mock:slow]",
                cancellation=cancellation,
            )
        )

    thread = threading.Thread(target=run)
    thread.start()
    sleep(0.1)
    cancellation.set()
    thread.join(timeout=3)
    assert not thread.is_alive()
    result = output[0]
    assert result.envelope.status == "cancelled"  # type: ignore[attr-defined]
    assert result.lifecycle["runtime_state"] == "stopped"  # type: ignore[attr-defined]


def test_mock_vision_preserves_original_image(tmp_path: Path) -> None:
    endpoint = mock_endpoint()
    endpoint["definition"]["modalities"] = ["text", "image"]
    config = make_test_config(tmp_path, endpoint=endpoint)
    image_path = tmp_path / "fixture.png"
    Image.new("RGB", (8, 8), "blue").save(image_path)
    result = Controller(config, ModelRegistry.load(config)).smoke(
        endpoint_id="mock-echo",
        prompt="describe",
        images=(image_path,),
    )
    assert result.envelope.modalities_consumed == ("text", "image")
    assert len(result.envelope.evidence_references) == 1
    assert len(list((tmp_path / "artifacts").rglob("*.png"))) == 1
    assert len(list((tmp_path / "artifacts").rglob("*.txt"))) == 2


def test_text_endpoint_rejects_images_before_start(tmp_path: Path) -> None:
    config = make_test_config(tmp_path, endpoint=mock_endpoint())
    image_path = tmp_path / "fixture.png"
    Image.new("RGB", (8, 8), "blue").save(image_path)
    with pytest.raises(ValueError, match="does not accept image"):
        Controller(config, ModelRegistry.load(config)).smoke(
            endpoint_id="mock-echo",
            prompt="describe",
            images=(image_path,),
        )


def test_malformed_image_is_rejected_before_start(tmp_path: Path) -> None:
    endpoint = mock_endpoint()
    endpoint["definition"]["modalities"] = ["text", "image"]
    config = make_test_config(tmp_path, endpoint=endpoint)
    image_path = tmp_path / "broken.png"
    image_path.write_bytes(b"this is not a PNG")
    with pytest.raises(ValueError, match="content does not match"):
        Controller(config, ModelRegistry.load(config)).smoke(
            endpoint_id="mock-echo",
            prompt="describe",
            images=(image_path,),
        )
