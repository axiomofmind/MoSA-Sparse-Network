from __future__ import annotations

import os
import threading
from pathlib import Path
from time import sleep

import pytest

from sparse_network.config import load_config
from sparse_network.controller import Controller
from sparse_network.models import ModelRegistry

pytestmark = pytest.mark.skipif(
    os.environ.get("SPARSE_RUN_MILESTONE4_TESTS") != "1",
    reason="Set SPARSE_RUN_MILESTONE4_TESTS=1 to run local Milestone 4 endpoints",
)


@pytest.mark.parametrize(
    ("endpoint_id", "image_name"),
    [
        ("lfm25-1.2b", None),
        ("nemotron-4b", None),
        ("gemma-e2b-vision", "screenshot.png"),
        ("qwen3-8b-fp8", None),
    ],
)
def test_cancel_then_recover(endpoint_id: str, image_name: str | None) -> None:
    config = load_config()
    registry = ModelRegistry.load(config)
    controller = Controller(config, registry)
    images: tuple[Path, ...] = ()
    if image_name is not None:
        image = config.root / ".sparse-data" / "fixtures" / "vision" / image_name
        if not image.exists():
            pytest.skip("Generate vision fixtures before the real admission tests")
        images = (image,)
    cancellation = threading.Event()
    ready = threading.Event()
    completed: list[object] = []
    failures: list[BaseException] = []

    def execute() -> None:
        try:
            completed.append(
                controller.smoke(
                    endpoint_id=endpoint_id,
                    prompt=(
                        "Write a detailed 100-part analysis and continue until the output limit."
                    ),
                    images=images,
                    cancellation=cancellation,
                    ready_event=ready,
                    timeout_seconds=180,
                )
            )
        except BaseException as exc:
            failures.append(exc)

    thread = threading.Thread(target=execute)
    thread.start()
    assert ready.wait(timeout=180)
    sleep(0.25)
    cancellation.set()
    thread.join(timeout=180)

    assert not thread.is_alive()
    assert not failures
    cancelled = completed[0]
    assert cancelled.envelope.status == "cancelled"  # type: ignore[attr-defined]
    assert cancelled.lifecycle["runtime_process_released"]  # type: ignore[attr-defined]

    recovered = Controller(config, registry).smoke(
        endpoint_id=endpoint_id,
        prompt="Reply with exactly: RECOVERED",
        timeout_seconds=180,
    )
    assert recovered.envelope.status == "answer"
    assert recovered.lifecycle["runtime_process_released"]
