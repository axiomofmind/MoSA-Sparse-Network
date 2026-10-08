from __future__ import annotations

import os
import threading
from time import sleep

import pytest

from sparse_network.config import load_config
from sparse_network.controller import Controller
from sparse_network.models import ModelRegistry


@pytest.mark.skipif(
    os.environ.get("SPARSE_RUN_REAL_MODEL_TEST") != "1",
    reason="Set SPARSE_RUN_REAL_MODEL_TEST=1 to run the real local Qwen lifecycle",
)
def test_real_qwen_smoke() -> None:
    config = load_config()
    result = Controller(config, ModelRegistry.load(config)).smoke(
        endpoint_id="qwen35-4b",
        prompt="Reply with exactly: ready",
    )
    assert result.envelope.status == "answer"
    assert result.answer
    assert result.lifecycle["runtime_state"] == "stopped"
    assert result.lifecycle["vram_reclaimed_within_tolerance"]


@pytest.mark.skipif(
    os.environ.get("SPARSE_RUN_REAL_MODEL_TEST") != "1",
    reason="Set SPARSE_RUN_REAL_MODEL_TEST=1 to run the real local Qwen lifecycle",
)
def test_real_qwen_cancellation_releases_runtime() -> None:
    config = load_config()
    controller = Controller(config, ModelRegistry.load(config))
    cancellation = threading.Event()
    completed: list[object] = []

    def execute() -> None:
        completed.append(
            controller.smoke(
                endpoint_id="qwen35-4b",
                prompt="Write a very long numbered explanation of local model orchestration.",
                cancellation=cancellation,
            )
        )

    thread = threading.Thread(target=execute)
    thread.start()
    sleep(3)
    cancellation.set()
    thread.join(timeout=60)

    assert not thread.is_alive()
    assert len(completed) == 1
    result = completed[0]
    assert result.envelope.status == "cancelled"  # type: ignore[attr-defined]
    assert result.lifecycle["runtime_state"] == "stopped"  # type: ignore[attr-defined]
    assert result.lifecycle["vram_reclaimed_within_tolerance"]  # type: ignore[attr-defined]
