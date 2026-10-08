from __future__ import annotations

import threading
from pathlib import Path
from time import monotonic, sleep

import pytest
import yaml

from sparse_network.config import AppConfig
from sparse_network.errors import QueueCapacityError, ResourceAdmissionError
from sparse_network.fleet import FleetManager, FleetState, validate_transition
from sparse_network.models import ModelRegistry


def _endpoint(endpoint_id: str, *, proposed_vram: int = 0) -> dict:
    return {
        "display_name": endpoint_id,
        "role": "fixture",
        "source": {"type": "builtin"},
        "runtime": {"adapter": "mock"},
        "modalities": ["text"],
        "capabilities": ["text_generation"],
        "context_size": 4096,
        "max_output_tokens": 512,
        "max_input_characters": 3584,
        "proposed_budget": {
            "incremental_vram_bytes": proposed_vram,
            "kv_cache_bytes": 0,
            "request_workspace_bytes": 0,
        },
        "admission": {"state": "admitted"},
    }


def _config(
    tmp_path: Path,
    *,
    queue_capacity: int = 4,
    proposed_vram: int = 0,
) -> AppConfig:
    configs = tmp_path / "configs"
    (configs / "rosters").mkdir(parents=True)
    (configs / "hardware").mkdir(parents=True)
    (configs / "models.yaml").write_text(
        yaml.safe_dump(
            {
                "schema": "sparse-network-model-registry.v1",
                "endpoints": {
                    "worker-a": _endpoint("worker-a", proposed_vram=proposed_vram),
                    "worker-b": _endpoint("worker-b"),
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (configs / "rosters" / "test.yaml").write_text(
        yaml.safe_dump(
            {
                "id": "test-roster",
                "hardware_profile": "test",
                "resident": ["worker-a", "worker-b"],
            }
        ),
        encoding="utf-8",
    )
    (configs / "hardware" / "test.yaml").write_text(
        yaml.safe_dump(
            {
                "scheduler": {"initial_parallel_generations": 1},
                "budgets": {"minimum_vram_reserve_gb": 0, "total_kv_cache_gb": 1},
            }
        ),
        encoding="utf-8",
    )
    return AppConfig(
        root=tmp_path,
        data={
            "paths": {
                "model_cache": str(tmp_path / "cache"),
                "artifacts": str(tmp_path / "artifacts"),
                "logs": str(tmp_path / "logs"),
                "runs": str(tmp_path / "runs"),
                "indexes": str(tmp_path / "indexes"),
            },
            "runtime_executables": {"llama_cpp": "llama-server"},
            "model_registry": str(configs / "models.yaml"),
            "endpoint_overrides": {},
            "controller": {"request_timeout_seconds": 5, "telemetry_interval_seconds": 0.05},
            "fleet": {
                "roster": str(configs / "rosters" / "test.yaml"),
                "queue_capacity": queue_capacity,
                "require_gpu_telemetry": True,
            },
        },
        sources=(),
    )


def _wait_for(predicate: object, *, timeout: float = 3) -> None:
    deadline = monotonic() + timeout
    while monotonic() < deadline:
        if predicate():  # type: ignore[operator]
            return
        sleep(0.01)
    raise AssertionError("condition did not become true")


def test_fleet_load_submit_and_shutdown(tmp_path: Path) -> None:
    config = _config(tmp_path)
    manager = FleetManager(config, ModelRegistry.load(config))
    loaded = manager.load_all()
    assert set(item["state"] for item in loaded["endpoints"].values()) == {"ready"}
    result = manager.submit(endpoint_id="worker-a", prompt="resident")
    assert result.answer == "Mock response: resident"
    assert result.lifecycle["cold_load_ms"] == 0
    assert result.lifecycle["runtime_resident"]
    assert manager.select_endpoint(capability="text_generation") == "worker-a"
    stopped = manager.shutdown()
    assert set(item["state"] for item in stopped["endpoints"].values()) == {"unavailable"}


def test_load_endpoints_prepares_only_requested_residents(tmp_path: Path) -> None:
    config = _config(tmp_path)
    manager = FleetManager(config, ModelRegistry.load(config))
    try:
        manager.load_endpoints(["worker-a"])
        assert manager.entries["worker-a"].state == FleetState.READY
        assert manager.entries["worker-b"].state == FleetState.UNAVAILABLE
        manager.load_all()
        assert manager.entries["worker-a"].state == FleetState.READY
        assert manager.entries["worker-b"].state == FleetState.READY
    finally:
        manager.shutdown()


def test_optional_endpoint_is_registered_but_loaded_only_on_demand(tmp_path: Path) -> None:
    config = _config(tmp_path)
    roster_path = tmp_path / "configs" / "rosters" / "test.yaml"
    roster = yaml.safe_load(roster_path.read_text(encoding="utf-8"))
    roster["resident"] = ["worker-a"]
    roster["optional"] = ["worker-b"]
    roster_path.write_text(yaml.safe_dump(roster), encoding="utf-8")

    manager = FleetManager(config, ModelRegistry.load(config))
    try:
        assert set(manager.entries) == {"worker-a", "worker-b"}
        manager.load_all()
        assert manager.entries["worker-a"].state == FleetState.READY
        assert manager.entries["worker-b"].state == FleetState.UNAVAILABLE

        manager.load_endpoints(["worker-b"])
        assert manager.entries["worker-b"].state == FleetState.READY
    finally:
        manager.shutdown()


def test_bounded_per_endpoint_queue_rejects_overflow(tmp_path: Path) -> None:
    config = _config(tmp_path, queue_capacity=1)
    manager = FleetManager(config, ModelRegistry.load(config))
    manager.load_all()
    cancellation = threading.Event()
    first: list[object] = []
    second: list[object] = []

    thread_one = threading.Thread(
        target=lambda: first.append(
            manager.submit(
                endpoint_id="worker-a",
                prompt="[mock:slow]",
                cancellation=cancellation,
                timeout_seconds=10,
            )
        )
    )
    thread_one.start()
    _wait_for(lambda: manager.status()["active_requests"] == 1)
    thread_two = threading.Thread(
        target=lambda: second.append(manager.submit(endpoint_id="worker-a", prompt="queued"))
    )
    thread_two.start()
    _wait_for(lambda: manager.status()["queue_depth"] == 1)
    with pytest.raises(QueueCapacityError):
        manager.submit(endpoint_id="worker-b", prompt="overflow")
    cancellation.set()
    thread_one.join(timeout=3)
    thread_two.join(timeout=3)
    assert not thread_one.is_alive() and not thread_two.is_alive()
    assert first[0].envelope.status == "cancelled"  # type: ignore[attr-defined]
    assert second[0].answer == "Mock response: queued"  # type: ignore[attr-defined]
    manager.shutdown()


def test_two_workers_can_run_different_endpoints_without_state_corruption(tmp_path: Path) -> None:
    config = _config(tmp_path)
    manager = FleetManager(
        config,
        ModelRegistry.load(config),
        maximum_parallel_generations=2,
    )
    manager.load_all()
    cancellations = [threading.Event(), threading.Event()]
    results: list[object] = []
    threads = [
        threading.Thread(
            target=lambda index=index: results.append(
                manager.submit(
                    endpoint_id=f"worker-{'a' if index == 0 else 'b'}",
                    prompt="[mock:slow]",
                    cancellation=cancellations[index],
                    timeout_seconds=10,
                )
            )
        )
        for index in range(2)
    ]
    for thread in threads:
        thread.start()
    _wait_for(lambda: manager.status()["active_requests"] == 2)
    for cancellation in cancellations:
        cancellation.set()
    for thread in threads:
        thread.join(timeout=3)
    assert len(results) == 2
    assert all(result.envelope.status == "cancelled" for result in results)  # type: ignore[attr-defined]
    assert set(item["state"] for item in manager.status()["endpoints"].values()) == {"ready"}
    manager.shutdown()


def test_priority_prevents_short_queued_work_from_starving(tmp_path: Path) -> None:
    config = _config(tmp_path)
    manager = FleetManager(config, ModelRegistry.load(config))
    manager.load_all()
    cancellation = threading.Event()
    completion_order: list[str] = []

    active = threading.Thread(
        target=lambda: manager.submit(
            endpoint_id="worker-a",
            prompt="[mock:slow]",
            cancellation=cancellation,
            timeout_seconds=10,
        )
    )
    low = threading.Thread(
        target=lambda: (
            manager.submit(endpoint_id="worker-b", prompt="low", priority=20),
            completion_order.append("low"),
        )
    )
    high = threading.Thread(
        target=lambda: (
            manager.submit(endpoint_id="worker-b", prompt="high", priority=0),
            completion_order.append("high"),
        )
    )
    active.start()
    _wait_for(lambda: manager.status()["active_requests"] == 1)
    low.start()
    high.start()
    _wait_for(lambda: manager.status()["queue_depth"] == 2)
    cancellation.set()
    for thread in (active, low, high):
        thread.join(timeout=3)
    assert completion_order == ["high", "low"]
    manager.shutdown()


def test_transitional_crash_state_is_quarantined(tmp_path: Path) -> None:
    config = _config(tmp_path)
    first = FleetManager(config, ModelRegistry.load(config))
    first.event_log.emit(
        event="state_transition",
        endpoint="worker-a",
        details={"from": "unavailable", "to": "loading", "reason": "test crash"},
    )
    first.shutdown()
    recovered = FleetManager(config, ModelRegistry.load(config))
    assert recovered.entries["worker-a"].state == FleetState.QUARANTINED
    recovered.shutdown()


def test_planned_overcommit_is_rejected_before_load(tmp_path: Path) -> None:
    config = _config(tmp_path, proposed_vram=3 * 1024**3)
    manager = FleetManager(config, ModelRegistry.load(config))
    manager.total_vram_bytes = 2 * 1024**3
    with pytest.raises(ResourceAdmissionError, match="Planned resident VRAM"):
        manager.load_all()
    manager.shutdown()


def test_illegal_state_transition_is_rejected() -> None:
    with pytest.raises(Exception, match="Illegal fleet transition"):
        validate_transition(FleetState.UNAVAILABLE, FleetState.BUSY)


def test_resident_selector_rejects_missing_capability(tmp_path: Path) -> None:
    config = _config(tmp_path)
    manager = FleetManager(config, ModelRegistry.load(config))
    manager.load_all()
    with pytest.raises(Exception, match="No resident endpoint"):
        manager.select_endpoint(capability="visual_understanding", required_modalities=("image",))
    manager.shutdown()
