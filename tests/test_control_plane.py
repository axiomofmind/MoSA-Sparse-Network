from __future__ import annotations

import json
import threading
from datetime import date
from pathlib import Path
from time import monotonic, sleep
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest
import yaml

import sparse_network.swap as swap_module
from sparse_network.api import build_server
from sparse_network.client import ControllerClient
from sparse_network.config import AppConfig
from sparse_network.fleet import ControllerEventLog, FleetManager
from sparse_network.models import ModelRegistry
from sparse_network.service import ControllerService
from sparse_network.swap import SwapCoordinator, SwapRequest


def _endpoint(endpoint_id: str, *, resident: bool = True) -> dict:
    return {
        "display_name": endpoint_id,
        "role": "resident" if resident else "exclusive_escalation_solver",
        "family": "qwen" if resident else "qwen3.8",
        "source": {"type": "builtin"},
        "runtime": {
            "adapter": "mock",
            "artifact_class": "complete_model",
            "exclusive_swap": not resident,
        },
        "modalities": ["text"],
        "capabilities": ["text_generation", "verification"],
        "context_size": 4096,
        "max_output_tokens": 512,
        "max_input_characters": 3584,
        "proposed_budget": {
            "process_ram_bytes": 0,
            "incremental_vram_bytes": 0,
            "kv_cache_bytes": 0,
            "request_workspace_bytes": 0,
            "vram_reclaim_tolerance_bytes": 0,
        },
        "admission": {"state": "admitted"},
        "license": {"reviewed_on": date(2026, 8, 23)},
    }


def _config(tmp_path: Path) -> AppConfig:
    configs = tmp_path / "configs"
    (configs / "rosters").mkdir(parents=True)
    (configs / "hardware").mkdir(parents=True)
    (configs / "models.yaml").write_text(
        yaml.safe_dump(
            {
                "schema": "sparse-network-model-registry.v1",
                "endpoints": {
                    "worker": _endpoint("worker"),
                    "qwen38-27b": _endpoint("qwen38-27b", resident=False),
                    "qwen38-27b-q4": _endpoint("qwen38-27b-q4", resident=False),
                },
            }
        ),
        encoding="utf-8",
    )
    (configs / "rosters" / "test.yaml").write_text(
        yaml.safe_dump(
            {
                "id": "test",
                "hardware_profile": "test",
                "resident": ["worker"],
                "exclusive_swap": ["qwen38-27b"],
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
    (tmp_path / "schemas").mkdir()
    (tmp_path / "schemas" / "openapi-v1.json").write_text(
        json.dumps({"openapi": "3.1.0", "info": {"title": "test", "version": "1"}}),
        encoding="utf-8",
    )
    (tmp_path / "dashboard").mkdir()
    (tmp_path / "dashboard" / "index.html").write_text(
        "<!doctype html><title>Fixture dashboard</title>", encoding="utf-8"
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
            "controller": {
                "request_timeout_seconds": 5,
                "telemetry_interval_seconds": 0.01,
            },
            "fleet": {
                "roster": str(configs / "rosters" / "test.yaml"),
                "queue_capacity": 4,
                "require_gpu_telemetry": False,
            },
        },
        sources=(),
    )


def _add_profile_switch_fixtures(config: AppConfig) -> None:
    models_path = config.model_registry_path
    models = yaml.safe_load(models_path.read_text(encoding="utf-8"))
    worker_two = _endpoint("worker-two")
    over_budget = _endpoint("over-budget")
    over_budget["proposed_budget"]["kv_cache_bytes"] = 2 * 1024**3
    missing = _endpoint("missing-model")
    missing["source"] = {
        "type": "huggingface",
        "repository": "fixture/missing",
        "revision": "0123456789abcdef",
        "files": {"model": "missing.gguf"},
    }
    licensed = _endpoint("licensed-worker")
    licensed["license"] = {"id": "fixture-license", "review_required": True}
    models["endpoints"].update(
        {
            "worker-two": worker_two,
            "over-budget": over_budget,
            "missing-model": missing,
            "licensed-worker": licensed,
        }
    )
    models_path.write_text(yaml.safe_dump(models), encoding="utf-8")
    roster_root = config.root / "configs" / "rosters"
    rosters = {
        "tier-16": ["worker"],
        "tier-24": ["worker-two"],
        "tier-32": ["worker", "worker-two"],
        "over-budget": ["over-budget"],
        "missing-artifacts": ["missing-model"],
        "broken-load": ["worker-two"],
        "license-review": ["licensed-worker"],
    }
    for profile_id, residents in rosters.items():
        (roster_root / f"{profile_id}.yaml").write_text(
            yaml.safe_dump(
                {
                    "id": profile_id,
                    "hardware_profile": "test",
                    "validation_state": "test_fixture",
                    "resident": residents,
                    "optional": [],
                    "exclusive_swap": (
                        ["qwen38-27b-q4"]
                        if profile_id in {"tier-16", "tier-24"}
                        else (["qwen38-27b"] if profile_id == "tier-32" else [])
                    ),
                }
            ),
            encoding="utf-8",
        )


def _wait(predicate: object, timeout: float = 3) -> None:
    deadline = monotonic() + timeout
    while monotonic() < deadline:
        if predicate():  # type: ignore[operator]
            return
        sleep(0.01)
    raise AssertionError("condition did not become true")


def test_event_log_monotonic_replay_and_retention(tmp_path: Path) -> None:
    log = ControllerEventLog(tmp_path / "events.jsonl", retention=100)
    for index in range(205):
        log.emit(event="telemetry", endpoint="worker", details={"sample": index})
    replay = log.read_after(200)
    assert [event["event_id"] for event in replay] == [201, 202, 203, 204, 205]
    assert len(log.read()) <= 105


def test_swap_restores_fleet_on_success_and_verification_failure(tmp_path: Path) -> None:
    config = _config(tmp_path)
    manager = FleetManager(config, ModelRegistry.load(config))
    manager.load_all()
    coordinator = SwapCoordinator(manager)
    success = coordinator.run(SwapRequest(prompt="sentinel", expected_contains="sentinel"))
    assert success.status == "answer"
    assert success.fleet_restored
    assert manager.status()["endpoints"]["worker"]["state"] == "ready"
    failure = coordinator.run(SwapRequest(prompt="sentinel", expected_contains="missing"))
    assert failure.status == "failed"
    assert failure.fleet_restored
    assert manager.status()["endpoints"]["worker"]["state"] == "ready"
    phases = [
        event["details"]["phase"]
        for event in manager.event_log.read()
        if event["event"] == "swap_phase"
    ]
    assert "loading_large" in phases
    assert phases[-1] == "failed"
    manager.shutdown()


def test_swap_restoration_requires_residents_but_not_optional_endpoints(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    roster_path = Path(str(config.data["fleet"]["roster"]))
    roster = yaml.safe_load(roster_path.read_text(encoding="utf-8"))
    roster["optional"] = ["qwen38-27b-q4"]
    roster_path.write_text(yaml.safe_dump(roster), encoding="utf-8")
    manager = FleetManager(config, ModelRegistry.load(config))
    manager.load_all()
    assert manager.residents_ready()
    assert manager.entries["qwen38-27b-q4"].state.value == "unavailable"
    result = SwapCoordinator(manager).run(
        SwapRequest(prompt="optional sentinel", expected_contains="optional sentinel")
    )
    assert result.status == "answer"
    assert result.fleet_restored
    assert manager.residents_ready()
    assert manager.entries["qwen38-27b-q4"].state.value == "unavailable"
    manager.shutdown()


def test_swap_coordinator_defaults_to_roster_declared_endpoint(tmp_path: Path) -> None:
    config = _config(tmp_path)
    roster_path = Path(str(config.data["fleet"]["roster"]))
    roster = yaml.safe_load(roster_path.read_text(encoding="utf-8"))
    roster["exclusive_swap"] = ["qwen38-27b-q4"]
    roster_path.write_text(yaml.safe_dump(roster), encoding="utf-8")
    manager = FleetManager(config, ModelRegistry.load(config))
    assert SwapCoordinator(manager).endpoint_id == "qwen38-27b-q4"
    manager.shutdown()


def test_service_rejects_swap_when_profile_has_no_exclusive_endpoint(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    roster_path = Path(str(config.data["fleet"]["roster"]))
    roster = yaml.safe_load(roster_path.read_text(encoding="utf-8"))
    roster["exclusive_swap"] = []
    roster_path.write_text(yaml.safe_dump(roster), encoding="utf-8")
    service = ControllerService(config, ModelRegistry.load(config))
    try:
        with pytest.raises(ValueError, match="does not provide a 27B escalation model"):
            service.run_swap({"prompt": "do not escalate"})
    finally:
        service.fleet.shutdown()


def test_api_auth_request_replay_and_state_reconstruction(tmp_path: Path) -> None:
    config = _config(tmp_path)
    registry = ModelRegistry.load(config)
    token = "0123456789abcdef"
    server = build_server(config, registry, port=0, token=token)
    server.service.start()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"
    client = ControllerClient(base_url, token)
    try:
        created = client.submit("worker", "api sentinel")
        request_id = created["id"]
        _wait(lambda: client.request_status(request_id)["state"] == "completed")
        state = client.state()
        assert state["fleet"]["endpoints"]["worker"]["state"] == "ready"
        assert any(item["id"] == request_id for item in state["requests"])
        _wait(lambda: "service_request_finished" in client.events(after=0))
        stream = client.events(after=0)
        assert "service_request_finished" in stream
        last_id = client.state()["last_event_id"]
        assert client.events(after=last_id) == ""
        answer_id = client.request_status(request_id)["result"]["envelope"][
            "answer_reference"
        ]
        content = client.request("GET", f"/v1/artifacts/{answer_id}/content")
        assert content["content"] == "Mock response: api sentinel"
        artifact = client.request("GET", f"/v1/artifacts/{answer_id}")
        assert "path" not in artifact
        assert artifact["content_available"] is True
        dashboard = client.request("GET", "/v1/dashboard")
        assert dashboard["session"]["role"] == "administrator"
        assert dashboard["session"]["read_only"] is False
        assert "profile:apply" in dashboard["session"]["permissions"]
        dashboard_request = next(
            item for item in dashboard["requests"] if item["id"] == request_id
        )
        assert dashboard_request["prompt"] == "[redacted]"
        assert dashboard_request["result"]["answer"] == "[redacted]"
        assert dashboard["models"][0]["license"]["reviewed_on"] == "2026-08-23"
        assert str(tmp_path) not in json.dumps(dashboard)

        runs = config.paths["runs"]
        assert runs is not None
        runs.mkdir(parents=True, exist_ok=True)
        (runs / "redaction-check.json").write_text(
            json.dumps(
                {
                    "schema": "test-run.v1",
                    "prompt": "private run prompt",
                    "report_path": str(tmp_path / "private.json"),
                }
            ),
            encoding="utf-8",
        )
        run = client.request("GET", "/v1/runs/redaction-check.json")
        assert run["prompt"] == "[redacted]"
        assert run["report_path"] == "[redacted]"

        with urlopen(base_url + "/dashboard/") as response:  # noqa: S310
            assert b"Fixture dashboard" in response.read()
            assert response.headers["Content-Security-Policy"].startswith("default-src")

        unauthenticated = Request(
            base_url + "/v1/fleet/operations/unload_all",
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(HTTPError) as error:
            urlopen(unauthenticated)  # noqa: S310
        assert error.value.code == 401

        with pytest.raises(HTTPError) as event_error:
            urlopen(base_url + "/v1/events")  # noqa: S310
        assert event_error.value.code == 401
    finally:
        server.shutdown()
        server.service.fleet.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_service_config_allowlist_and_antidoom_status(tmp_path: Path) -> None:
    config = _config(tmp_path)
    service = ControllerService(config, ModelRegistry.load(config))
    result = service.update_config(
        {"fleet.queue_capacity": 7}, confirmation="APPLY CONFIG"
    )
    assert result["changes"]["fleet.queue_capacity"] == 7
    assert service.fleet.queue_capacity == 7
    assert service.antidoom_status()["runtime_detector"]["maximum_retries"] == 1
    tool_event = service.record_tool_event(
        request_id="request-tool", tool="fixture", state="proposed"
    )
    assert tool_event["event"] == "tool_event"
    with pytest.raises(ValueError, match="not mutable"):
        service.update_config(
            {"paths.model_cache": "elsewhere"}, confirmation="APPLY CONFIG"
        )
    service.fleet.shutdown()


def test_coordination_evaluation_is_derived_from_saved_workflow_traces(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    runs = config.paths["runs"]
    assert runs is not None
    trace_root = runs / "execution-graphs"
    trace_root.mkdir(parents=True)
    (trace_root / "workflow.json").write_text(
        json.dumps(
            {
                "schema": "sparse-network-workflow-trace.v1",
                "status": "accepted",
                "mode": "mosa",
                "trigger": "deterministic_failure",
                "stages": [
                    {
                        "stage": "draft",
                        "endpoint": "worker",
                        "verification": {"accepted": True},
                        "context": {"stages": []},
                        "result": {"envelope": {"resource_usage": {"elapsed_ms": 5}}},
                    }
                ],
                "comparison": {"early_stopped": True},
            }
        ),
        encoding="utf-8",
    )
    (trace_root / "top1.json").write_text(
        json.dumps({"schema": "sparse-network-execution-trace.v1"}),
        encoding="utf-8",
    )
    service = ControllerService(config, ModelRegistry.load(config))
    result = service.create_evaluation({"suite": "coordination"}, actor="evaluator")
    assert result["baselines"][0]["metrics"]["run_count"] == 1.0
    assert result["coordination"]["metrics"]["first_pass_acceptance_rate"] == 1.0
    assert (runs / f"{result['id']}.json").exists()
    service.fleet.shutdown()


def test_profile_switches_mock_16_24_32_and_survives_restart(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _add_profile_switch_fixtures(config)
    registry = ModelRegistry.load(config)
    service = ControllerService(config, registry)
    service.start()
    try:
        for profile_id, residents, verifier in (
            ("tier-16", {"worker"}, "qwen38-27b-q4"),
            ("tier-24", {"worker-two"}, "qwen38-27b-q4"),
            ("tier-32", {"worker", "worker-two"}, "qwen38-27b"),
        ):
            plan = service.plan_profile_switch({"profile_id": profile_id}, actor="operator")
            assert plan["valid"]
            assert plan["confirmation_phrase"] == f"APPLY {profile_id}"
            with pytest.raises(ValueError, match="confirmation phrase"):
                service.apply_profile_switch(
                    {"plan_id": plan["id"], "confirmation": "wrong"},
                    actor="operator",
                )
            result = service.apply_profile_switch(
                {
                    "plan_id": plan["id"],
                    "confirmation": plan["confirmation_phrase"],
                },
                actor="operator",
            )
            assert result["status"] == "completed"
            assert set(service.fleet.entries) == residents
            assert service.swap.endpoint_id == verifier
            assert service.fleet.status()["active_escalation_endpoint"] == verifier
            assert all(value == "answer" for value in result["smoke_results"].values())
        service.fleet.shutdown()
        restarted = ControllerService(config, registry)
        try:
            assert restarted.fleet.roster_id == "tier-32"
            assert restarted.swap.endpoint_id == "qwen38-27b"
            restarted.start()
            assert all(
                value["state"] == "ready"
                for value in restarted.fleet.status()["endpoints"].values()
            )
        finally:
            restarted.fleet.shutdown()
    finally:
        if any(entry.runtime is not None for entry in service.fleet.entries.values()):
            service.fleet.shutdown()


def test_profile_plan_rejects_budget_and_missing_artifacts(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _add_profile_switch_fixtures(config)
    service = ControllerService(config, ModelRegistry.load(config))
    over_budget = service.plan_profile_switch({"profile_id": "over-budget"})
    assert not over_budget["valid"]
    assert over_budget["validation"]["resource_errors"]
    missing = service.plan_profile_switch({"profile_id": "missing-artifacts"})
    assert not missing["valid"]
    assert missing["validation"]["invalid_artifacts"] == ["missing-model"]
    service.fleet.shutdown()


def test_profile_switch_requires_license_ack_and_validates_custom_schema(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    _add_profile_switch_fixtures(config)
    service = ControllerService(config, ModelRegistry.load(config))
    service.start()
    try:
        licensed = service.plan_profile_switch({"profile_id": "license-review"})
        assert licensed["valid"]
        assert not licensed["ready_to_apply"]
        with pytest.raises(ValueError, match="license acknowledgement"):
            service.apply_profile_switch(
                {
                    "plan_id": licensed["id"],
                    "confirmation": licensed["confirmation_phrase"],
                }
            )

        custom = service.plan_profile_switch(
            {
                "custom_profile": {
                    "schema": "sparse-network-custom-roster.v1",
                    "id": "custom-worker",
                    "hardware_profile": "test",
                    "validation_state": "experimental",
                    "resident": ["worker"],
                    "optional": [],
                    "exclusive_swap": [],
                    "cpu": [],
                }
            }
        )
        assert custom["valid"]
        result = service.apply_profile_switch(
            {
                "plan_id": custom["id"],
                "confirmation": custom["confirmation_phrase"],
            }
        )
        assert result["profile_id"] == "custom-worker"
        with pytest.raises(ValueError, match="schema"):
            service.plan_profile_switch(
                {"custom_profile": {"schema": "unversioned", "id": "custom-bad"}}
            )
    finally:
        service.fleet.shutdown()


def test_profile_partial_load_failure_rolls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    _add_profile_switch_fixtures(config)
    service = ControllerService(config, ModelRegistry.load(config))
    service.start()
    plan = service.plan_profile_switch({"profile_id": "broken-load"})
    original_load = FleetManager.load_all

    def fail_selected(manager: FleetManager) -> dict:
        if manager.roster_id == "broken-load":
            raise RuntimeError("simulated partial load failure")
        return original_load(manager)

    monkeypatch.setattr(FleetManager, "load_all", fail_selected)
    try:
        with pytest.raises(RuntimeError, match="simulated partial load failure"):
            service.apply_profile_switch(
                {
                    "plan_id": plan["id"],
                    "confirmation": plan["confirmation_phrase"],
                }
            )
        assert service.fleet.roster_id == "test"
        assert service.fleet.status()["endpoints"]["worker"]["state"] == "ready"
        assert any(
            event["event"] == "profile_switch_rolled_back"
            for event in service.fleet.event_log.read()
        )
    finally:
        service.fleet.shutdown()


def test_api_roles_enforce_server_permissions(tmp_path: Path) -> None:
    config = _config(tmp_path)
    server = build_server(
        config,
        ModelRegistry.load(config),
        port=0,
        token="administrator-token-123",
        role_tokens={
            "viewer": "viewer-token-123456",
            "operator": "operator-token-1234",
            "evaluator": "evaluator-token-123",
        },
    )
    server.service.start()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"
    viewer = ControllerClient(base_url, "viewer-token-123456")
    evaluator = ControllerClient(base_url, "evaluator-token-123")
    try:
        snapshot = viewer.request("GET", "/v1/dashboard")
        assert snapshot["session"] == {
            "role": "viewer",
            "permissions": ["dashboard:read"],
            "read_only": True,
        }
        with pytest.raises(RuntimeError, match="403"):
            viewer.operation("smoke", "worker")
        evaluation = evaluator.request(
            "POST",
            "/v1/evaluations",
            {
                "suite": "roles",
                "baselines": [{"id": "top-1", "metrics": {"acceptance": 90}}],
            },
        )
        assert evaluation["actor"] == "evaluator"
        with pytest.raises(RuntimeError, match="403"):
            evaluator.request(
                "PATCH",
                "/v1/config",
                {
                    "changes": {"fleet.queue_capacity": 5},
                    "confirmation": "APPLY CONFIG",
                },
            )
    finally:
        server.shutdown()
        server.service.fleet.shutdown()
        server.server_close()
        thread.join(timeout=2)
@pytest.mark.parametrize(
    "phase",
    [
        "preserving",
        "draining",
        "reclaiming",
        "loading_large",
        "solving",
        "verifying",
        "unloading_large",
        "restoring",
    ],
)
def test_restart_recovery_restores_every_transitional_swap_phase(
    tmp_path: Path, phase: str
) -> None:
    config = _config(tmp_path)
    manager = FleetManager(config, ModelRegistry.load(config))
    manager.event_log.emit(
        event="swap_phase",
        endpoint="qwen38-27b",
        execution_id="swap-interrupted",
        details={"phase": phase},
    )
    coordinator = SwapCoordinator(manager)
    assert coordinator.recover_if_interrupted()
    assert manager.status()["endpoints"]["worker"]["state"] == "ready"
    assert not coordinator.recover_if_interrupted()
    manager.shutdown()


def test_swap_transient_drain_failure_preserves_input_and_residents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    manager = FleetManager(config, ModelRegistry.load(config))
    manager.load_all()
    original = manager.unload_all
    attempts = 0

    def fail_once(*, timeout_seconds: float = 120) -> dict[str, object]:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("resident refused to drain")
        return original(timeout_seconds=timeout_seconds)

    monkeypatch.setattr(manager, "unload_all", fail_once)
    result = SwapCoordinator(manager).run(SwapRequest(prompt="preserve this"))
    assert result.status == "failed"
    assert result.fleet_restored
    assert manager.controller.artifacts.read_text(result.preserved_artifact) == "preserve this"
    manager.shutdown()


def test_swap_large_load_failure_restores_residents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    manager = FleetManager(config, ModelRegistry.load(config))
    manager.load_all()
    original = manager.controller.create_runtime

    def fail_large(endpoint: object) -> object:
        if getattr(endpoint, "id", None) == "qwen38-27b":
            raise RuntimeError("large load failed")
        return original(endpoint)  # type: ignore[arg-type]

    monkeypatch.setattr(manager.controller, "create_runtime", fail_large)
    result = SwapCoordinator(manager).run(SwapRequest(prompt="load failure"))
    assert result.status == "failed"
    assert result.fleet_restored
    manager.shutdown()


def test_swap_reclaim_and_ram_failures_are_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    manager = FleetManager(config, ModelRegistry.load(config))
    manager.load_all()
    monkeypatch.setattr(swap_module, "gpu_memory_used_bytes", lambda _index: 1)
    reclaim = SwapCoordinator(manager).run(SwapRequest(prompt="reclaim failure"))
    assert reclaim.status == "failed" and reclaim.fleet_restored
    monkeypatch.setattr(swap_module, "gpu_memory_used_bytes", lambda _index: 0)
    endpoint = manager.registry.get("qwen38-27b")
    endpoint.proposed_budget["process_ram_bytes"] = 1
    monkeypatch.setattr(
        swap_module.psutil, "virtual_memory", lambda: SimpleNamespace(available=0)
    )
    ram = SwapCoordinator(manager).run(SwapRequest(prompt="ram failure"))
    assert ram.status == "failed" and ram.fleet_restored
    manager.shutdown()


def test_swap_cancelled_generation_restores_residents(tmp_path: Path) -> None:
    config = _config(tmp_path)
    manager = FleetManager(config, ModelRegistry.load(config))
    manager.load_all()
    cancellation = threading.Event()
    cancellation.set()
    result = SwapCoordinator(manager).run(
        SwapRequest(prompt="cancelled"), cancellation=cancellation
    )
    assert result.status == "failed"
    assert result.fleet_restored
    manager.shutdown()


def test_swap_cancellation_at_large_generation_restores_residents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    manager = FleetManager(config, ModelRegistry.load(config))
    manager.load_all()
    original_smoke = manager.controller.smoke

    def cancel_large(**kwargs: object) -> object:
        if kwargs.get("endpoint_id") == "qwen38-27b":
            cancellation = kwargs.get("cancellation")
            assert isinstance(cancellation, threading.Event)
            cancellation.set()
        return original_smoke(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(manager.controller, "smoke", cancel_large)
    result = SwapCoordinator(manager).run(SwapRequest(prompt="cancel at generation"))
    assert result.status == "failed"
    assert result.fleet_restored
    manager.shutdown()


def test_swap_generation_timeout_and_restoration_failure_are_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    manager = FleetManager(config, ModelRegistry.load(config))
    manager.load_all()
    timeout = SwapCoordinator(manager).run(
        SwapRequest(prompt="[mock:slow]", timeout_seconds=0.01)
    )
    assert timeout.status == "failed"
    assert timeout.fleet_restored

    coordinator = SwapCoordinator(manager)
    monkeypatch.setattr(
        coordinator,
        "_restore_residents",
        lambda: (_ for _ in ()).throw(RuntimeError("partial restoration failure")),
    )
    failed_restore = coordinator.run(SwapRequest(prompt="restore failure"))
    assert failed_restore.status == "failed"
    assert not failed_restore.fleet_restored
    assert "restoration failure" in (failed_restore.error or "")
    monkeypatch.undo()
    manager.load_all()
    manager.resume_admission(reason="test cleanup")
    manager.shutdown()
