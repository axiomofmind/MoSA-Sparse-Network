"""Exercise the admitted top-2 and MoSA workflows on the resident fleet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import monotonic, sleep
from typing import Any

from sparse_network.config import load_config
from sparse_network.execution import ExecutionRequest
from sparse_network.fleet import FleetManager
from sparse_network.models import ModelRegistry
from sparse_network.routing import RouteRequest, StaticRouter
from sparse_network.telemetry import gpu_memory_used_bytes
from sparse_network.workflows import WorkflowExecutor, WorkflowRequest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve(strict=False)
    config = load_config()
    registry = ModelRegistry.load(config)
    manager = FleetManager(config, registry, maximum_parallel_generations=2)
    router = StaticRouter(config, registry, resident_endpoint_ids=set(manager.entries))
    executor = WorkflowExecutor(manager, router)
    profile: dict[str, Any] = {"schema": "sparse-network-workflow-profile.v1"}
    fatal_error: str | None = None
    try:
        profile["loaded"] = manager.load_all()
        top2 = executor.run(
            WorkflowRequest(
                execution=ExecutionRequest(
                    route=RouteRequest(
                        prompt="@route:general Reply with exactly: TOP2_WORKFLOW_OK"
                    ),
                    expected_contains="TOP2_WORKFLOW_OK",
                    timeout_seconds=180,
                ),
                mode="top-2",
                trigger="disputed",
            )
        )
        early = executor.run(
            WorkflowRequest(
                execution=ExecutionRequest(
                    route=RouteRequest(
                        prompt="@route:general Reply with exactly: MOSA_EARLY_OK"
                    ),
                    expected_contains="MOSA_EARLY_OK",
                    timeout_seconds=180,
                ),
                mode="mosa",
                trigger="deterministic_failure",
            )
        )
        full = executor.run(
            WorkflowRequest(
                execution=ExecutionRequest(
                    route=RouteRequest(
                        prompt="@route:general Reply with exactly: MOSA_FINAL_OK"
                    ),
                    expected_contains="MOSA_FINAL_OK",
                    timeout_seconds=180,
                ),
                mode="mosa",
                trigger="disputed",
            )
        )
        profile["top2"] = top2.to_dict()
        profile["mosa_early"] = early.to_dict()
        profile["mosa_full"] = full.to_dict()
        profile["before_shutdown"] = manager.status()
    except BaseException as exc:
        fatal_error = f"{type(exc).__name__}: {exc}"
        profile["fatal_error"] = fatal_error
    finally:
        try:
            profile["shutdown"] = manager.shutdown()
        except BaseException as exc:
            profile["shutdown_error"] = f"{type(exc).__name__}: {exc}"
    telemetry_index = next(
        (
            entry.endpoint.telemetry_gpu_index
            for entry in manager.entries.values()
            if entry.endpoint.telemetry_gpu_index is not None
        ),
        None,
    )
    initial = int(profile.get("loaded", {}).get("initial_vram_bytes", 0))
    final = gpu_memory_used_bytes(telemetry_index)
    deadline = monotonic() + 15
    while final > initial + 512 * 1024**2 and monotonic() < deadline:
        sleep(0.25)
        final = gpu_memory_used_bytes(telemetry_index)
    profile["final_vram_bytes"] = final
    profile["peak_vram_bytes"] = manager.peak_vram_bytes
    profile["vram_reclaimed"] = final <= initial + 512 * 1024**2
    profile["reserve_held"] = (
        not manager.total_vram_bytes
        or manager.peak_vram_bytes + manager.minimum_vram_reserve_bytes
        <= manager.total_vram_bytes
    )
    profile["passed"] = (
        fatal_error is None
        and "shutdown_error" not in profile
        and profile.get("top2", {}).get("status") == "accepted"
        and len(profile.get("top2", {}).get("stages", [])) == 2
        and profile.get("top2", {}).get("comparison", {}).get("family_diverse", False)
        and profile.get("mosa_early", {}).get("status") == "accepted"
        and len(profile.get("mosa_early", {}).get("stages", [])) == 1
        and profile.get("mosa_early", {}).get("comparison", {}).get(
            "early_stopped", False
        )
        and profile.get("mosa_full", {}).get("status") == "accepted"
        and len(profile.get("mosa_full", {}).get("stages", [])) == 3
        and profile["vram_reclaimed"]
        and profile["reserve_held"]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(profile, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(profile, indent=2, sort_keys=True))
    if not profile["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
