"""Exercise Milestone 7 static routes against the complete resident fleet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import monotonic, sleep
from typing import Any

from sparse_network.config import load_config
from sparse_network.execution import ExecutionRequest, Top1Executor
from sparse_network.fleet import FleetManager
from sparse_network.models import ModelRegistry
from sparse_network.routing import RouteRequest, StaticRouter
from sparse_network.telemetry import gpu_memory_used_bytes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    image = args.image.resolve(strict=True)
    output = args.output.resolve(strict=False)
    config = load_config()
    registry = ModelRegistry.load(config)
    manager = FleetManager(config, registry)
    router = StaticRouter(config, registry, resident_endpoint_ids=set(manager.entries))
    executor = Top1Executor(manager, router)
    cases: list[dict[str, Any]] = [
        {
            "name": "classification",
            "route": RouteRequest(
                prompt="@route:classify Reply with exactly: ROUTE_LFM_OK"
            ),
            "endpoint": "lfm25-1.2b",
            "expected": "ROUTE_LFM_OK",
            "status": "accepted",
        },
        {
            "name": "general_generation",
            "route": RouteRequest(
                prompt="@route:general Reply with exactly: ROUTE_QWEN35_OK"
            ),
            "endpoint": "qwen35-4b",
            "expected": "ROUTE_QWEN35_OK",
            "status": "accepted",
        },
        {
            "name": "tool_verification",
            "route": RouteRequest(
                prompt="@route:verify Reply with exactly: ROUTE_NEMOTRON_OK"
            ),
            "endpoint": "nemotron-4b",
            "expected": "ROUTE_NEMOTRON_OK",
            "status": "accepted",
        },
        {
            "name": "visual_understanding",
            "route": RouteRequest(
                prompt="@route:vision Read the prominent status and include exactly: BUILD FAILED",
                images=(image,),
            ),
            "endpoint": "gemma-e2b-vision",
            "expected": "BUILD FAILED",
            "status": "accepted",
        },
        {
            "name": "difficult_reasoning",
            "route": RouteRequest(
                prompt="@route:reason Reply with exactly: ROUTE_QWEN8_OK"
            ),
            "endpoint": "qwen3-8b-fp8",
            "expected": "ROUTE_QWEN8_OK",
            "status": "accepted",
        },
        {
            "name": "high_risk_hold",
            "route": RouteRequest(
                prompt="Reply with exactly: ROUTE_HUMAN_REVIEW_OK",
                high_risk=True,
            ),
            "endpoint": "qwen3-8b-fp8",
            "expected": "ROUTE_HUMAN_REVIEW_OK",
            "status": "needs_human_review",
        },
    ]
    profile: dict[str, Any] = {
        "schema": "sparse-network-routing-profile.v1",
        "cases": [],
    }
    fatal_error: str | None = None
    try:
        profile["loaded"] = manager.load_all()
        for case in cases:
            result = executor.run(
                ExecutionRequest(
                    route=case["route"],
                    expected_contains=str(case["expected"]),
                    timeout_seconds=180,
                )
            )
            invocation_count = sum(
                node.kind == "model_invoke" for node in result.graph.nodes
            )
            profile["cases"].append(
                {
                    "name": case["name"],
                    "expected_endpoint": case["endpoint"],
                    "expected_status": case["status"],
                    "decision": result.decision.to_dict(),
                    "status": result.status,
                    "invocation_count": invocation_count,
                    "verification": (
                        result.verification.to_dict() if result.verification else None
                    ),
                    "answer": (
                        result.model_result.answer if result.model_result is not None else None
                    ),
                    "trace_path": str(result.trace_path),
                    "passed": (
                        result.decision.endpoint == case["endpoint"]
                        and result.status == case["status"]
                        and invocation_count == 1
                        and result.verification is not None
                        and result.verification.accepted
                    ),
                }
            )

        denied = executor.run(
            ExecutionRequest(
                route=RouteRequest(
                    prompt="Run an untrusted shell",
                    requested_tools=("shell",),
                    authorized_tools=("shell",),
                )
            )
        )
        audio = executor.run(
            ExecutionRequest(
                route=RouteRequest(
                    prompt="Transcribe this",
                    audio=(Path("unavailable-audio.wav"),),
                )
            )
        )
        profile["controlled_rejections"] = {
            "tool": denied.to_dict(),
            "audio": audio.to_dict(),
            "passed": (
                denied.status == "route_rejected"
                and audio.status == "route_rejected"
                and all(node.kind != "model_invoke" for node in denied.graph.nodes)
                and all(node.kind != "model_invoke" for node in audio.graph.nodes)
            ),
        }
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
        and len(profile["cases"]) == len(cases)
        and all(case["passed"] for case in profile["cases"])
        and profile.get("controlled_rejections", {}).get("passed", False)
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
