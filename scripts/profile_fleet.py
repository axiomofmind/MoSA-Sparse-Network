"""Exercise and measure the complete resident fleet on local hardware."""

from __future__ import annotations

import argparse
import json
import threading
from pathlib import Path
from time import monotonic, sleep
from typing import Any

from sparse_network.config import load_config
from sparse_network.contracts import SmokeResult
from sparse_network.fleet import FleetManager, FleetState
from sparse_network.models import ModelRegistry
from sparse_network.telemetry import gpu_memory_used_bytes

TEXT_CASES = (
    ("lfm25-1.2b", "Reply with exactly: FLEET_LFM_OK", "FLEET_LFM_OK"),
    ("qwen35-4b", "Reply with exactly: FLEET_QWEN35_OK", "FLEET_QWEN35_OK"),
    ("nemotron-4b", "Reply with exactly: FLEET_NEMOTRON_OK", "FLEET_NEMOTRON_OK"),
    ("qwen3-8b-fp8", "Reply with exactly: FLEET_QWEN8_OK", "FLEET_QWEN8_OK"),
)


def _weight_bytes(manager: FleetManager) -> tuple[int, dict[str, int]]:
    by_endpoint: dict[str, int] = {}
    seen: set[Path] = set()
    for endpoint_id, entry in manager.entries.items():
        total = 0
        for key, path in entry.endpoint.artifact_paths(manager.config.paths["model_cache"]).items():
            if key != "model" and key != "projector" and not key.startswith("model_shard_"):
                continue
            resolved = path.resolve(strict=True)
            if resolved not in seen:
                total += resolved.stat().st_size
                seen.add(resolved)
        by_endpoint[endpoint_id] = total
    return sum(by_endpoint.values()), by_endpoint


def _record(result: SmokeResult, expected: str) -> dict[str, Any]:
    value = result.to_dict()
    value["expected"] = expected
    value["passed"] = result.envelope.status == "answer" and expected in (result.answer or "")
    return value


def _wait_for_state(
    manager: FleetManager,
    endpoint_id: str,
    state: FleetState,
    *,
    timeout_seconds: float,
) -> bool:
    deadline = monotonic() + timeout_seconds
    while monotonic() < deadline:
        if manager.entries[endpoint_id].state == state:
            return True
        sleep(0.05)
    return False


def _cancellation_and_restore(manager: FleetManager) -> dict[str, Any]:
    endpoint_id = "qwen3-8b-fp8"
    cancellation = threading.Event()
    outcome: dict[str, Any] = {}

    def submit() -> None:
        try:
            result = manager.submit(
                endpoint_id=endpoint_id,
                prompt=(
                    "Write a detailed numbered technical guide with at least 700 tokens about "
                    "testing a local inference scheduler. Do not conclude early."
                ),
                cancellation=cancellation,
                timeout_seconds=180,
                priority=0,
            )
            outcome["request"] = result.to_dict()
        except BaseException as exc:
            outcome["error"] = f"{type(exc).__name__}: {exc}"

    thread = threading.Thread(target=submit, name="fleet-profile-cancellation")
    thread.start()
    reached_busy = _wait_for_state(
        manager,
        endpoint_id,
        FleetState.BUSY,
        timeout_seconds=30,
    )
    if reached_busy:
        sleep(1)
    cancellation.set()
    thread.join(timeout=60)
    outcome["reached_busy"] = reached_busy
    outcome["thread_released"] = not thread.is_alive()
    outcome["state_after_cancellation"] = manager.entries[endpoint_id].state.value
    request = outcome.get("request", {})
    outcome["cancelled"] = request.get("envelope", {}).get("status") == "cancelled"
    if manager.entries[endpoint_id].state == FleetState.FAILED:
        manager.restore_endpoint(endpoint_id)
    outcome["state_after_restore"] = manager.entries[endpoint_id].state.value
    restored = manager.submit(
        endpoint_id=endpoint_id,
        prompt="Reply with exactly: FLEET_RESTORE_OK",
        timeout_seconds=180,
        priority=0,
    )
    outcome["restored_request"] = _record(restored, "FLEET_RESTORE_OK")
    outcome["passed"] = (
        reached_busy
        and not thread.is_alive()
        and bool(outcome["cancelled"])
        and outcome["state_after_restore"] == FleetState.READY.value
        and outcome["restored_request"]["passed"]
    )
    return outcome


def _parallel_stress(manager: FleetManager, image: Path) -> dict[str, Any]:
    work = {
        "qwen3-8b-fp8": {
            "prompt": "Reply with exactly: FLEET_PARALLEL_QWEN_OK",
            "images": (),
            "expected": "FLEET_PARALLEL_QWEN_OK",
        },
        "gemma-e2b-vision": {
            "prompt": "Read the prominent build status. Include exactly: BUILD FAILED",
            "images": (image,),
            "expected": "BUILD FAILED",
        },
    }
    barrier = threading.Barrier(3)
    results: dict[str, Any] = {}
    errors: dict[str, str] = {}

    def submit(endpoint_id: str) -> None:
        case = work[endpoint_id]
        try:
            barrier.wait(timeout=10)
            result = manager.submit(
                endpoint_id=endpoint_id,
                prompt=str(case["prompt"]),
                images=case["images"],
                timeout_seconds=180,
            )
            results[endpoint_id] = _record(result, str(case["expected"]))
        except BaseException as exc:
            errors[endpoint_id] = f"{type(exc).__name__}: {exc}"

    threads = [
        threading.Thread(target=submit, args=(endpoint_id,), name=f"stress-{endpoint_id}")
        for endpoint_id in work
    ]
    for thread in threads:
        thread.start()
    barrier.wait(timeout=10)
    observed_two_active = False
    deadline = monotonic() + 30
    while any(thread.is_alive() for thread in threads) and monotonic() < deadline:
        observed_two_active = observed_two_active or manager.status()["active_requests"] == 2
        sleep(0.05)
    for thread in threads:
        thread.join(timeout=180)
    return {
        "results": results,
        "errors": errors,
        "observed_two_active": observed_two_active,
        "threads_released": all(not thread.is_alive() for thread in threads),
        "passed": (
            not errors
            and len(results) == 2
            and all(value["passed"] for value in results.values())
            and observed_two_active
            and all(not thread.is_alive() for thread in threads)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--parallel", type=int, choices=(1, 2), default=1)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.rounds <= 0:
        parser.error("--rounds must be positive")
    image = args.image.resolve(strict=True)
    output = args.output.resolve(strict=False)
    config = load_config()
    manager = FleetManager(
        config,
        ModelRegistry.load(config),
        maximum_parallel_generations=args.parallel,
    )
    profile: dict[str, Any] = {
        "schema": "sparse-network-fleet-profile.v1",
        "parallel_limit": args.parallel,
        "rounds": args.rounds,
        "requests": [],
    }
    fatal_error: str | None = None
    try:
        weight_total, weight_by_endpoint = _weight_bytes(manager)
        profile["weight_artifact_bytes"] = weight_total
        profile["weight_artifact_bytes_by_endpoint"] = weight_by_endpoint
        profile["loaded"] = manager.load_all()
        for round_number in range(1, args.rounds + 1):
            for endpoint_id, prompt, expected in TEXT_CASES:
                result = manager.submit(
                    endpoint_id=endpoint_id,
                    prompt=prompt,
                    timeout_seconds=180,
                )
                recorded = _record(result, expected)
                recorded["round"] = round_number
                profile["requests"].append(recorded)
            vision = manager.submit(
                endpoint_id="gemma-e2b-vision",
                prompt="Read the prominent build status. Include exactly: BUILD FAILED",
                images=(image,),
                timeout_seconds=180,
            )
            recorded_vision = _record(vision, "BUILD FAILED")
            recorded_vision["round"] = round_number
            profile["requests"].append(recorded_vision)
        profile["cancellation_and_restore"] = _cancellation_and_restore(manager)
        if args.parallel == 2:
            profile["parallel_stress"] = _parallel_stress(manager, image)
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
    peak = manager.peak_vram_bytes
    profile["final_vram_bytes"] = final
    profile["peak_vram_bytes"] = peak
    profile["minimum_vram_reserve_bytes"] = manager.minimum_vram_reserve_bytes
    profile["minimum_observed_headroom_bytes"] = max(0, manager.total_vram_bytes - peak)
    profile["reserve_held"] = (
        not manager.total_vram_bytes
        or peak + manager.minimum_vram_reserve_bytes <= manager.total_vram_bytes
    )
    profile["vram_reclaimed"] = final <= initial + 512 * 1024**2
    profile["passed"] = (
        fatal_error is None
        and "shutdown_error" not in profile
        and all(request["passed"] for request in profile["requests"])
        and profile.get("cancellation_and_restore", {}).get("passed", False)
        and (args.parallel == 1 or profile.get("parallel_stress", {}).get("passed", False))
        and profile["reserve_held"]
        and profile["vram_reclaimed"]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(profile, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(profile, indent=2, sort_keys=True))
    if not profile["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
