"""Measure independent endpoints coexisting before fleet scheduling."""

from __future__ import annotations

import argparse
import json
import threading
from pathlib import Path
from time import monotonic

from sparse_network.config import load_config
from sparse_network.controller import Controller
from sparse_network.models import ModelRegistry
from sparse_network.telemetry import ResourceSampler, gpu_memory_used_bytes
from sparse_network.validation import validate_endpoint_artifacts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--small", default="lfm25-1.2b")
    parser.add_argument("--large", default="qwen3-8b-fp8")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = load_config()
    registry = ModelRegistry.load(config)
    small_endpoint = registry.get(args.small)
    large_endpoint = registry.get(args.large)
    validate_endpoint_artifacts(small_endpoint, config.paths["model_cache"])
    validate_endpoint_artifacts(large_endpoint, config.paths["model_cache"])
    controller = Controller(config, registry)
    small = controller.create_runtime(small_endpoint)
    large = controller.create_runtime(large_endpoint)
    initial = gpu_memory_used_bytes(large_endpoint.telemetry_gpu_index)
    small_load_started = monotonic()
    small.start()
    small_load_ms = round((monotonic() - small_load_started) * 1000)
    after_small = gpu_memory_used_bytes(large_endpoint.telemetry_gpu_index)
    large_load_started = monotonic()
    try:
        large.start()
        large_load_ms = round((monotonic() - large_load_started) * 1000)
        both_loaded = gpu_memory_used_bytes(large_endpoint.telemetry_gpu_index)
        sampler = ResourceSampler(
            pid=None,
            gpu_index=large_endpoint.telemetry_gpu_index,
            interval=0.1,
        )
        sampler.start()
        small_result = small.generate(
            prompt="Reply with exactly: COEXISTENCE_SMALL_OK",
            images=(),
            cancellation=threading.Event(),
            timeout_seconds=60,
        )
        large_result = large.generate(
            prompt="Reply with exactly: COEXISTENCE_OK",
            images=(),
            cancellation=threading.Event(),
            timeout_seconds=180,
        )
        snapshot = sampler.stop()
    finally:
        large.stop()
        small.stop()
    final = gpu_memory_used_bytes(large_endpoint.telemetry_gpu_index)
    result = {
        "schema": "sparse-network-coexistence-profile.v1",
        "small_endpoint": small_endpoint.id,
        "large_endpoint": large_endpoint.id,
        "initial_vram_bytes": initial,
        "after_small_vram_bytes": after_small,
        "both_loaded_vram_bytes": both_loaded,
        "peak_vram_bytes": snapshot.peak_vram_bytes,
        "final_vram_bytes": final,
        "small_load_ms": small_load_ms,
        "large_load_ms": large_load_ms,
        "small_answer": small_result.text,
        "large_answer": large_result.text,
        "answers_valid": (
            "COEXISTENCE_SMALL_OK" in small_result.text
            and "COEXISTENCE_OK" in large_result.text
        ),
        "both_processes_released": small.pid is None and large.pid is None,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        output = args.output.resolve(strict=False)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
