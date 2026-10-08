"""Measure model, projector, image, and text-generation VRAM components."""

from __future__ import annotations

import argparse
import json
import threading
from dataclasses import replace
from pathlib import Path
from time import monotonic
from typing import Any

from sparse_network.config import load_config
from sparse_network.models import ModelRegistry
from sparse_network.runtimes import LlamaCppAdapter
from sparse_network.telemetry import ResourceSampler, gpu_memory_used_bytes
from sparse_network.validation import validate_endpoint_artifacts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="gemma-e2b-vision")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = load_config()
    endpoint = ModelRegistry.load(config).get(args.endpoint)
    validate_endpoint_artifacts(endpoint, config.paths["model_cache"])
    paths = endpoint.artifact_paths(config.paths["model_cache"])
    model_path = paths["model"]
    projector_path = paths["projector"]
    logs = config.paths["logs"]
    if logs is None:
        raise ValueError("paths.logs must be configured")
    settings = config.data.get("controller", {})

    def measure(
        name: str,
        *,
        projector: Path | None,
        prompt: str | None = None,
        images: tuple[Path, ...] = (),
        output_tokens: int = 1,
    ) -> dict[str, Any]:
        measured_endpoint = replace(endpoint, max_output_tokens=output_tokens)
        runtime = LlamaCppAdapter(
            measured_endpoint,
            executable=config.runtime_executable("llama_cpp"),
            model_path=model_path,
            projector_path=projector,
            log_path=logs / f"{endpoint.id}-profile-{name}.log",
            host=str(settings.get("bind_host", "127.0.0.1")),
            startup_timeout_seconds=float(settings.get("startup_timeout_seconds", 120)),
            shutdown_grace_seconds=float(settings.get("shutdown_grace_seconds", 10)),
        )
        initial = gpu_memory_used_bytes(endpoint.telemetry_gpu_index)
        started = monotonic()
        runtime.start()
        load_ms = round((monotonic() - started) * 1000)
        loaded = gpu_memory_used_bytes(endpoint.telemetry_gpu_index)
        peak = loaded
        peak_ram = 0
        request_ms = 0
        try:
            if prompt is not None:
                sampler = ResourceSampler(
                    pid=runtime.pid,
                    gpu_index=endpoint.telemetry_gpu_index,
                    interval=float(settings.get("telemetry_interval_seconds", 0.2)),
                )
                sampler.start()
                runtime.generate(
                    prompt=prompt,
                    images=images,
                    cancellation=threading.Event(),
                    timeout_seconds=180,
                )
                snapshot = sampler.stop()
                peak = snapshot.peak_vram_bytes
                peak_ram = snapshot.peak_ram_bytes
                request_ms = snapshot.elapsed_ms
        finally:
            runtime.stop()
        final = gpu_memory_used_bytes(endpoint.telemetry_gpu_index)
        return {
            "initial_vram_bytes": initial,
            "loaded_vram_bytes": loaded,
            "peak_vram_bytes": peak,
            "final_vram_bytes": final,
            "incremental_loaded_vram_bytes": max(0, loaded - initial),
            "incremental_request_vram_bytes": max(0, peak - loaded),
            "peak_process_ram_bytes": peak_ram,
            "load_ms": load_ms,
            "request_ms": request_ms,
        }

    model_only = measure("model", projector=None)
    model_projector = measure("projector", projector=projector_path)
    image_encoding = measure(
        "image",
        projector=projector_path,
        prompt="Identify the image in one word.",
        images=(args.image.resolve(strict=True),),
        output_tokens=1,
    )
    generation = measure(
        "generation",
        projector=projector_path,
        prompt="Write 100 numbered words.",
        output_tokens=128,
    )
    result = {
        "schema": "sparse-network-vision-component-profile.v1",
        "endpoint": endpoint.id,
        "model_revision": endpoint.model_revision,
        "model_only": model_only,
        "model_and_projector": model_projector,
        "image_encoding": image_encoding,
        "text_generation": generation,
        "estimated_projector_vram_bytes": max(
            0,
            model_projector["incremental_loaded_vram_bytes"]
            - model_only["incremental_loaded_vram_bytes"],
        ),
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        output = args.output.resolve(strict=False)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
