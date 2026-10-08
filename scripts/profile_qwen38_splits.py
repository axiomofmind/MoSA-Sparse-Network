"""Measure candidate GPU-layer splits for the exclusive Qwen3.8 endpoint."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from sparse_network.config import AppConfig, load_config
from sparse_network.controller import Controller
from sparse_network.models import ModelRegistry
from sparse_network.validation import validate_endpoint_artifacts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, nargs="+", default=[48, 52, 56])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base = load_config()
    profiles: list[dict[str, Any]] = []
    for layers in args.layers:
        data = copy.deepcopy(base.data)
        data.setdefault("endpoint_overrides", {}).setdefault("qwen38-27b", {}).setdefault(
            "runtime", {}
        )["gpu_layers"] = layers
        config = AppConfig(root=base.root, data=data, sources=base.sources)
        registry = ModelRegistry.load(config)
        endpoint = registry.get("qwen38-27b")
        validate_endpoint_artifacts(endpoint, config.paths["model_cache"])
        try:
            result = Controller(config, registry).smoke(
                endpoint_id="qwen38-27b",
                prompt="Reply with exactly: QWEN38_SPLIT_OK",
                timeout_seconds=240,
            )
            profiles.append(
                {
                    "gpu_layers": layers,
                    "passed": (
                        result.envelope.status == "answer"
                        and "QWEN38_SPLIT_OK" in (result.answer or "")
                        and result.lifecycle["vram_reclaimed_within_tolerance"]
                    ),
                    "result": result.to_dict(),
                }
            )
        except BaseException as exc:
            profiles.append(
                {
                    "gpu_layers": layers,
                    "passed": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    value = {
        "schema": "sparse-network-qwen38-split-profile.v1",
        "profiles": profiles,
        "passed": any(profile["passed"] for profile in profiles),
    }
    output = args.output.resolve(strict=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(value, indent=2, sort_keys=True))
    if not value["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
