"""Read-only host and configuration diagnostics."""

from __future__ import annotations

import platform
import shutil
import sys
from typing import Any

import psutil

from .config import AppConfig
from .models import ModelRegistry
from .telemetry import disk_free_bytes, query_nvidia_gpus


def run_doctor(config: AppConfig, registry: ModelRegistry) -> dict[str, Any]:
    paths = config.paths
    cache = paths["model_cache"]
    artifacts = paths["artifacts"]
    runtime_results: dict[str, dict[str, Any]] = {}
    configured_runtimes = config.data.get("runtime_executables", {})
    for runtime_name, configured in configured_runtimes.items():
        found = None
        if isinstance(configured, str):
            found = shutil.which(configured)
            if found is None:
                candidate = config.root / configured
                if candidate.exists():
                    found = str(candidate.resolve())
        runtime_results[str(runtime_name)] = {
            "configured": configured,
            "resolved": found,
            "available": found is not None,
        }

    endpoints = []
    for endpoint in registry.all():
        try:
            artifact_paths = endpoint.artifact_paths(cache)
            present = all(path.is_file() for path in artifact_paths.values())
        except Exception:
            present = endpoint.source.type == "builtin"
        endpoints.append(
            {
                "id": endpoint.id,
                "adapter": endpoint.adapter,
                "source_type": endpoint.source.type,
                "artifacts_present": present,
            }
        )

    return {
        "mutated": False,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": sys.version.split()[0],
        },
        "memory": {
            "total_bytes": psutil.virtual_memory().total,
            "available_bytes": psutil.virtual_memory().available,
        },
        "gpus": query_nvidia_gpus(),
        "paths": {
            "model_cache": str(cache) if cache else None,
            "model_cache_exists": bool(cache and cache.exists()),
            "artifacts": str(artifacts) if artifacts else None,
            "artifact_disk_free_bytes": disk_free_bytes(artifacts) if artifacts else None,
        },
        "runtimes": runtime_results,
        "configuration_sources": [str(path) for path in config.sources],
        "endpoints": endpoints,
    }
