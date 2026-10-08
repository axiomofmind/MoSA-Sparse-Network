from __future__ import annotations

from pathlib import Path

import yaml

from sparse_network.config import AppConfig


def make_test_config(tmp_path: Path, *, endpoint: dict) -> AppConfig:
    configs = tmp_path / "configs"
    configs.mkdir()
    registry_path = configs / "models.yaml"
    registry_path.write_text(
        yaml.safe_dump(
            {
                "schema": "sparse-network-model-registry.v1",
                "endpoints": {endpoint["id"]: endpoint["definition"]},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    data = {
        "paths": {
            "model_cache": str(tmp_path / "cache"),
            "artifacts": str(tmp_path / "artifacts"),
            "logs": str(tmp_path / "logs"),
            "runs": str(tmp_path / "runs"),
        },
        "runtime_executables": {"llama_cpp": "llama-server"},
        "controller": {
            "bind_host": "127.0.0.1",
            "startup_timeout_seconds": 5,
            "request_timeout_seconds": 1,
            "shutdown_grace_seconds": 1,
            "telemetry_interval_seconds": 0.05,
        },
        "model_registry": str(registry_path),
        "endpoint_overrides": {},
    }
    return AppConfig(root=tmp_path, data=data, sources=())
