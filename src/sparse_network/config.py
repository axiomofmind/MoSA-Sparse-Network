"""Portable layered configuration loading."""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .errors import ConfigurationError


def project_root() -> Path:
    """Return the source checkout root for the current package."""

    return Path(__file__).resolve().parents[2]


def _read_yaml(path: Path, *, required: bool) -> dict[str, Any]:
    if not path.exists():
        if required:
            raise ConfigurationError(f"Configuration file does not exist: {path}")
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Invalid YAML in {path}: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigurationError(f"Top-level configuration must be a mapping: {path}")
    return data


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge mappings while replacing scalar and list values."""

    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _resolve_path(value: str | None, root: Path) -> Path | None:
    if value is None:
        return None
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    return candidate.resolve(strict=False)


@dataclass(frozen=True)
class AppConfig:
    """Validated application configuration and its project root."""

    root: Path
    data: dict[str, Any]
    sources: tuple[Path, ...]

    @property
    def paths(self) -> dict[str, Path | None]:
        configured = self.data.get("paths", {})
        return {
            name: _resolve_path(configured.get(name), self.root)
            for name in ("model_cache", "artifacts", "logs", "runs", "indexes")
        }

    @property
    def model_registry_path(self) -> Path:
        value = self.data.get("model_registry")
        if not isinstance(value, str) or not value:
            raise ConfigurationError("model_registry must be a non-empty path")
        return _resolve_path(value, self.root)  # type: ignore[return-value]

    def runtime_executable(self, adapter: str) -> str:
        runtimes = self.data.get("runtime_executables", {})
        value = runtimes.get(adapter)
        if not isinstance(value, str) or not value:
            raise ConfigurationError(f"No executable configured for runtime adapter {adapter!r}")
        return value


def load_config(
    *,
    root: Path | None = None,
    local_config: Path | None = None,
) -> AppConfig:
    """Load defaults, an optional local file, and bounded environment overrides."""

    root = (root or project_root()).resolve()
    default_path = root / "configs" / "default.yaml"
    data = _read_yaml(default_path, required=True)
    sources: list[Path] = [default_path]

    env_config = os.environ.get("SPARSE_CONFIG")
    selected_local = local_config
    if selected_local is None and env_config:
        selected_local = Path(env_config)
    if selected_local is None:
        candidate = root / "config.local.yaml"
        if candidate.exists():
            selected_local = candidate

    if selected_local is not None:
        selected_local = selected_local.expanduser()
        if not selected_local.is_absolute():
            selected_local = root / selected_local
        selected_local = selected_local.resolve(strict=False)
        data = deep_merge(data, _read_yaml(selected_local, required=True))
        sources.append(selected_local)

    env_path_map = {
        "SPARSE_MODEL_CACHE": "model_cache",
        "SPARSE_ARTIFACT_DIR": "artifacts",
        "SPARSE_LOG_DIR": "logs",
        "SPARSE_RUN_DIR": "runs",
    }
    for env_name, path_name in env_path_map.items():
        if value := os.environ.get(env_name):
            data.setdefault("paths", {})[path_name] = value
    if value := os.environ.get("SPARSE_LLAMA_SERVER"):
        data.setdefault("runtime_executables", {})["llama_cpp"] = value

    return AppConfig(root=root, data=data, sources=tuple(sources))
