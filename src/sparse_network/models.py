"""Model registry definitions and portable Hugging Face path resolution."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .config import AppConfig, deep_merge
from .errors import ConfigurationError


@dataclass(frozen=True)
class HuggingFaceArtifactSource:
    repository: str
    revision: str
    path: str
    download_size_bytes: int
    files: tuple[str, ...]


@dataclass(frozen=True)
class ModelSource:
    type: str
    repository: str | None = None
    revision: str | None = None
    base_path: str | None = None
    download_size_bytes: int | None = None
    files: dict[str, str] = field(default_factory=dict)
    hashes: dict[str, str] = field(default_factory=dict)
    artifacts: dict[str, HuggingFaceArtifactSource] = field(default_factory=dict)


@dataclass(frozen=True)
class EndpointDefinition:
    id: str
    display_name: str
    role: str
    family: str | None
    source: ModelSource
    runtime: dict[str, Any]
    modalities: tuple[str, ...]
    capabilities: tuple[str, ...]
    context_size: int
    max_output_tokens: int
    environment: dict[str, str]
    telemetry_gpu_index: int | None
    admission: dict[str, Any]
    license: dict[str, Any]
    max_input_characters: int = 0
    max_input_tokens: int = 0
    proposed_budget: dict[str, Any] = field(default_factory=dict)

    @property
    def adapter(self) -> str:
        value = self.runtime.get("adapter")
        if not isinstance(value, str) or not value:
            raise ConfigurationError(f"Endpoint {self.id} has no runtime adapter")
        return value

    @property
    def model_revision(self) -> str | None:
        if self.source.type == "huggingface_bundle":
            return "+".join(
                sorted({artifact.revision for artifact in self.source.artifacts.values()})
            )
        return self.source.revision

    def artifact_paths(self, model_cache: Path | None) -> dict[str, Path]:
        if self.source.type == "builtin":
            return {}
        if self.source.type == "local":
            if not self.source.base_path:
                raise ConfigurationError(
                    f"Local endpoint {self.id} needs source.base_path"
                )
            local_root = Path(self.source.base_path).expanduser()
            if not local_root.is_absolute():
                raise ConfigurationError(
                    f"Local endpoint {self.id} source.base_path must be absolute"
                )
            return {
                name: local_root / relative for name, relative in self.source.files.items()
            }
        if self.source.type == "huggingface_bundle":
            if model_cache is None:
                raise ConfigurationError(
                    f"Endpoint {self.id} needs paths.model_cache in local configuration"
                )
            if not self.source.artifacts:
                raise ConfigurationError(
                    f"Bundled endpoint {self.id} needs source.artifacts"
                )
            return {
                name: (
                    model_cache
                    / ("models--" + artifact.repository.replace("/", "--"))
                    / "snapshots"
                    / artifact.revision
                    / artifact.path
                )
                for name, artifact in self.source.artifacts.items()
            }
        if self.source.type != "huggingface":
            raise ConfigurationError(
                f"Endpoint {self.id} uses unsupported source type {self.source.type!r}"
            )
        if model_cache is None:
            raise ConfigurationError(
                f"Endpoint {self.id} needs paths.model_cache in local configuration"
            )
        if not self.source.repository or not self.source.revision:
            raise ConfigurationError(
                f"Endpoint {self.id} needs a repository and pinned revision"
            )
        repository_dir = "models--" + self.source.repository.replace("/", "--")
        snapshot_root = model_cache / repository_dir / "snapshots" / self.source.revision
        return {name: snapshot_root / relative for name, relative in self.source.files.items()}


def _endpoint_from_mapping(endpoint_id: str, data: dict[str, Any]) -> EndpointDefinition:
    source_data = data.get("source", {})
    if not isinstance(source_data, dict):
        raise ConfigurationError(f"Endpoint {endpoint_id} source must be a mapping")
    runtime = data.get("runtime", {})
    if not isinstance(runtime, dict):
        raise ConfigurationError(f"Endpoint {endpoint_id} runtime must be a mapping")
    source_files = source_data.get("files", {})
    if not isinstance(source_files, dict):
        raise ConfigurationError(f"Endpoint {endpoint_id} source.files must be a mapping")
    artifact_sources = source_data.get("artifacts", {})
    if not isinstance(artifact_sources, dict):
        raise ConfigurationError(f"Endpoint {endpoint_id} source.artifacts must be a mapping")
    parsed_artifacts: dict[str, HuggingFaceArtifactSource] = {}
    for name, value in artifact_sources.items():
        if not isinstance(value, dict):
            raise ConfigurationError(
                f"Endpoint {endpoint_id} source artifact {name} must be a mapping"
            )
        files = value.get("files", [])
        if not isinstance(files, list):
            raise ConfigurationError(
                f"Endpoint {endpoint_id} source artifact {name} files must be a list"
            )
        try:
            parsed_artifacts[str(name)] = HuggingFaceArtifactSource(
                repository=str(value["repository"]),
                revision=str(value["revision"]),
                path=str(value["path"]),
                download_size_bytes=int(value["download_size_bytes"]),
                files=tuple(str(file) for file in files),
            )
        except KeyError as exc:
            raise ConfigurationError(
                f"Endpoint {endpoint_id} source artifact {name} is incomplete"
            ) from exc
    environment = data.get("environment", {})
    if not isinstance(environment, dict):
        raise ConfigurationError(f"Endpoint {endpoint_id} environment must be a mapping")
    telemetry_gpu_index = data.get("telemetry_gpu_index")
    if telemetry_gpu_index is not None and not isinstance(telemetry_gpu_index, int):
        raise ConfigurationError(f"Endpoint {endpoint_id} telemetry_gpu_index must be an integer")
    return EndpointDefinition(
        id=endpoint_id,
        display_name=str(data.get("display_name", endpoint_id)),
        role=str(data.get("role", "unspecified")),
        family=data.get("family"),
        source=ModelSource(
            type=str(source_data.get("type", "")),
            repository=source_data.get("repository"),
            revision=source_data.get("revision"),
            base_path=source_data.get("base_path"),
            download_size_bytes=(
                int(source_data["download_size_bytes"])
                if source_data.get("download_size_bytes") is not None
                else None
            ),
            files={str(key): str(value) for key, value in source_files.items()},
            hashes={
                str(key): str(value).lower()
                for key, value in source_data.get("hashes", {}).items()
            },
            artifacts=parsed_artifacts,
        ),
        runtime=copy.deepcopy(runtime),
        modalities=tuple(str(value) for value in data.get("modalities", [])),
        capabilities=tuple(str(value) for value in data.get("capabilities", [])),
        context_size=int(data.get("context_size", 0)),
        max_output_tokens=int(data.get("max_output_tokens", 0)),
        max_input_characters=int(data.get("max_input_characters", 0)),
        max_input_tokens=int(data.get("max_input_tokens", 0)),
        environment={str(key): str(value) for key, value in environment.items()},
        telemetry_gpu_index=telemetry_gpu_index,
        admission=copy.deepcopy(data.get("admission", {})),
        license=copy.deepcopy(data.get("license", {})),
        proposed_budget=copy.deepcopy(data.get("proposed_budget", {})),
    )


class ModelRegistry:
    """Loaded endpoint registry with local endpoint overrides applied."""

    def __init__(self, endpoints: dict[str, EndpointDefinition]) -> None:
        self._endpoints = endpoints

    @classmethod
    def load(cls, config: AppConfig, *, apply_overrides: bool = True) -> ModelRegistry:
        path = config.model_registry_path
        if not path.exists():
            raise ConfigurationError(f"Model registry does not exist: {path}")
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            raise ConfigurationError(f"Invalid model registry YAML: {exc}") from exc
        endpoint_map = raw.get("endpoints")
        if not isinstance(endpoint_map, dict):
            raise ConfigurationError("Model registry must contain an endpoints mapping")
        overrides = config.data.get("endpoint_overrides", {}) if apply_overrides else {}
        if not isinstance(overrides, dict):
            raise ConfigurationError("endpoint_overrides must be a mapping")
        endpoints: dict[str, EndpointDefinition] = {}
        for endpoint_id, endpoint_data in endpoint_map.items():
            if not isinstance(endpoint_data, dict):
                raise ConfigurationError(f"Endpoint {endpoint_id} must be a mapping")
            local_override = overrides.get(endpoint_id, {})
            if not isinstance(local_override, dict):
                raise ConfigurationError(f"Override for {endpoint_id} must be a mapping")
            merged = deep_merge(endpoint_data, local_override)
            endpoints[str(endpoint_id)] = _endpoint_from_mapping(str(endpoint_id), merged)
        return cls(endpoints)

    def get(self, endpoint_id: str) -> EndpointDefinition:
        try:
            return self._endpoints[endpoint_id]
        except KeyError as exc:
            choices = ", ".join(sorted(self._endpoints))
            raise ConfigurationError(
                f"Unknown endpoint {endpoint_id!r}; available: {choices}"
            ) from exc

    def all(self) -> tuple[EndpointDefinition, ...]:
        return tuple(self._endpoints[key] for key in sorted(self._endpoints))
