"""Read-only model artifact validation."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from .errors import ArtifactValidationError
from .models import EndpointDefinition

_HASH_CACHE: dict[tuple[str, int, int, str], bool] = {}


@dataclass(frozen=True)
class ArtifactCheck:
    name: str
    path: str
    resolved_path: str
    size_bytes: int
    format: str
    valid: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EndpointValidation:
    endpoint: str
    source_type: str
    repository: str | None
    revision: str | None
    valid: bool
    artifacts: tuple[ArtifactCheck, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "endpoint": self.endpoint,
            "source_type": self.source_type,
            "repository": self.repository,
            "revision": self.revision,
            "valid": self.valid,
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
        }


def _format_for(path: Path) -> str:
    return path.suffix.lower().lstrip(".") or "unknown"


def _validate_file(
    path: Path,
    repository_root: Path,
    name: str,
    *,
    resolved_root: Path | None = None,
) -> ArtifactCheck:
    declared = Path(os.path.abspath(path))
    declared_repository = Path(os.path.abspath(repository_root))
    if not declared.is_relative_to(declared_repository):
        raise ArtifactValidationError(
            f"Artifact {path} resolves outside repository root {declared_repository}"
        )
    if not path.exists():
        raise ArtifactValidationError(f"Required artifact does not exist: {path}")
    if not path.is_file():
        raise ArtifactValidationError(f"Required artifact is not a file: {path}")
    resolved = path.resolve(strict=True)
    allowed_root = (resolved_root or repository_root).resolve(strict=True)
    if not resolved.is_relative_to(allowed_root):
        raise ArtifactValidationError(
            f"Artifact {path} resolves outside allowed cache root {allowed_root}"
        )

    size = resolved.stat().st_size
    if size <= 8:
        raise ArtifactValidationError(f"Artifact is implausibly small: {resolved}")

    artifact_format = _format_for(path)
    if artifact_format == "gguf":
        with resolved.open("rb") as handle:
            if handle.read(4) != b"GGUF":
                raise ArtifactValidationError(f"GGUF magic is invalid: {resolved}")
            version = int.from_bytes(handle.read(4), byteorder="little", signed=False)
            tensor_count = int.from_bytes(handle.read(8), byteorder="little", signed=False)
            metadata_count = int.from_bytes(handle.read(8), byteorder="little", signed=False)
        if version not in {2, 3}:
            raise ArtifactValidationError(f"Unsupported GGUF version {version}: {resolved}")
        if tensor_count > 10_000_000 or metadata_count > 10_000_000:
            raise ArtifactValidationError(f"Implausible GGUF header counts: {resolved}")
    elif artifact_format == "safetensors":
        with resolved.open("rb") as handle:
            header_size = int.from_bytes(handle.read(8), byteorder="little", signed=False)
        if header_size <= 0 or header_size >= size:
            raise ArtifactValidationError(f"Safetensors header is invalid: {resolved}")

    return ArtifactCheck(
        name=name,
        path=str(path),
        resolved_path=str(resolved),
        size_bytes=size,
        format=artifact_format,
        valid=True,
    )


def _validate_expected_hash(path: Path, expected: str) -> None:
    normalized = expected.removeprefix("sha256:").lower()
    if len(normalized) != 64 or any(value not in "0123456789abcdef" for value in normalized):
        raise ArtifactValidationError(f"Invalid expected SHA-256 for {path.name}")
    stat = path.stat()
    key = (str(path.resolve(strict=True)), stat.st_size, stat.st_mtime_ns, normalized)
    if key in _HASH_CACHE:
        return
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024**2):
            digest.update(chunk)
    if digest.hexdigest() != normalized:
        raise ArtifactValidationError(f"SHA-256 mismatch: {path}")
    _HASH_CACHE[key] = True


def validate_endpoint_artifacts(
    endpoint: EndpointDefinition,
    model_cache: Path | None,
) -> EndpointValidation:
    """Validate all files for an endpoint without loading its model."""

    if endpoint.source.type == "builtin":
        return EndpointValidation(
            endpoint=endpoint.id,
            source_type="builtin",
            repository=None,
            revision=None,
            valid=True,
            artifacts=(),
        )

    paths = endpoint.artifact_paths(model_cache)
    if not paths:
        raise ArtifactValidationError(f"Endpoint {endpoint.id} has no required files")
    if endpoint.source.type == "local":
        if endpoint.source.base_path is None:
            raise ArtifactValidationError(f"Local endpoint {endpoint.id} needs a base path")
        repository_root = Path(endpoint.source.base_path)
    elif endpoint.source.type == "huggingface":
        if model_cache is None or endpoint.source.repository is None:
            raise ArtifactValidationError(f"Cannot determine repository root for {endpoint.id}")
        repository_root = (
            model_cache / ("models--" + endpoint.source.repository.replace("/", "--"))
        )
    elif endpoint.source.type == "huggingface_bundle":
        if model_cache is None:
            raise ArtifactValidationError(f"Cannot determine cache root for {endpoint.id}")
        repository_root = model_cache
    else:
        raise ArtifactValidationError(
            f"Cannot validate source type {endpoint.source.type!r} for {endpoint.id}"
        )
    resolved_root = (
        model_cache
        if endpoint.source.type in {"huggingface", "huggingface_bundle"}
        else repository_root
    )
    checks = tuple(
        _validate_file(
            path,
            (
                model_cache
                / (
                    "models--"
                    + endpoint.source.artifacts[name].repository.replace("/", "--")
                )
                if endpoint.source.type == "huggingface_bundle" and model_cache is not None
                else repository_root
            ),
            name,
            resolved_root=resolved_root,
        )
        for name, path in sorted(paths.items())
    )
    for name, expected in endpoint.source.hashes.items():
        path = paths.get(name)
        if path is None:
            raise ArtifactValidationError(
                f"Endpoint {endpoint.id} declares a hash for unknown artifact {name}"
            )
        _validate_expected_hash(path.resolve(strict=True), expected)
    if endpoint.runtime.get("artifact_class") == "complete_model":
        model_path = paths.get("model")
        if model_path is None:
            raise ArtifactValidationError(
                f"Complete-model endpoint {endpoint.id} has no model artifact"
            )
        lowered = model_path.name.lower()
        if any(marker in lowered for marker in ("dflash", "draft", "mtp")):
            raise ArtifactValidationError(
                f"Complete-model endpoint {endpoint.id} references an auxiliary artifact"
            )
        minimum_size = int(endpoint.runtime.get("minimum_complete_model_bytes", 256 * 1024**2))
        if model_path.resolve(strict=True).stat().st_size < minimum_size:
            raise ArtifactValidationError(
                f"Complete-model endpoint {endpoint.id} is smaller than its declared minimum"
            )
    if (
        endpoint.adapter == "llama_cpp"
        and "image" in endpoint.modalities
        and "projector" not in paths
    ):
        raise ArtifactValidationError(
            f"Vision endpoint {endpoint.id} needs a projector from its pinned revision"
        )
    for name, index_path in paths.items():
        if not index_path.name.endswith(".safetensors.index.json"):
            continue
        try:
            index_data = json.loads(index_path.read_text(encoding="utf-8"))
            weight_map = index_data["weight_map"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ArtifactValidationError(
                f"Safetensors index is invalid: {index_path}"
            ) from exc
        if not isinstance(weight_map, dict) or not weight_map:
            raise ArtifactValidationError(f"Safetensors index has no weight map: {index_path}")
        declared_files = {path.name for path in paths.values()}
        missing_shards = sorted(set(weight_map.values()) - declared_files)
        if missing_shards:
            raise ArtifactValidationError(
                f"Safetensors index {name} references undeclared shards: {missing_shards}"
            )
    return EndpointValidation(
        endpoint=endpoint.id,
        source_type=endpoint.source.type,
        repository=(
            endpoint.source.repository
            if endpoint.source.type == "huggingface"
            else (
                ",".join(
                    sorted(
                        {
                            artifact.repository
                            for artifact in endpoint.source.artifacts.values()
                        }
                    )
                )
                if endpoint.source.type == "huggingface_bundle"
                else endpoint.source.base_path
            )
        ),
        revision=endpoint.model_revision,
        valid=all(check.valid for check in checks),
        artifacts=checks,
    )
