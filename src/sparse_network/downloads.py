"""Sequential, registry-driven acquisition of pinned Hugging Face artifacts."""

from __future__ import annotations

import shutil
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .errors import ConfigurationError, ModelDownloadError
from .models import ModelRegistry
from .validation import validate_endpoint_artifacts


@dataclass(frozen=True)
class SnapshotDownload:
    repository: str
    revision: str
    files: tuple[str, ...]
    size_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ModelDownload:
    endpoint: str
    display_name: str
    snapshots: tuple[SnapshotDownload, ...]
    license_id: str | None
    license_review_required: bool

    @property
    def size_bytes(self) -> int:
        return sum(snapshot.size_bytes for snapshot in self.snapshots)

    def to_dict(self) -> dict[str, Any]:
        return {
            "endpoint": self.endpoint,
            "display_name": self.display_name,
            "size_bytes": self.size_bytes,
            "license_id": self.license_id,
            "license_review_required": self.license_review_required,
            "snapshots": [snapshot.to_dict() for snapshot in self.snapshots],
        }


@dataclass(frozen=True)
class ModelDownloadPlan:
    items: tuple[ModelDownload, ...]
    skipped: tuple[str, ...]

    @property
    def total_size_bytes(self) -> int:
        return sum(item.size_bytes for item in self.items)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "sparse-network-model-download-plan.v1",
            "total_size_bytes": self.total_size_bytes,
            "items": [item.to_dict() for item in self.items],
            "skipped_non_huggingface": list(self.skipped),
        }


def build_download_plan(
    registry: ModelRegistry,
    *,
    endpoint_ids: Iterable[str] = (),
) -> ModelDownloadPlan:
    """Return admitted Hugging Face endpoints ordered by declared download size."""

    requested = tuple(dict.fromkeys(str(value) for value in endpoint_ids))
    endpoints = (
        tuple(registry.get(endpoint_id) for endpoint_id in requested)
        if requested
        else registry.all()
    )
    items: list[ModelDownload] = []
    skipped: list[str] = []
    for endpoint in endpoints:
        if endpoint.admission.get("state") != "admitted":
            skipped.append(endpoint.id)
            continue
        source = endpoint.source
        snapshots: tuple[SnapshotDownload, ...]
        if source.type == "huggingface":
            if not source.repository or not source.revision:
                raise ConfigurationError(
                    f"Downloadable endpoint {endpoint.id} needs a repository and revision"
                )
            if source.download_size_bytes is None or source.download_size_bytes <= 0:
                raise ConfigurationError(
                    f"Downloadable endpoint {endpoint.id} needs source.download_size_bytes"
                )
            files = tuple(sorted(set(source.files.values())))
            if not files:
                raise ConfigurationError(f"Downloadable endpoint {endpoint.id} has no files")
            snapshots = (
                SnapshotDownload(
                    repository=source.repository,
                    revision=source.revision,
                    files=files,
                    size_bytes=source.download_size_bytes,
                ),
            )
        elif source.type == "huggingface_bundle":
            snapshots = tuple(
                SnapshotDownload(
                    repository=artifact.repository,
                    revision=artifact.revision,
                    files=tuple(sorted(set(artifact.files))),
                    size_bytes=artifact.download_size_bytes,
                )
                for artifact in source.artifacts.values()
            )
            if not snapshots:
                raise ConfigurationError(
                    f"Downloadable endpoint {endpoint.id} has no bundled snapshots"
                )
        else:
            skipped.append(endpoint.id)
            continue
        license_id = endpoint.license.get("id")
        items.append(
            ModelDownload(
                endpoint=endpoint.id,
                display_name=endpoint.display_name,
                snapshots=snapshots,
                license_id=str(license_id) if license_id else None,
                license_review_required=bool(endpoint.license.get("review_required", False)),
            )
        )
    items.sort(key=lambda item: (item.size_bytes, item.endpoint))
    return ModelDownloadPlan(items=tuple(items), skipped=tuple(sorted(skipped)))


def _is_cached(registry: ModelRegistry, item: ModelDownload, cache_dir: Path) -> bool:
    endpoint = registry.get(item.endpoint)
    paths = endpoint.artifact_paths(cache_dir)
    download_paths = [
        cache_dir
        / ("models--" + snapshot.repository.replace("/", "--"))
        / "snapshots"
        / snapshot.revision
        / relative
        for snapshot in item.snapshots
        for relative in snapshot.files
    ]
    return (
        bool(paths)
        and all(path.is_file() for path in paths.values())
        and bool(download_paths)
        and all(path.is_file() for path in download_paths)
    )


def remaining_download_bytes(
    registry: ModelRegistry,
    plan: ModelDownloadPlan,
    cache_dir: Path,
) -> int:
    return sum(
        item.size_bytes for item in plan.items if not _is_cached(registry, item, cache_dir)
    )


def pull_download_plan(
    registry: ModelRegistry,
    plan: ModelDownloadPlan,
    *,
    cache_dir: Path,
    force: bool = False,
    continue_on_error: bool = False,
    downloader: Callable[..., str] | None = None,
    progress: Callable[[int, int, ModelDownload], None] | None = None,
) -> dict[str, Any]:
    """Download and validate each planned snapshot serially."""

    cache_dir = cache_dir.expanduser().resolve(strict=False)
    cache_dir.mkdir(parents=True, exist_ok=True)
    remaining = (
        plan.total_size_bytes
        if force
        else remaining_download_bytes(registry, plan, cache_dir)
    )
    free_bytes = shutil.disk_usage(cache_dir).free
    if remaining > free_bytes:
        raise ModelDownloadError(
            f"Model downloads need up to {remaining} bytes but only {free_bytes} bytes are free"
        )
    if downloader is None:
        from huggingface_hub import snapshot_download

        downloader = snapshot_download

    results: list[dict[str, Any]] = []
    for index, item in enumerate(plan.items, 1):
        if progress is not None:
            progress(index, len(plan.items), item)
        try:
            if not force and _is_cached(registry, item, cache_dir):
                validation = validate_endpoint_artifacts(
                    registry.get(item.endpoint), cache_dir
                )
                status = "already_present"
            else:
                for snapshot in item.snapshots:
                    downloader(
                        repo_id=snapshot.repository,
                        repo_type="model",
                        revision=snapshot.revision,
                        cache_dir=str(cache_dir),
                        allow_patterns=list(snapshot.files),
                    )
                if not _is_cached(registry, item, cache_dir):
                    raise ModelDownloadError(
                        f"Download completed without every declared file for {item.endpoint}"
                    )
                validation = validate_endpoint_artifacts(
                    registry.get(item.endpoint), cache_dir
                )
                status = "downloaded"
            results.append(
                {
                    "endpoint": item.endpoint,
                    "snapshots": [snapshot.to_dict() for snapshot in item.snapshots],
                    "size_bytes": item.size_bytes,
                    "status": status,
                    "artifacts": len(validation.artifacts),
                }
            )
        except Exception as exc:
            results.append(
                {
                    "endpoint": item.endpoint,
                    "snapshots": [snapshot.to_dict() for snapshot in item.snapshots],
                    "size_bytes": item.size_bytes,
                    "status": "failed",
                    "error": str(exc),
                }
            )
            if not continue_on_error:
                raise ModelDownloadError(
                    f"Download failed for {item.endpoint}: {exc}"
                ) from exc
    return {
        "schema": "sparse-network-model-download-result.v1",
        "cache_dir": str(cache_dir),
        "total_size_bytes": plan.total_size_bytes,
        "remaining_size_bytes_at_start": remaining,
        "items": results,
        "skipped_non_huggingface": list(plan.skipped),
        "passed": all(item["status"] != "failed" for item in results),
    }
