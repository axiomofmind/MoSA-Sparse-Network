from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from struct import pack

import pytest

from sparse_network.errors import ArtifactValidationError
from sparse_network.models import EndpointDefinition, ModelSource
from sparse_network.validation import validate_endpoint_artifacts


def endpoint() -> EndpointDefinition:
    return EndpointDefinition(
        id="test-model",
        display_name="Test model",
        role="fixture",
        family="test",
        source=ModelSource(
            type="huggingface",
            repository="example/test-model",
            revision="abc123",
            files={"model": "test.gguf"},
        ),
        runtime={"adapter": "llama_cpp"},
        modalities=("text",),
        capabilities=("text_generation",),
        context_size=128,
        max_output_tokens=16,
        environment={},
        telemetry_gpu_index=None,
        admission={"state": "candidate"},
        license={},
    )


def model_path(cache: Path) -> Path:
    return cache / "models--example--test-model" / "snapshots" / "abc123" / "test.gguf"


def test_valid_gguf_is_accepted(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    path = model_path(cache)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"GGUF" + pack("<IQQ", 3, 0, 0) + b"\x00" * 44)
    result = validate_endpoint_artifacts(endpoint(), cache)
    assert result.valid
    assert result.artifacts[0].size_bytes == 68


def test_invalid_gguf_magic_is_rejected(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    path = model_path(cache)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"NOPE" + b"\x00" * 64)
    with pytest.raises(ArtifactValidationError, match="GGUF magic"):
        validate_endpoint_artifacts(endpoint(), cache)


def test_missing_model_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ArtifactValidationError, match="does not exist"):
        validate_endpoint_artifacts(endpoint(), tmp_path / "cache")


def test_artifact_path_cannot_escape_repository(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    (cache / "models--example--test-model").mkdir(parents=True)
    outside = tmp_path / "outside.gguf"
    outside.write_bytes(b"GGUF" + pack("<IQQ", 3, 0, 0) + b"\x00" * 44)
    escaped = endpoint()
    object.__setattr__(
        escaped,
        "source",
        ModelSource(
            type="huggingface",
            repository="example/test-model",
            revision="abc123",
            files={"model": "../../../../outside.gguf"},
        ),
    )
    with pytest.raises(ArtifactValidationError, match="outside repository root"):
        validate_endpoint_artifacts(escaped, cache)


def test_huggingface_snapshot_symlink_may_use_shared_cache_blob(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    blob = cache / "blobs" / "model"
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"GGUF" + pack("<IQQ", 3, 0, 0) + b"\x00" * 44)
    path = model_path(cache)
    path.parent.mkdir(parents=True)
    try:
        path.symlink_to(blob)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")
    assert validate_endpoint_artifacts(endpoint(), cache).valid


def test_vision_endpoint_requires_projector(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    path = model_path(cache)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"GGUF" + pack("<IQQ", 3, 0, 0) + b"\x00" * 44)
    vision = endpoint()
    object.__setattr__(vision, "modalities", ("text", "image"))
    with pytest.raises(ArtifactValidationError, match="needs a projector"):
        validate_endpoint_artifacts(vision, cache)


def test_transformers_vision_endpoint_does_not_require_external_projector(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "cache"
    vision = endpoint()
    object.__setattr__(
        vision,
        "source",
        ModelSource(
            type="huggingface",
            repository="example/test-model",
            revision="abc123",
            files={"model": "model.safetensors"},
        ),
    )
    path = vision.artifact_paths(cache)["model"]
    path.parent.mkdir(parents=True)
    path.write_bytes(pack("<Q", 2) + b"{}" + b"\x00")
    object.__setattr__(vision, "modalities", ("text", "image"))
    object.__setattr__(vision, "runtime", {"adapter": "decision_transformers"})
    assert validate_endpoint_artifacts(vision, cache).valid


def test_local_source_is_contained_and_hash_verified(tmp_path: Path) -> None:
    local_root = tmp_path / "manual-model"
    local_root.mkdir()
    payload = b"GGUF" + pack("<IQQ", 3, 0, 0) + b"\x00" * 44
    (local_root / "model.gguf").write_bytes(payload)
    local = endpoint()
    object.__setattr__(
        local,
        "source",
        ModelSource(
            type="local",
            revision=f"sha256:{sha256(payload).hexdigest()}",
            base_path=str(local_root),
            files={"model": "model.gguf"},
            hashes={"model": sha256(payload).hexdigest()},
        ),
    )
    assert validate_endpoint_artifacts(local, None).valid
    object.__setattr__(
        local,
        "source",
        ModelSource(
            type="local",
            base_path=str(local_root),
            files={"model": "model.gguf"},
            hashes={"model": "0" * 64},
        ),
    )
    with pytest.raises(ArtifactValidationError, match="SHA-256 mismatch"):
        validate_endpoint_artifacts(local, None)
