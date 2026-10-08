"""Controller-owned artifact persistence."""

from __future__ import annotations

import json
import mimetypes
import os
import tempfile
from contextlib import suppress
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import uuid4


@dataclass(frozen=True)
class ArtifactRecord:
    id: str
    request_id: str
    kind: str
    media_type: str
    path: str
    size_bytes: int
    content_sha256: str
    created_at: str
    provenance: dict[str, Any]
    access_control: dict[str, Any]
    retention: dict[str, Any]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ArtifactStore:
    """Immutable filesystem artifacts with provenance and retention metadata."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve(strict=False)

    def initialize(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _atomic_write(path: Path, content: bytes) -> None:
        if path.exists():
            raise FileExistsError(f"Immutable artifact already exists: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=path.name, dir=path.parent)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        except Exception:
            with suppress(FileNotFoundError):
                os.unlink(temporary)
            raise

    @staticmethod
    def _default_access_control() -> dict[str, Any]:
        return {
            "scope": "private",
            "allowed_principals": [],
            "redaction_state": "none",
        }

    @staticmethod
    def _default_retention() -> dict[str, Any]:
        return {"class": "persistent", "expires_at": None, "pinned": False}

    def _put(
        self,
        *,
        request_id: str,
        kind: str,
        payload: bytes,
        suffix: str,
        media_type: str,
        provenance: dict[str, Any] | None,
        access_control: dict[str, Any] | None,
        retention: dict[str, Any] | None,
        metadata: dict[str, Any] | None,
    ) -> ArtifactRecord:
        self.initialize()
        artifact_id = f"artifact-{uuid4()}"
        request_dir = self.root / request_id
        artifact_path = request_dir / f"{artifact_id}{suffix}"
        self._atomic_write(artifact_path, payload)
        record = ArtifactRecord(
            id=artifact_id,
            request_id=request_id,
            kind=kind,
            media_type=media_type,
            path=str(artifact_path),
            size_bytes=len(payload),
            content_sha256=sha256(payload).hexdigest(),
            created_at=datetime.now(UTC).isoformat(),
            provenance=provenance or {"type": "controller_created", "parents": []},
            access_control={**self._default_access_control(), **(access_control or {})},
            retention={**self._default_retention(), **(retention or {})},
            metadata=metadata or {},
        )
        manifest_path = request_dir / f"{artifact_id}.json"
        self._atomic_write(
            manifest_path,
            json.dumps(record.to_dict(), indent=2, sort_keys=True).encode("utf-8"),
        )
        return record

    def put_text(
        self,
        *,
        request_id: str,
        kind: str,
        text: str,
        provenance: dict[str, Any] | None = None,
        access_control: dict[str, Any] | None = None,
        retention: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactRecord:
        payload = text.encode("utf-8")
        return self._put(
            request_id=request_id,
            kind=kind,
            payload=payload,
            suffix=".txt",
            media_type="text/plain; charset=utf-8",
            provenance=provenance,
            access_control=access_control,
            retention=retention,
            metadata=metadata or {},
        )

    def put_bytes(
        self,
        *,
        request_id: str,
        kind: str,
        payload: bytes,
        suffix: str,
        media_type: str,
        provenance: dict[str, Any] | None = None,
        access_control: dict[str, Any] | None = None,
        retention: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactRecord:
        """Persist generated binary content without routing through a host path."""

        if not suffix.startswith(".") or any(value in suffix for value in ("/", "\\")):
            raise ValueError("artifact suffix must be a simple extension")
        return self._put(
            request_id=request_id,
            kind=kind,
            payload=payload,
            suffix=suffix.lower(),
            media_type=media_type,
            provenance=provenance,
            access_control=access_control,
            retention=retention,
            metadata=metadata or {},
        )

    def put_file(
        self,
        *,
        request_id: str,
        kind: str,
        source: Path,
        provenance: dict[str, Any] | None = None,
        access_control: dict[str, Any] | None = None,
        retention: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactRecord:
        payload = source.read_bytes()
        suffix = source.suffix.lower() or ".bin"
        return self._put(
            request_id=request_id,
            kind=kind,
            payload=payload,
            suffix=suffix,
            media_type=mimetypes.guess_type(source.name)[0] or "application/octet-stream",
            provenance=provenance
            or {"type": "file_import", "parents": [], "source_name": source.name},
            access_control=access_control,
            retention=retention,
            metadata={"source_name": source.name, **(metadata or {})},
        )

    def get(self, artifact_id: str, *, verify_content: bool = True) -> ArtifactRecord:
        manifests = list(self.root.glob(f"*/{artifact_id}.json"))
        if len(manifests) != 1:
            raise FileNotFoundError(f"Unknown artifact: {artifact_id}")
        data = json.loads(manifests[0].read_text(encoding="utf-8"))
        record = ArtifactRecord(**data)
        payload_path = Path(record.path).resolve(strict=True)
        if not payload_path.is_relative_to(self.root):
            raise ValueError(f"Artifact path escapes store root: {artifact_id}")
        if verify_content:
            payload = payload_path.read_bytes()
            if (
                len(payload) != record.size_bytes
                or sha256(payload).hexdigest() != record.content_sha256
            ):
                raise ValueError(f"Artifact content verification failed: {artifact_id}")
        return record

    def read_text(self, artifact_id: str) -> str:
        record = self.get(artifact_id)
        if not record.media_type.startswith("text/"):
            raise ValueError(f"Artifact is not text: {artifact_id}")
        return Path(record.path).read_text(encoding="utf-8")

    def list_records(
        self,
        *,
        request_id: str | None = None,
        kinds: set[str] | None = None,
        limit: int = 200,
    ) -> tuple[ArtifactRecord, ...]:
        """List verified artifact metadata without exposing payload content."""

        if limit <= 0 or limit > 1000:
            raise ValueError("artifact list limit must be 1..1000")
        root = self.root / request_id if request_id else self.root
        pattern = "*.json" if request_id else "*/*.json"
        records: list[ArtifactRecord] = []
        for manifest in root.glob(pattern):
            try:
                record = ArtifactRecord(**json.loads(manifest.read_text(encoding="utf-8")))
            except (OSError, TypeError, json.JSONDecodeError):
                continue
            if kinds is not None and record.kind not in kinds:
                continue
            records.append(record)
        records.sort(key=lambda record: (record.created_at, record.id), reverse=True)
        return tuple(records[:limit])

    def garbage_collect(
        self,
        *,
        now: datetime | None = None,
        protected_ids: set[str] | None = None,
    ) -> tuple[str, ...]:
        """Remove only expired, unpinned, unreferenced artifacts."""

        now = now or datetime.now(UTC)
        manifests = list(self.root.glob("*/*.json"))
        records: list[tuple[Path, ArtifactRecord]] = []
        referenced = set(protected_ids or set())
        for manifest in manifests:
            try:
                record = ArtifactRecord(**json.loads(manifest.read_text(encoding="utf-8")))
            except (OSError, TypeError, json.JSONDecodeError):
                continue
            records.append((manifest, record))
            referenced.update(str(value) for value in record.provenance.get("parents", []))
            referenced.update(
                str(value) for value in record.metadata.get("evidence_references", [])
            )
        removed: list[str] = []
        for manifest, record in records:
            expires_at = record.retention.get("expires_at")
            if not isinstance(expires_at, str) or record.retention.get("pinned", False):
                continue
            expiry = datetime.fromisoformat(expires_at)
            if expiry.tzinfo is None:
                expiry = expiry.replace(tzinfo=UTC)
            if expiry > now or record.id in referenced:
                continue
            payload = Path(record.path).resolve(strict=False)
            if payload.is_relative_to(self.root):
                with suppress(FileNotFoundError):
                    payload.unlink()
                with suppress(FileNotFoundError):
                    manifest.unlink()
                removed.append(record.id)
        return tuple(removed)
