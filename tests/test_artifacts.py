from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from sparse_network.artifacts import ArtifactStore


def test_artifact_records_hash_provenance_and_access(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    record = store.put_text(
        request_id="request-test",
        kind="tool_result",
        text="immutable",
        provenance={"type": "tool", "parents": ["artifact-parent"]},
        access_control={"redaction_state": "reviewed"},
    )
    loaded = store.get(record.id)
    assert loaded.content_sha256 == record.content_sha256
    assert loaded.provenance["parents"] == ["artifact-parent"]
    assert loaded.access_control["scope"] == "private"
    assert loaded.access_control["redaction_state"] == "reviewed"


def test_artifact_tampering_is_detected(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    record = store.put_text(request_id="request-test", kind="answer", text="original")
    Path(record.path).write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="verification failed"):
        store.get(record.id)


def test_garbage_collection_preserves_referenced_artifact(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    expired = (datetime.now(UTC) - timedelta(days=1)).isoformat()
    parent = store.put_text(
        request_id="request-test",
        kind="original_document",
        text="parent",
        retention={"expires_at": expired},
    )
    child = store.put_text(
        request_id="request-test",
        kind="evidence_chunk",
        text="child",
        provenance={"type": "chunk", "parents": [parent.id]},
        retention={"expires_at": expired},
    )
    removed = store.garbage_collect()
    assert parent.id not in removed
    assert child.id in removed
    assert store.get(parent.id).id == parent.id
