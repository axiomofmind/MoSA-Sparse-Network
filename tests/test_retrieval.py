from __future__ import annotations

from pathlib import Path

import pytest
from conftest import make_test_config

from sparse_network.artifacts import ArtifactStore
from sparse_network.controller import Controller
from sparse_network.models import ModelRegistry
from sparse_network.retrieval import HashEmbeddingBackend, VectorIndex, chunk_text


def _index(tmp_path: Path, *, revision: str = "v1") -> VectorIndex:
    return VectorIndex(
        root=tmp_path / "indexes",
        artifacts=ArtifactStore(tmp_path / "artifacts"),
        backend=HashEmbeddingBackend(revision=revision),
        maximum_chunk_characters=96,
        overlap_characters=12,
    )


def _mock_endpoint() -> dict:
    return {
        "id": "mock-echo",
        "definition": {
            "display_name": "Mock Echo",
            "role": "fixture",
            "source": {"type": "builtin"},
            "runtime": {"adapter": "mock"},
            "modalities": ["text"],
            "capabilities": ["text_generation"],
            "context_size": 4096,
            "max_output_tokens": 512,
            "max_input_characters": 3584,
            "admission": {"state": "admitted"},
        },
    }


def test_chunk_coordinates_reconstruct_original_text() -> None:
    text = "alpha line\nbeta line\n\ngamma paragraph with enough words to split cleanly"
    chunks = chunk_text(text, maximum_characters=64, overlap_characters=8)
    assert len(chunks) >= 2
    for chunk in chunks:
        assert text[chunk.char_start : chunk.char_end] == chunk.text
        assert chunk.line_start >= 1
        assert chunk.line_end >= chunk.line_start


def test_index_search_returns_stable_evidence_reference(tmp_path: Path) -> None:
    source = tmp_path / "manual.md"
    source.write_text(
        "# Cooling\nThe emergency cooling valve is cobalt blue.\n\n"
        "# Network\nThe service port is 8443 and requires TLS.",
        encoding="utf-8",
    )
    index = _index(tmp_path)
    index.backend.start()
    indexed = index.ingest(source)
    first = index.search("What color is the emergency cooling valve?", top_k=1)
    second = index.search("What color is the emergency cooling valve?", top_k=1)
    index.backend.stop()
    assert indexed["chunks_added"] >= 1
    assert "cobalt blue" in first.hits[0].text
    assert first.evidence_references == second.evidence_references
    evidence = ArtifactStore(tmp_path / "artifacts").get(first.evidence_references[0])
    assert evidence.provenance["parents"] == [first.hits[0].document_artifact_id]


def test_embedding_revision_creates_new_index(tmp_path: Path) -> None:
    assert _index(tmp_path, revision="v1").index_id != _index(tmp_path, revision="v2").index_id


def test_invalid_utf8_document_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "broken.md"
    source.write_bytes(b"\xff\xfe\x00")
    with pytest.raises(ValueError, match="not valid UTF-8"):
        _index(tmp_path).ingest(source)


def test_redacted_evidence_is_rejected(tmp_path: Path) -> None:
    endpoint = _mock_endpoint()
    config = make_test_config(tmp_path, endpoint=endpoint)
    store = ArtifactStore(tmp_path / "artifacts")
    evidence = store.put_text(
        request_id="request-source",
        kind="evidence_chunk",
        text="secret",
        access_control={"redaction_state": "pending"},
    )
    with pytest.raises(ValueError, match="not cleared"):
        Controller(config, ModelRegistry.load(config)).smoke(
            endpoint_id="mock-echo",
            prompt="use evidence",
            evidence_references=(evidence.id,),
        )


def test_retrieved_evidence_is_passed_to_endpoint(tmp_path: Path) -> None:
    source = tmp_path / "runbook.md"
    source.write_text(
        "Restart policy: retry twice, then escalate to the operator.", encoding="utf-8"
    )
    index = _index(tmp_path)
    index.backend.start()
    index.ingest(source)
    result = index.search("What is the restart policy?", top_k=1)
    index.backend.stop()
    config = make_test_config(tmp_path, endpoint=_mock_endpoint())
    controller_result = Controller(config, ModelRegistry.load(config)).smoke(
        endpoint_id="mock-echo",
        prompt="What is the restart policy?",
        evidence_references=result.evidence_references,
    )
    assert controller_result.envelope.evidence_references == result.evidence_references
    assert "retry twice" in (controller_result.answer or "")
    assert controller_result.lifecycle["original_request_reference"].startswith("artifact-")
