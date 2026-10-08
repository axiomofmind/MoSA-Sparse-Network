"""Versioned CPU embedding indexes with stable artifact provenance."""

from __future__ import annotations

import json
import math
import os
import queue
import re
import shutil
import sqlite3
import subprocess
import threading
from array import array
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from time import monotonic
from typing import Any, Protocol, TextIO
from uuid import uuid4

from .artifacts import ArtifactStore
from .config import AppConfig
from .errors import ConfigurationError, RequestFailedError, RuntimeUnavailableError
from .models import EndpointDefinition, ModelRegistry
from .validation import validate_endpoint_artifacts

TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)


@dataclass(frozen=True)
class TextChunk:
    ordinal: int
    text: str
    char_start: int
    char_end: int
    line_start: int
    line_end: int


@dataclass(frozen=True)
class SearchHit:
    artifact_id: str
    document_artifact_id: str
    source_name: str
    text: str
    char_start: int
    char_end: int
    line_start: int
    line_end: int
    semantic_score: float
    rerank_score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SearchResult:
    index_id: str
    embedding_endpoint: str
    embedding_revision: str
    query: str
    hits: tuple[SearchHit, ...]
    elapsed_ms: int

    @property
    def evidence_references(self) -> tuple[str, ...]:
        return tuple(hit.artifact_id for hit in self.hits)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "sparse-network-retrieval-result.v1",
            "index_id": self.index_id,
            "embedding_endpoint": self.embedding_endpoint,
            "embedding_revision": self.embedding_revision,
            "query": self.query,
            "evidence_references": list(self.evidence_references),
            "elapsed_ms": self.elapsed_ms,
            "hits": [hit.to_dict() for hit in self.hits],
        }


class EmbeddingBackend(Protocol):
    endpoint_id: str
    revision: str
    dimension: int

    @property
    def pid(self) -> int | None: ...

    def start(self) -> None: ...

    def encode(self, texts: list[str], *, mode: str) -> list[list[float]]: ...

    def stop(self) -> None: ...


class HashEmbeddingBackend:
    """Deterministic model-free backend for tests and clean-clone demos."""

    def __init__(
        self,
        *,
        endpoint_id: str = "hash-embedding",
        revision: str = "v1",
        dimension: int = 256,
    ) -> None:
        self.endpoint_id = endpoint_id
        self.revision = revision
        self.dimension = dimension

    @property
    def pid(self) -> int | None:
        return None

    def start(self) -> None:
        return

    def encode(self, texts: list[str], *, mode: str) -> list[list[float]]:
        del mode
        vectors: list[list[float]] = []
        for text in texts:
            vector = [0.0] * self.dimension
            for token in TOKEN_PATTERN.findall(text.casefold()):
                digest = sha256(token.encode("utf-8")).digest()
                index = int.from_bytes(digest[:4], "little") % self.dimension
                vector[index] += 1.0 if digest[4] & 1 else -1.0
            norm = math.sqrt(sum(value * value for value in vector)) or 1.0
            vectors.append([value / norm for value in vector])
        return vectors

    def stop(self) -> None:
        return


class WorkerEmbeddingBackend:
    def __init__(
        self,
        *,
        endpoint: EndpointDefinition,
        executable: str,
        model_path: Path,
        worker_path: Path,
        log_path: Path,
        startup_timeout_seconds: float,
    ) -> None:
        self.endpoint = endpoint
        self.endpoint_id = endpoint.id
        self.revision = endpoint.model_revision or "unversioned"
        self.dimension = int(endpoint.runtime["dimension"])
        self.executable = executable
        self.model_path = model_path
        self.worker_path = worker_path
        self.log_path = log_path
        self.startup_timeout_seconds = startup_timeout_seconds
        self._process: subprocess.Popen[str] | None = None
        self._log_handle: TextIO | None = None
        self._messages: queue.Queue[dict[str, Any]] = queue.Queue()
        self._reader: threading.Thread | None = None

    @property
    def pid(self) -> int | None:
        return self._process.pid if self._process is not None else None

    def _resolved_executable(self) -> str:
        candidate = Path(self.executable)
        if candidate.exists():
            return str(candidate.resolve())
        located = shutil.which(self.executable)
        if located:
            return located
        raise RuntimeUnavailableError(f"Embedding Python executable not found: {self.executable}")

    def _read_messages(self) -> None:
        process = self._process
        if process is None or process.stdout is None:
            return
        for line in process.stdout:
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                message = {"event": "protocol_error", "error": line.rstrip()}
            if isinstance(message, dict):
                self._messages.put(message)

    def _next_message(self, timeout: float) -> dict[str, Any]:
        try:
            return self._messages.get(timeout=max(0.05, timeout))
        except queue.Empty as exc:
            raise TimeoutError from exc

    def _log_tail(self) -> str:
        try:
            content = self.log_path.read_text(encoding="utf-8", errors="replace")
            return "\n".join(content.splitlines()[-60:])
        except OSError:
            return ""

    def start(self) -> None:
        if self._process is not None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = self.log_path.open("a", encoding="utf-8")
        runtime = self.endpoint.runtime
        command = [
            self._resolved_executable(),
            str(self.worker_path),
            "--model",
            str(self.model_path),
        ]
        if self.endpoint.adapter == "embedding_sentence_transformers":
            command.extend(
                [
                    "--dimension",
                    str(self.dimension),
                    "--max-length",
                    str(int(runtime.get("max_length", 8192))),
                    "--batch-size",
                    str(int(runtime.get("batch_size", 8))),
                    "--threads",
                    str(int(runtime.get("threads", 0))),
                    "--query-prompt",
                    str(runtime.get("query_prompt", "SearchQuery")),
                    "--document-prompt",
                    str(runtime.get("document_prompt", "Document")),
                ]
            )
            if tuple(runtime.get("active_modalities", ())) == ("text",):
                command.append("--text-only")
        else:
            command.extend(
                [
                    "--pooling",
                    str(runtime["pooling"]),
                    "--dimension",
                    str(self.dimension),
                    "--max-length",
                    str(int(runtime.get("max_length", 2048))),
                    "--threads",
                    str(int(runtime.get("threads", 0))),
                    "--query-instruction",
                    str(runtime.get("query_instruction", "")),
                ]
            )
        environment = os.environ.copy()
        environment.update(self.endpoint.environment)
        environment["PYTHONNOUSERSITE"] = "1"
        creation_flags = int(getattr(subprocess, "CREATE_NO_WINDOW", 0)) if os.name == "nt" else 0
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._log_handle,
            text=True,
            encoding="utf-8",
            bufsize=1,
            env=environment,
            creationflags=creation_flags,
        )
        self._reader = threading.Thread(target=self._read_messages, daemon=True)
        self._reader.start()
        deadline = monotonic() + self.startup_timeout_seconds
        while monotonic() < deadline:
            process = self._process
            if process.poll() is not None and self._messages.empty():
                self.stop()
                raise RuntimeUnavailableError(f"Embedding worker exited early: {self._log_tail()}")
            try:
                message = self._next_message(min(0.25, deadline - monotonic()))
            except TimeoutError:
                continue
            if message.get("event") == "ready":
                if int(message.get("dimension", 0)) != self.dimension:
                    self.stop()
                    raise RuntimeUnavailableError("Embedding worker dimension mismatch")
                return
            if message.get("event") in {"startup_error", "protocol_error"}:
                self.stop()
                raise RuntimeUnavailableError(f"Embedding worker failed: {message.get('error')}")
        self.stop()
        raise RuntimeUnavailableError("Embedding worker startup timed out")

    def encode(self, texts: list[str], *, mode: str) -> list[list[float]]:
        process = self._process
        if process is None or process.stdin is None:
            raise RequestFailedError("Embedding worker is not running")
        process.stdin.write(json.dumps({"command": "embed", "mode": mode, "texts": texts}) + "\n")
        process.stdin.flush()
        deadline = monotonic() + 300
        while monotonic() < deadline:
            if process.poll() is not None and self._messages.empty():
                raise RequestFailedError(f"Embedding worker exited: {self._log_tail()}")
            try:
                message = self._next_message(min(0.25, deadline - monotonic()))
            except TimeoutError:
                continue
            if message.get("event") == "error":
                raise RequestFailedError(f"Embedding failed: {message.get('error')}")
            if message.get("event") == "embeddings":
                raw_vectors = message.get("vectors")
                if not isinstance(raw_vectors, list):
                    raise RequestFailedError("Embedding worker returned invalid vectors")
                return [[float(value) for value in vector] for vector in raw_vectors]
        raise RequestFailedError("Embedding request timed out")

    def stop(self) -> None:
        process = self._process
        if process is not None and process.poll() is None and process.stdin is not None:
            try:
                process.stdin.write('{"command":"shutdown"}\n')
                process.stdin.flush()
                process.wait(timeout=10)
            except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                process.kill()
                process.wait(timeout=5)
        self._process = None
        if self._reader is not None:
            self._reader.join(timeout=1)
            self._reader = None
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None


def chunk_text(
    text: str,
    *,
    maximum_characters: int,
    overlap_characters: int,
) -> tuple[TextChunk, ...]:
    if maximum_characters < 64:
        raise ValueError("maximum_characters must be at least 64")
    if overlap_characters < 0 or overlap_characters >= maximum_characters:
        raise ValueError("overlap_characters must be between zero and the chunk size")
    chunks: list[TextChunk] = []
    start = 0
    ordinal = 0
    while start < len(text):
        end = min(len(text), start + maximum_characters)
        if end < len(text):
            lower_bound = start + maximum_characters // 2
            candidates = [
                text.rfind("\n\n", lower_bound, end),
                text.rfind("\n", lower_bound, end),
                text.rfind(" ", lower_bound, end),
            ]
            split = max(candidates)
            if split > start:
                end = split + (2 if text[split : split + 2] == "\n\n" else 1)
        chunk_value = text[start:end]
        if chunk_value.strip():
            chunks.append(
                TextChunk(
                    ordinal=ordinal,
                    text=chunk_value,
                    char_start=start,
                    char_end=end,
                    line_start=text.count("\n", 0, start) + 1,
                    line_end=text.count("\n", 0, max(start, end - 1)) + 1,
                )
            )
            ordinal += 1
        if end >= len(text):
            break
        start = max(start + 1, end - overlap_characters)
    return tuple(chunks)


def _serialize_vector(vector: list[float]) -> bytes:
    values = array("f", vector)
    return values.tobytes()


def _deserialize_vector(payload: bytes) -> list[float]:
    values = array("f")
    values.frombytes(payload)
    return list(values)


def _dot(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Embedding dimensions do not match")
    return sum(a * b for a, b in zip(left, right, strict=True))


def _lexical_score(query: str, passage: str) -> float:
    query_tokens = set(TOKEN_PATTERN.findall(query.casefold()))
    if not query_tokens:
        return 0.0
    passage_tokens = set(TOKEN_PATTERN.findall(passage.casefold()))
    return len(query_tokens & passage_tokens) / len(query_tokens)


class VectorIndex:
    def __init__(
        self,
        *,
        root: Path,
        artifacts: ArtifactStore,
        backend: EmbeddingBackend,
        maximum_chunk_characters: int = 1200,
        overlap_characters: int = 120,
    ) -> None:
        self.root = root.resolve(strict=False)
        self.artifacts = artifacts
        self.backend = backend
        self.maximum_chunk_characters = maximum_chunk_characters
        self.overlap_characters = overlap_characters
        identity = (
            f"{backend.endpoint_id}\0{backend.revision}\0{backend.dimension}\0"
            f"{maximum_chunk_characters}\0{overlap_characters}\0chunk-v1"
        )
        self.index_id = f"index-{sha256(identity.encode('utf-8')).hexdigest()[:20]}"
        self.directory = self.root / self.index_id
        self.database_path = self.directory / "vectors.sqlite"

    def _connect(self) -> sqlite3.Connection:
        self.directory.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.database_path)
        connection.execute("PRAGMA foreign_keys = ON")
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS manifest (
                singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                index_id TEXT NOT NULL,
                endpoint_id TEXT NOT NULL,
                revision TEXT NOT NULL,
                dimension INTEGER NOT NULL,
                chunk_characters INTEGER NOT NULL,
                overlap_characters INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS documents (
                artifact_id TEXT PRIMARY KEY,
                source_name TEXT NOT NULL,
                content_sha256 TEXT NOT NULL UNIQUE
            );
            CREATE TABLE IF NOT EXISTS chunks (
                artifact_id TEXT PRIMARY KEY,
                document_artifact_id TEXT NOT NULL REFERENCES documents(artifact_id),
                ordinal INTEGER NOT NULL,
                text TEXT NOT NULL,
                char_start INTEGER NOT NULL,
                char_end INTEGER NOT NULL,
                line_start INTEGER NOT NULL,
                line_end INTEGER NOT NULL,
                vector BLOB NOT NULL,
                UNIQUE(document_artifact_id, ordinal)
            );
            """
        )
        row = connection.execute(
            """
            SELECT index_id, endpoint_id, revision, dimension,
                   chunk_characters, overlap_characters
            FROM manifest WHERE singleton = 1
            """
        ).fetchone()
        expected = (
            self.index_id,
            self.backend.endpoint_id,
            self.backend.revision,
            self.backend.dimension,
            self.maximum_chunk_characters,
            self.overlap_characters,
        )
        if row is None:
            connection.execute(
                "INSERT INTO manifest VALUES (1, ?, ?, ?, ?, ?, ?)", expected
            )
            connection.commit()
        elif tuple(row) != expected:
            connection.close()
            raise ConfigurationError("Vector index manifest does not match its embedding space")
        return connection

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": "sparse-network-vector-index.v1",
            "index_id": self.index_id,
            "embedding_endpoint": self.backend.endpoint_id,
            "embedding_revision": self.backend.revision,
            "dimension": self.backend.dimension,
            "maximum_chunk_characters": self.maximum_chunk_characters,
            "overlap_characters": self.overlap_characters,
            "database": str(self.database_path),
        }

    def ingest(self, source: Path) -> dict[str, Any]:
        resolved = source.resolve(strict=True)
        supported = {".txt", ".md", ".rst", ".log", ".csv", ".json", ".yaml", ".yml"}
        if resolved.suffix.casefold() not in supported:
            raise ValueError(f"Unsupported text document type: {resolved.name}")
        payload = resolved.read_bytes()
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"Document is not valid UTF-8: {resolved.name}") from exc
        content_hash = sha256(payload).hexdigest()
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT artifact_id FROM documents WHERE content_sha256 = ?", (content_hash,)
            ).fetchone()
            if existing is not None:
                return {**self.manifest(), "document_artifact_id": existing[0], "chunks_added": 0}
        request_id = f"request-{uuid4()}"
        document = self.artifacts.put_file(
            request_id=request_id,
            kind="original_document",
            source=resolved,
            provenance={"type": "document_import", "parents": [], "source_name": resolved.name},
            metadata={"derived": False},
        )
        chunks = chunk_text(
            text,
            maximum_characters=self.maximum_chunk_characters,
            overlap_characters=self.overlap_characters,
        )
        if not chunks:
            raise ValueError(f"Document contains no indexable text: {resolved.name}")
        vectors = self.backend.encode([chunk.text for chunk in chunks], mode="passage")
        if len(vectors) != len(chunks):
            raise RequestFailedError("Embedding backend returned the wrong vector count")
        rows: list[tuple[Any, ...]] = []
        for chunk, vector in zip(chunks, vectors, strict=True):
            if len(vector) != self.backend.dimension:
                raise RequestFailedError("Embedding backend returned the wrong dimension")
            evidence = self.artifacts.put_text(
                request_id=request_id,
                kind="evidence_chunk",
                text=chunk.text,
                provenance={"type": "document_chunk", "parents": [document.id]},
                metadata={
                    "derived": True,
                    "source_name": resolved.name,
                    "ordinal": chunk.ordinal,
                    "char_start": chunk.char_start,
                    "char_end": chunk.char_end,
                    "line_start": chunk.line_start,
                    "line_end": chunk.line_end,
                    "index_id": self.index_id,
                },
            )
            rows.append(
                (
                    evidence.id,
                    document.id,
                    chunk.ordinal,
                    chunk.text,
                    chunk.char_start,
                    chunk.char_end,
                    chunk.line_start,
                    chunk.line_end,
                    _serialize_vector(vector),
                )
            )
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO documents VALUES (?, ?, ?)",
                (document.id, resolved.name, document.content_sha256),
            )
            connection.executemany(
                "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", rows
            )
            connection.commit()
        return {**self.manifest(), "document_artifact_id": document.id, "chunks_added": len(rows)}

    def search(self, query: str, *, top_k: int = 5) -> SearchResult:
        if not query.strip():
            raise ValueError("query must not be empty")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        started = monotonic()
        query_vector = self.backend.encode([query], mode="query")[0]
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT chunks.artifact_id, chunks.document_artifact_id,
                       documents.source_name, chunks.text, chunks.char_start,
                       chunks.char_end, chunks.line_start, chunks.line_end,
                       chunks.vector
                FROM chunks JOIN documents
                  ON documents.artifact_id = chunks.document_artifact_id
                """
            ).fetchall()
        candidates: list[SearchHit] = []
        for row in rows:
            semantic = _dot(query_vector, _deserialize_vector(row[8]))
            lexical = _lexical_score(query, str(row[3]))
            candidates.append(
                SearchHit(
                    artifact_id=str(row[0]),
                    document_artifact_id=str(row[1]),
                    source_name=str(row[2]),
                    text=str(row[3]),
                    char_start=int(row[4]),
                    char_end=int(row[5]),
                    line_start=int(row[6]),
                    line_end=int(row[7]),
                    semantic_score=round(semantic, 8),
                    rerank_score=round(semantic * 0.85 + lexical * 0.15, 8),
                )
            )
        candidates.sort(key=lambda hit: (-hit.rerank_score, hit.artifact_id))
        return SearchResult(
            index_id=self.index_id,
            embedding_endpoint=self.backend.endpoint_id,
            embedding_revision=self.backend.revision,
            query=query,
            hits=tuple(candidates[:top_k]),
            elapsed_ms=round((monotonic() - started) * 1000),
        )


def create_embedding_backend(
    config: AppConfig,
    registry: ModelRegistry,
    endpoint_id: str,
) -> EmbeddingBackend:
    if endpoint_id == "hash-embedding":
        return HashEmbeddingBackend()
    endpoint = registry.get(endpoint_id)
    supported_adapters = {"embedding_transformers", "embedding_sentence_transformers"}
    if endpoint.adapter not in supported_adapters:
        raise ConfigurationError(f"Endpoint {endpoint_id} is not an embedding endpoint")
    validate_endpoint_artifacts(endpoint, config.paths["model_cache"])
    paths = endpoint.artifact_paths(config.paths["model_cache"])
    config_path = paths.get("config")
    if config_path is None:
        raise ConfigurationError(f"Embedding endpoint {endpoint_id} has no config artifact")
    logs = config.paths["logs"]
    if logs is None:
        raise ConfigurationError("paths.logs must be configured")
    controller = config.data.get("controller", {})
    worker_name = (
        "embeddinggemma_worker.py"
        if endpoint.adapter == "embedding_sentence_transformers"
        else "embedding_worker.py"
    )
    return WorkerEmbeddingBackend(
        endpoint=endpoint,
        executable=config.runtime_executable("embeddings"),
        model_path=config_path.parent,
        worker_path=config.root / "src" / "sparse_network" / worker_name,
        log_path=logs / f"{endpoint.id}-embedding.log",
        startup_timeout_seconds=float(controller.get("startup_timeout_seconds", 120)),
    )


def create_vector_index(
    config: AppConfig,
    registry: ModelRegistry,
    endpoint_id: str,
) -> tuple[EmbeddingBackend, VectorIndex]:
    indexes = config.paths["indexes"]
    artifacts = config.paths["artifacts"]
    if indexes is None or artifacts is None:
        raise ConfigurationError("paths.indexes and paths.artifacts must be configured")
    retrieval = config.data.get("retrieval", {})
    backend = create_embedding_backend(config, registry, endpoint_id)
    return backend, VectorIndex(
        root=indexes,
        artifacts=ArtifactStore(artifacts),
        backend=backend,
        maximum_chunk_characters=int(retrieval.get("maximum_chunk_characters", 1200)),
        overlap_characters=int(retrieval.get("overlap_characters", 120)),
    )
