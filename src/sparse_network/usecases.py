"""Versioned, policy-bounded use-case workflows and immutable run manifests."""

from __future__ import annotations

import base64
import csv
import io
import json
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from collections.abc import Callable
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic
from typing import Any
from uuid import uuid4

import pypdfium2 as pdfium  # type: ignore[import-untyped]
import yaml
from pypdf import PdfReader

from .config import AppConfig
from .coordination import CoordinationContext, StageHandoff
from .errors import ConfigurationError, RequestFailedError
from .fleet import FleetManager, FleetState
from .models import EndpointDefinition, ModelRegistry
from .retrieval import chunk_text, create_vector_index

USE_CASE_SCHEMA = "sparse-network-use-case-config.v1"
RUN_SCHEMA = "sparse-network-use-case-run.v1"
PRESENTATION_SCHEMA = "sparse-network-result-view.v1"
READINESS_SCHEMA = "sparse-network-use-case-readiness.v1"
INPUT_KINDS_WITH_DIRECTORIES = {"repository", "working_directory"}
VISUAL_KINDS = {"screenshot", "diagram", "page_image", "image", "figure", "image_record"}
MEDIA_KINDS = {"audio", "video"}
TEXT_SUFFIXES = {
    ".cfg",
    ".conf",
    ".csv",
    ".diff",
    ".ini",
    ".js",
    ".json",
    ".jsonl",
    ".log",
    ".md",
    ".patch",
    ".py",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".yaml",
    ".yml",
}
FACT_CLASSES = {"observed", "inferred", "proposed", "action"}
DEFAULT_OCR_MODES = ("fast", "complex")


class UseCaseManager:
    """Execute finite templates without allowing model-authored graph expansion."""

    def __init__(
        self,
        config: AppConfig,
        registry: ModelRegistry,
        fleet: FleetManager,
    ) -> None:
        self.config = config
        self.registry = registry
        self.fleet = fleet
        routing = config.data.get("routing", {})
        configured = routing.get("use_cases") if isinstance(routing, dict) else None
        self.path: Path | None = None
        raw: dict[str, Any] = {
            "schema": USE_CASE_SCHEMA,
            "version": 1,
            "maximum_model_stages": 3,
            "maximum_inputs": 100,
            "maximum_batch_records": 1000,
            "allowed_source_roots": [],
            "tools": {},
            "templates": {},
        }
        if isinstance(configured, str) and configured:
            path = Path(configured)
            if not path.is_absolute():
                path = config.root / path
            self.path = path.resolve(strict=True)
            try:
                loaded = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
            except yaml.YAMLError as exc:
                raise ConfigurationError(f"Invalid use-case configuration: {exc}") from exc
            if not isinstance(loaded, dict):
                raise ConfigurationError("Use-case configuration must be a mapping")
            raw = loaded
        if raw.get("schema") != USE_CASE_SCHEMA:
            raise ConfigurationError("Unsupported or missing use-case configuration schema")
        self.version = int(raw.get("version", 1))
        self.maximum_model_stages = int(raw.get("maximum_model_stages", 3))
        self.maximum_inputs = int(raw.get("maximum_inputs", 100))
        self.maximum_batch_records = int(raw.get("maximum_batch_records", 1000))
        if not 1 <= self.maximum_model_stages <= 3:
            raise ConfigurationError("Use-case model stages must be bounded to 1..3")
        tools = raw.get("tools", {})
        templates = raw.get("templates", {})
        if not isinstance(tools, dict) or not isinstance(templates, dict):
            raise ConfigurationError("Use-case tools and templates must be mappings")
        self.tools = {str(key): deepcopy(value) for key, value in tools.items()}
        self.templates = {str(key): deepcopy(value) for key, value in templates.items()}
        local_policy = config.data.get("use_cases", {})
        if not isinstance(local_policy, dict):
            raise ConfigurationError("use_cases local policy must be a mapping")
        self.allowed_roots = self._resolve_roots(
            local_policy.get(
                "allowed_source_roots", raw.get("allowed_source_roots", [])
            )
        )
        runs = config.paths["runs"]
        if runs is None:
            raise ConfigurationError("paths.runs must be configured")
        self.runs_root = runs / "use-cases"
        self._validate()

    def _resolve_roots(self, values: Any) -> tuple[Path, ...]:
        if not isinstance(values, list):
            raise ConfigurationError("allowed_source_roots must be a list")
        roots: list[Path] = []
        for value in values:
            path = Path(str(value))
            if not path.is_absolute():
                path = self.config.root / path
            roots.append(path.resolve(strict=False))
        return tuple(roots)

    def _validate(self) -> None:
        for tool_id, tool in self.tools.items():
            if not isinstance(tool, dict):
                raise ConfigurationError(f"Tool {tool_id} must be a mapping")
            if tool.get("mode") not in {"read_only", "state_changing"}:
                raise ConfigurationError(f"Tool {tool_id} has an invalid mode")
            argv = tool.get("argv")
            if (
                not isinstance(argv, list)
                or not argv
                or not all(isinstance(value, str) and value for value in argv)
            ):
                raise ConfigurationError(f"Tool {tool_id} needs a fixed argv list")
        for template_id, template in self.templates.items():
            if not isinstance(template, dict):
                raise ConfigurationError(f"Template {template_id} must be a mapping")
            stages = template.get("stages", [])
            if not isinstance(stages, list) or len(stages) > self.maximum_model_stages:
                raise ConfigurationError(f"Template {template_id} exceeds the model-stage bound")
            allowed_tools = template.get("allowed_tools", [])
            if not isinstance(allowed_tools, list) or set(allowed_tools) - set(self.tools):
                raise ConfigurationError(f"Template {template_id} references unknown tools")
            for stage in stages:
                if not isinstance(stage, dict) or not stage.get("id"):
                    raise ConfigurationError(f"Template {template_id} has an invalid stage")
                preferred_by_mode = stage.get("preferred_by_ocr_mode")
                if preferred_by_mode is not None and (
                    template.get("kind") != "document"
                    or not isinstance(preferred_by_mode, dict)
                    or set(preferred_by_mode) != set(DEFAULT_OCR_MODES)
                    or not all(
                        isinstance(values, list)
                        and values
                        and all(isinstance(value, str) and value for value in values)
                        for values in preferred_by_mode.values()
                    )
                ):
                    raise ConfigurationError(
                        f"Template {template_id} has invalid OCR endpoint preferences"
                    )
            task_ui = template.get("task_ui", {})
            if not isinstance(task_ui, dict):
                raise ConfigurationError(f"Template {template_id} task_ui must be a mapping")
            for required in ("action", "prompt_label", "default_path_kind", "category"):
                if task_ui and not str(task_ui.get(required, "")).strip():
                    raise ConfigurationError(
                        f"Template {template_id} task_ui needs {required}"
                    )
            if task_ui and task_ui.get("default_path_kind") not in template.get(
                "input_kinds", []
            ):
                raise ConfigurationError(
                    f"Template {template_id} task_ui default path kind is not admitted"
                )
            if task_ui and task_ui.get("default_paste_kind") not in template.get(
                "input_kinds", []
            ):
                raise ConfigurationError(
                    f"Template {template_id} task_ui default paste kind is not admitted"
                )
            ocr_modes = task_ui.get("ocr_modes", []) if task_ui else []
            if ocr_modes:
                if template.get("kind") != "document" or not isinstance(ocr_modes, list):
                    raise ConfigurationError(
                        f"Template {template_id} has invalid OCR mode configuration"
                    )
                mode_ids = {
                    str(value.get("id", ""))
                    for value in ocr_modes
                    if isinstance(value, dict)
                }
                if mode_ids != set(DEFAULT_OCR_MODES):
                    raise ConfigurationError(
                        f"Template {template_id} OCR modes must be fast and complex"
                    )
                if task_ui.get("default_ocr_mode", "fast") not in mode_ids:
                    raise ConfigurationError(
                        f"Template {template_id} default OCR mode is invalid"
                    )

    def _endpoint_available(self, endpoint_id: str) -> bool:
        try:
            endpoint = self.registry.get(endpoint_id)
        except ConfigurationError:
            return False
        entry = self.fleet.entries.get(endpoint.id)
        return entry is not None and entry.state in {FleetState.READY, FleetState.BUSY}

    @staticmethod
    def _ocr_mode(template: dict[str, Any], payload: dict[str, Any]) -> str | None:
        if str(template.get("kind")) != "document":
            return None
        task_ui = template.get("task_ui", {})
        configured = task_ui.get("ocr_modes", []) if isinstance(task_ui, dict) else []
        allowed = {
            str(value.get("id", ""))
            for value in configured
            if isinstance(value, dict) and value.get("id")
        } or set(DEFAULT_OCR_MODES)
        default = str(task_ui.get("default_ocr_mode", "fast"))
        mode = str(payload.get("ocr_mode", default)).strip().lower()
        if mode not in allowed:
            raise ValueError(f"unsupported OCR mode: {mode}")
        return mode

    def _configured_stages(
        self, template: dict[str, Any], payload: dict[str, Any]
    ) -> list[dict[str, Any]]:
        ocr_mode = self._ocr_mode(template, payload)
        stages: list[dict[str, Any]] = []
        for value in template.get("stages", []):
            stage = deepcopy(value)
            by_mode = stage.get("preferred_by_ocr_mode", {})
            if ocr_mode is not None and stage.get("id") == "extract":
                if isinstance(by_mode, dict) and ocr_mode in by_mode:
                    preferred = by_mode[ocr_mode]
                    if not isinstance(preferred, list):
                        raise ConfigurationError(
                            f"OCR stage preference for {ocr_mode} must be a list"
                        )
                    stage["preferred"] = [str(endpoint_id) for endpoint_id in preferred]
                stage["ocr_mode"] = ocr_mode
            stages.append(stage)
        return stages

    def catalog(self) -> dict[str, Any]:
        templates: list[dict[str, Any]] = []
        for template_id, template in self.templates.items():
            stages: list[dict[str, Any]] = []
            for value in template.get("stages", []):
                stage = deepcopy(value)
                preferred_by_mode = stage.get("preferred_by_ocr_mode", {})
                available_by_mode: dict[str, list[str]] = {}
                if isinstance(preferred_by_mode, dict):
                    for mode, values in preferred_by_mode.items():
                        if not isinstance(values, list):
                            continue
                        available_by_mode[str(mode)] = [
                            str(endpoint_id)
                            for endpoint_id in values
                            if self._endpoint_available(str(endpoint_id))
                        ]
                preferred = [str(item) for item in stage.get("preferred", [])]
                preferred.extend(
                    str(endpoint_id)
                    for values in preferred_by_mode.values()
                    if isinstance(values, list)
                    for endpoint_id in values
                )
                stage["available_endpoints"] = [
                    endpoint_id
                    for endpoint_id in dict.fromkeys(preferred)
                    if self._endpoint_available(endpoint_id)
                ]
                if available_by_mode:
                    stage["available_by_ocr_mode"] = available_by_mode
                resolved = self._resolve_endpoint(stage)
                if resolved is not None and resolved.id not in stage["available_endpoints"]:
                    stage["available_endpoints"].append(resolved.id)
                if resolved is not None and available_by_mode:
                    for endpoints in available_by_mode.values():
                        if not endpoints and resolved.id == "mock-echo":
                            endpoints.append(resolved.id)
                stage["available"] = bool(stage["available_endpoints"])
                stages.append(stage)
            tool_values = []
            for tool_id in template.get("allowed_tools", []):
                tool = self.tools[str(tool_id)]
                argv = [str(value) for value in tool["argv"]]
                executable = sys.executable if argv[0] == "{python}" else argv[0]
                tool_values.append(
                    {
                        "id": tool_id,
                        "label": tool.get("label", tool_id),
                        "mode": tool["mode"],
                        "available": Path(executable).is_file()
                        or shutil.which(executable) is not None,
                    }
                )
            templates.append(
                {
                    "id": template_id,
                    "version": self.version,
                    **{
                        key: deepcopy(template.get(key))
                        for key in (
                            "title",
                            "kind",
                            "description",
                            "input_kinds",
                            "frozen_escalation_triggers",
                            "dashboard_panels",
                            "task_ui",
                        )
                    },
                    "stages": stages,
                    "tools": tool_values,
                }
            )
        return {
            "schema": "sparse-network-use-case-catalog.v1",
            "config_version": self.version,
            "maximum_model_stages": self.maximum_model_stages,
            "templates": templates,
        }

    def _safe_source(self, value: str) -> Path:
        source = Path(value).expanduser().resolve(strict=True)
        if not self.allowed_roots or not any(
            source == root or source.is_relative_to(root) for root in self.allowed_roots
        ):
            raise PermissionError("Local source is outside the configured use-case roots")
        return source

    def _directory_artifact(self, run_id: str, kind: str, source: Path) -> str:
        files = sorted(
            str(path.relative_to(source)).replace("\\", "/")
            for path in source.rglob("*")
            if path.is_file()
        )[:2000]
        record = self.fleet.controller.artifacts.put_text(
            request_id=run_id,
            kind=kind,
            text=json.dumps({"root_name": source.name, "files": files}, indent=2),
            provenance={"type": "directory_manifest", "parents": []},
            metadata={"source_name": source.name, "file_count": len(files)},
        )
        return record.id

    def _render_pdf_pages(self, run_id: str, source: Path, parent_id: str) -> list[str]:
        configured = self.config.data.get("runtime_executables", {}).get("pdf_renderer", "pdftoppm")
        executable = shutil.which(str(configured))
        page_ids: list[str] = []
        with tempfile.TemporaryDirectory(prefix="sparse-pdf-") as temporary:
            prefix = Path(temporary) / "page"
            if executable is not None:
                completed = subprocess.run(
                    [executable, "-png", str(source), str(prefix)],
                    check=False,
                    capture_output=True,
                    timeout=120,
                )
                if completed.returncode != 0:
                    raise RequestFailedError("Configured PDF renderer failed")
            else:
                try:
                    document = pdfium.PdfDocument(str(source))
                    for page_number in range(len(document)):
                        rendered = document[page_number].render(scale=2.0).to_pil()
                        rendered.save(prefix.parent / f"page-{page_number + 1}.png")
                    document.close()
                except Exception as exc:
                    raise RequestFailedError(
                        f"Built-in PDF page rendering failed: {exc}"
                    ) from exc
            for page_number, page in enumerate(sorted(prefix.parent.glob("page-*.png")), 1):
                artifact = self.fleet.controller.artifacts.put_file(
                    request_id=run_id,
                    kind="page_image",
                    source=page,
                    provenance={"type": "pdf_page_render", "parents": [parent_id]},
                    metadata={"page_number": page_number, "source_name": source.name},
                )
                page_ids.append(artifact.id)
        if not page_ids:
            raise RequestFailedError("PDF renderer produced no page images")
        return page_ids

    def _text_evidence(
        self,
        run_id: str,
        artifact_id: str,
        *,
        page_number: int | None = None,
    ) -> list[str]:
        """Create bounded, citable chunks from one immutable text artifact."""

        record = self.fleet.controller.artifacts.get(artifact_id)
        is_text = record.media_type.startswith("text/") or Path(record.path).suffix.casefold() in (
            TEXT_SUFFIXES
        )
        if not is_text or record.kind == "evidence_chunk":
            return [record.id] if record.kind == "evidence_chunk" else []
        text = (
            self.fleet.controller.artifacts.read_text(record.id)
            if record.media_type.startswith("text/")
            else Path(record.path).read_text(encoding="utf-8")
        )
        retrieval = self.config.data.get("retrieval", {})
        chunks = chunk_text(
            text,
            maximum_characters=int(retrieval.get("maximum_chunk_characters", 1200)),
            overlap_characters=int(retrieval.get("overlap_characters", 120)),
        )
        source_name = str(record.metadata.get("source_name") or record.kind)
        evidence_ids: list[str] = []
        for chunk in chunks:
            evidence = self.fleet.controller.artifacts.put_text(
                request_id=run_id,
                kind="evidence_chunk",
                text=chunk.text,
                provenance={"type": "use_case_source_chunk", "parents": [record.id]},
                access_control=record.access_control,
                retention=record.retention,
                metadata={
                    "derived": True,
                    "source_name": source_name,
                    "ordinal": chunk.ordinal,
                    "char_start": chunk.char_start,
                    "char_end": chunk.char_end,
                    "line_start": chunk.line_start,
                    "line_end": chunk.line_end,
                    **({"page_number": page_number} if page_number is not None else {}),
                },
            )
            evidence_ids.append(evidence.id)
        return evidence_ids

    def _pdf_evidence(self, run_id: str, artifact_id: str) -> list[str]:
        """Extract text from digital PDF pages; scanned pages remain image-only."""

        record = self.fleet.controller.artifacts.get(artifact_id)
        try:
            reader = PdfReader(record.path)
        except Exception:
            return []
        evidence_ids: list[str] = []
        for page_number, page in enumerate(reader.pages, 1):
            try:
                page_text = str(page.extract_text() or "").strip()
            except Exception:
                page_text = ""
            if not page_text:
                continue
            page_artifact = self.fleet.controller.artifacts.put_text(
                request_id=run_id,
                kind="extracted_page_text",
                text=page_text,
                provenance={"type": "pdf_text_extraction", "parents": [record.id]},
                access_control=record.access_control,
                retention=record.retention,
                metadata={
                    "derived": True,
                    "source_name": str(record.metadata.get("source_name") or "document.pdf"),
                    "page_number": page_number,
                },
            )
            evidence_ids.extend(
                self._text_evidence(run_id, page_artifact.id, page_number=page_number)
            )
        return evidence_ids

    def _input_pdf_has_text(self, value: dict[str, Any]) -> bool:
        try:
            if value.get("artifact_id"):
                record = self.fleet.controller.artifacts.get(str(value["artifact_id"]))
                reader = PdfReader(record.path)
            elif value.get("path"):
                reader = PdfReader(self._safe_source(str(value["path"])))
            elif value.get("content_base64"):
                payload = base64.b64decode(str(value["content_base64"]), validate=True)
                reader = PdfReader(io.BytesIO(payload))
            else:
                return False
            return any(str(page.extract_text() or "").strip() for page in reader.pages)
        except Exception:
            return False

    def _ingest_inputs(
        self,
        run_id: str,
        template: dict[str, Any],
        values: Any,
    ) -> tuple[list[dict[str, Any]], list[Path]]:
        if not isinstance(values, list) or len(values) > self.maximum_inputs:
            raise ValueError(f"inputs must be a list of at most {self.maximum_inputs} items")
        admitted_kinds = {str(value) for value in template.get("input_kinds", [])}
        ingested: list[dict[str, Any]] = []
        image_paths: list[Path] = []
        for index, value in enumerate(values):
            if not isinstance(value, dict):
                raise ValueError("each use-case input must be an object")
            kind = str(value.get("kind", ""))
            if kind not in admitted_kinds:
                raise ValueError(f"input kind is not admitted by this template: {kind}")
            metadata = deepcopy(value.get("metadata", {}))
            if not isinstance(metadata, dict):
                raise ValueError("input metadata must be an object")
            artifact_ids: list[str] = []
            source_path: Path | None = None
            if value.get("artifact_id"):
                artifact = self.fleet.controller.artifacts.get(str(value["artifact_id"]))
                artifact_ids.append(artifact.id)
            elif value.get("path"):
                source_path = self._safe_source(str(value["path"]))
                if source_path.is_dir():
                    if kind not in INPUT_KINDS_WITH_DIRECTORIES:
                        raise ValueError(f"input kind {kind} does not accept a directory")
                    artifact_ids.append(self._directory_artifact(run_id, kind, source_path))
                else:
                    artifact = self.fleet.controller.artifacts.put_file(
                        request_id=run_id,
                        kind=kind,
                        source=source_path,
                        provenance={"type": "local_use_case_input", "parents": []},
                        retention=value.get("retention"),
                        metadata=metadata,
                    )
                    artifact_ids.append(artifact.id)
                    if (
                        kind == "pdf"
                        and bool(value.get("render_pages", True))
                        and not self._input_pdf_has_text({"artifact_id": artifact.id})
                    ):
                        artifact_ids.extend(
                            self._render_pdf_pages(run_id, source_path, artifact.id)
                        )
            elif "content" in value:
                artifact = self.fleet.controller.artifacts.put_text(
                    request_id=run_id,
                    kind=kind,
                    text=str(value["content"]),
                    provenance={"type": "inline_use_case_input", "parents": []},
                    retention=value.get("retention"),
                    metadata=metadata,
                )
                artifact_ids.append(artifact.id)
            elif "content_base64" in value:
                payload = base64.b64decode(str(value["content_base64"]), validate=True)
                maximum = int(
                    self.config.data.get("controller", {}).get(
                        "maximum_image_bytes", 20 * 1024 * 1024
                    )
                )
                if len(payload) > maximum:
                    raise ValueError("inline binary input exceeds the configured size limit")
                artifact = self.fleet.controller.artifacts.put_bytes(
                    request_id=run_id,
                    kind=kind,
                    payload=payload,
                    suffix=str(value.get("suffix", ".bin")),
                    media_type=str(value.get("media_type", "application/octet-stream")),
                    provenance={"type": "inline_use_case_input", "parents": []},
                    retention=value.get("retention"),
                    metadata=metadata,
                )
                artifact_ids.append(artifact.id)
                if (
                    kind == "pdf"
                    and bool(value.get("render_pages", True))
                    and not self._input_pdf_has_text({"artifact_id": artifact.id})
                ):
                    page_ids = self._render_pdf_pages(run_id, Path(artifact.path), artifact.id)
                    artifact_ids.extend(page_ids)
            else:
                raise ValueError(f"input {index} has no artifact, path, or content")
            evidence_ids: list[str] = []
            if kind == "pdf" and artifact_ids:
                evidence_ids.extend(self._pdf_evidence(run_id, artifact_ids[0]))
            for artifact_id in artifact_ids:
                evidence_ids.extend(self._text_evidence(run_id, artifact_id))
            if kind in VISUAL_KINDS or (kind == "pdf" and not evidence_ids):
                image_paths.extend(
                    Path(record.path)
                    for artifact_id in artifact_ids
                    if (record := self.fleet.controller.artifacts.get(artifact_id))
                    and record.media_type.startswith("image/")
                )
            ingested.append(
                {
                    "ordinal": index,
                    "kind": kind,
                    "artifact_ids": artifact_ids,
                    "evidence_ids": evidence_ids,
                    "metadata": metadata,
                }
            )
        return ingested, image_paths

    @staticmethod
    def _condition_matches(
        condition: str,
        *,
        has_visual: bool,
        has_audio: bool,
        has_transcript: bool,
        flags: set[str],
    ) -> bool:
        return {
            "always": True,
            "visual": has_visual,
            "failed_or_ambiguous": bool(flags & {"failed", "ambiguous"}),
            "ambiguous_or_high_risk": bool(flags & {"ambiguous", "high_risk"}),
            "exceptional": bool(flags & {"malformed", "visual", "ambiguous", "conflicting"}),
            "audio_without_transcript": has_audio and not has_transcript,
            "multi_speaker": "multi_speaker" in flags,
        }.get(condition, False)

    def _resolve_endpoint(self, stage: dict[str, Any]) -> EndpointDefinition | None:
        preferred = [str(value) for value in stage.get("preferred", [])]
        for endpoint_id in preferred:
            if self._endpoint_available(endpoint_id):
                return self.registry.get(endpoint_id)
        strict_preferred = bool(stage.get("strict_preferred", False))
        capability = str(stage.get("capability", ""))
        for endpoint_id, entry in self.fleet.entries.items():
            if entry.state in {FleetState.READY, FleetState.BUSY} and (
                (not strict_preferred and capability in entry.endpoint.capabilities)
                or endpoint_id == "mock-echo"
            ):
                return entry.endpoint
        return None

    def _continuation_parent(
        self, payload: dict[str, Any], template_id: str
    ) -> dict[str, Any] | None:
        parent_run_id = str(payload.get("parent_run_id", "")).strip()
        if not parent_run_id:
            return None
        if not parent_run_id.startswith("usecase-") or Path(parent_run_id).name != parent_run_id:
            raise ValueError("invalid parent use-case run identifier")
        try:
            parent = self.get(parent_run_id)
        except FileNotFoundError as exc:
            raise ValueError(f"unknown parent use-case run: {parent_run_id}") from exc
        if parent.get("template_id") != template_id:
            raise ValueError("follow-up must use the same task template as its parent")
        return parent

    def _continuation_inputs(
        self, run_id: str, parent: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], list[Path], str]:
        inherited: list[dict[str, Any]] = []
        images: list[Path] = []
        for value in parent.get("inputs", []):
            if not isinstance(value, dict):
                continue
            artifact_ids = [str(item) for item in value.get("artifact_ids", [])]
            evidence_ids = [str(item) for item in value.get("evidence_ids", [])]
            for artifact_id in [*artifact_ids, *evidence_ids]:
                self.fleet.controller.artifacts.get(artifact_id)
            inherited.append(
                {
                    "ordinal": len(inherited),
                    "kind": str(value.get("kind", "source")),
                    "artifact_ids": artifact_ids,
                    "evidence_ids": evidence_ids,
                    "metadata": deepcopy(value.get("metadata", {})),
                }
            )
            if value.get("kind") in VISUAL_KINDS or (
                value.get("kind") == "pdf" and not evidence_ids
            ):
                images.extend(
                    Path(record.path)
                    for artifact_id in artifact_ids
                    if (record := self.fleet.controller.artifacts.get(artifact_id))
                    and record.media_type.startswith("image/")
                )

        presentation = parent.get("presentation", {})
        deliverable = presentation.get("deliverable", {}) if isinstance(presentation, dict) else {}
        prior_text = str(deliverable.get("text", "")).strip()
        if not prior_text and deliverable:
            prior_text = json.dumps(deliverable, indent=2, sort_keys=True)
        if not prior_text:
            prior_text = "The previous task produced no textual deliverable."
        prior = self.fleet.controller.artifacts.put_text(
            request_id=run_id,
            kind="prior_result",
            text=prior_text,
            provenance={
                "type": "use_case_continuation",
                "parents": [str(value) for value in parent.get("outputs", [])],
            },
            metadata={
                "source_name": "Previous task result",
                "parent_run_id": str(parent["id"]),
                "parent_version_id": presentation.get("version", {}).get("id")
                if isinstance(presentation, dict)
                else None,
            },
        )
        inherited.append(
            {
                "ordinal": len(inherited),
                "kind": "prior_result",
                "artifact_ids": [prior.id],
                "evidence_ids": self._text_evidence(run_id, prior.id),
                "metadata": deepcopy(prior.metadata),
            }
        )
        if len(inherited) > self.maximum_inputs:
            raise ValueError(
                f"continued task exceeds the limit of {self.maximum_inputs} inherited inputs"
            )
        return inherited, images, prior.id

    def readiness(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Resolve request-specific required stages without mutating fleet state."""

        template_id = str(payload.get("template_id", ""))
        try:
            template = self.templates[template_id]
        except KeyError as exc:
            raise ValueError(f"unknown use-case template: {template_id}") from exc
        parent = self._continuation_parent(payload, template_id)
        values = payload.get("inputs", [])
        if not isinstance(values, list):
            raise ValueError("inputs must be a list")
        admitted = {str(value) for value in template.get("input_kinds", [])}
        kinds: list[str] = []
        for value in values:
            if not isinstance(value, dict):
                raise ValueError("each use-case input must be an object")
            kind = str(value.get("kind", ""))
            if kind not in admitted:
                raise ValueError(f"input kind is not admitted by this template: {kind}")
            kinds.append(kind)
        parent_inputs = [
            value
            for value in (parent.get("inputs", []) if parent else [])
            if isinstance(value, dict)
        ]
        parent_kinds = [str(value.get("kind", "")) for value in parent_inputs]
        flags = {str(value) for value in payload.get("flags", [])}
        if payload.get("high_risk"):
            flags.add("high_risk")
        has_visual = any(
            kind in VISUAL_KINDS
            or (kind == "pdf" and not self._input_pdf_has_text(value))
            for kind, value in zip(kinds, values, strict=True)
        ) or any(
            kind in VISUAL_KINDS
            or any(
                self.fleet.controller.artifacts.get(str(artifact_id)).media_type.startswith(
                    "image/"
                )
                for artifact_id in value.get("artifact_ids", [])
            )
            for kind, value in zip(parent_kinds, parent_inputs, strict=True)
        )
        all_kinds = [*parent_kinds, *kinds]
        has_audio = any(value in MEDIA_KINDS for value in all_kinds)
        has_transcript = "transcript" in all_kinds
        stages: list[dict[str, Any]] = []
        for stage_value in self._configured_stages(template, payload)[
            : self.maximum_model_stages
        ]:
            stage = deepcopy(stage_value)
            required = self._condition_matches(
                str(stage.get("condition", "always")),
                has_visual=has_visual,
                has_audio=has_audio,
                has_transcript=has_transcript,
                flags=flags,
            )
            resolved = self._resolve_endpoint(stage) if required else None
            candidate = resolved
            if candidate is None and required:
                for endpoint_id in stage.get("preferred", []):
                    entry = self.fleet.entries.get(str(endpoint_id))
                    if entry is not None:
                        candidate = entry.endpoint
                        break
            if candidate is None and required:
                capability = str(stage.get("capability", ""))
                strict_preferred = bool(stage.get("strict_preferred", False))
                candidate = next(
                    (
                        entry.endpoint
                        for endpoint_id, entry in self.fleet.entries.items()
                        if (not strict_preferred and capability in entry.endpoint.capabilities)
                        or endpoint_id == "mock-echo"
                    ),
                    None,
                )
            if not required:
                status = "optional"
            elif resolved is not None:
                status = "ready"
            elif candidate is not None:
                status = "needs_preparation"
            else:
                status = "unavailable"
            stages.append(
                {
                    "id": str(stage.get("id", "stage")),
                    "role": str(stage.get("role", "model work")),
                    "capability": str(stage.get("capability", "")),
                    "required": required,
                    "status": status,
                    "endpoint": candidate.id if candidate is not None else None,
                    **(
                        {"ocr_mode": str(stage["ocr_mode"])}
                        if stage.get("ocr_mode")
                        else {}
                    ),
                }
            )
        required_stages = [stage for stage in stages if stage["required"]]
        if any(stage["status"] == "unavailable" for stage in required_stages):
            state = "blocked"
        elif any(stage["status"] == "needs_preparation" for stage in required_stages):
            state = "needs_preparation"
        else:
            state = "ready"
        alternatives: list[dict[str, str]] = []
        if any(
            stage["required"]
            and stage["capability"] == "audio_transcription"
            and stage["status"] != "ready"
            for stage in stages
        ):
            alternatives.append(
                {
                    "action": "add_transcript",
                    "label": "Add a transcript",
                    "reason": "Transcription is not ready.",
                }
            )
        return {
            "schema": READINESS_SCHEMA,
            "template_id": template_id,
            "state": state,
            "required": required_stages,
            "optional": [stage for stage in stages if not stage["required"]],
            "blocking_reasons": [
                f"{stage['role']} is unavailable"
                for stage in required_stages
                if stage["status"] == "unavailable"
            ],
            "preparation": {
                "needed": state == "needs_preparation",
                "endpoints": [
                    stage["endpoint"]
                    for stage in required_stages
                    if stage["status"] == "needs_preparation" and stage["endpoint"]
                ],
            },
            "alternatives": alternatives,
            "continuation": {
                "parent_run_id": parent.get("id") if parent else None,
                "thread_id": (
                    parent.get("thread_id", parent.get("id")) if parent else None
                ),
            },
        }

    def _result_presentation(
        self,
        *,
        run_id: str,
        payload: dict[str, Any],
        inputs: list[dict[str, Any]],
        outputs: list[str],
        claims: list[dict[str, Any]],
        checks: list[dict[str, Any]],
        gaps: list[str],
    ) -> dict[str, Any]:
        sources: list[dict[str, Any]] = []
        source_by_artifact: dict[str, dict[str, Any]] = {}
        for value in inputs:
            for artifact_id in [
                *value.get("artifact_ids", []),
                *value.get("evidence_ids", []),
            ]:
                record = self.fleet.controller.artifacts.get(str(artifact_id))
                source = {
                    "id": record.id,
                    "name": str(record.metadata.get("source_name") or value["kind"]),
                    "kind": record.kind,
                    "media_type": record.media_type,
                    "page": record.metadata.get("page_number"),
                    "region": record.metadata.get("region"),
                    "line_start": record.metadata.get("line_start"),
                    "line_end": record.metadata.get("line_end"),
                    "artifact_id": record.id,
                }
                sources.append(source)
                source_by_artifact[record.id] = source
        answer = ""
        for artifact_id in reversed(outputs):
            try:
                record = self.fleet.controller.artifacts.get(str(artifact_id))
                if record.media_type.startswith("text/"):
                    answer = self.fleet.controller.artifacts.read_text(record.id)
                    break
            except (FileNotFoundError, ValueError):
                continue
        if not answer and claims:
            answer = "\n\n".join(str(claim["text"]) for claim in claims if claim["text"])
        if not answer and payload.get("plan_only"):
            answer = (
                "The requested workflow was checked and preserved as a plan. "
                "Models were not run."
            )
        citations: list[dict[str, Any]] = []
        for claim in claims:
            for reference in claim.get("evidence_references", []):
                citation_source: dict[str, Any] | None = source_by_artifact.get(
                    str(reference)
                )
                citations.append(
                    {
                        "claim_id": claim["id"],
                        "label": (
                            citation_source["name"]
                            if citation_source
                            else str(reference)
                        ),
                        "artifact_id": str(reference),
                        "page": claim.get("page")
                        or (citation_source or {}).get("page"),
                        "supported": bool(claim.get("supported")),
                    }
                )
        cited_artifacts = {str(value["artifact_id"]) for value in citations}
        for source in sources:
            artifact_id = str(source["artifact_id"])
            if answer and artifact_id in answer and artifact_id not in cited_artifacts:
                citations.append(
                    {
                        "claim_id": "$deliverable",
                        "label": source["name"],
                        "artifact_id": artifact_id,
                        "page": source.get("page"),
                        "supported": True,
                    }
                )
                cited_artifacts.add(artifact_id)
        intent = str(payload.get("output_intent", "answer"))
        deliverable: dict[str, Any] = {"type": intent, "text": answer}
        if intent == "table":
            fields = payload.get("document_fields", [])
            if not isinstance(fields, list):
                fields = []
            raw_rows = deepcopy(payload.get("extracted_rows", []))
            if not raw_rows and answer:
                candidate = answer.strip()
                if candidate.startswith("```"):
                    candidate = candidate.removeprefix("```json").removeprefix("```")
                    candidate = candidate.removesuffix("```").strip()
                try:
                    decoded = json.loads(candidate)
                    if isinstance(decoded, dict) and isinstance(decoded.get("rows"), list):
                        raw_rows = decoded["rows"]
                    elif isinstance(decoded, list):
                        raw_rows = decoded
                    elif isinstance(decoded, dict):
                        raw_rows = [decoded]
                except json.JSONDecodeError:
                    raw_rows = []
            field_ids = {
                str(field.get("id", field.get("label", "field")))
                for field in fields
                if isinstance(field, dict)
            }
            if raw_rows and all(
                isinstance(value, dict)
                and str(value.get("id", "")) in field_ids
                and "value" in value
                for value in raw_rows
            ):
                normalized_row: dict[str, Any] = {}
                citations_by_field: dict[str, Any] = {}
                for value in raw_rows:
                    field_id = str(value["id"])
                    normalized_row[field_id] = value.get("value")
                    if value.get("source"):
                        citations_by_field[field_id] = value["source"]
                if citations_by_field:
                    normalized_row["_citations"] = citations_by_field
                raw_rows = [normalized_row]
            deliverable = {
                "type": "table",
                "columns": [
                    {
                        "id": str(field.get("id", field.get("label", "field"))),
                        "label": str(field.get("label", field.get("id", "Field"))),
                        "value_type": str(field.get("type", "string")),
                        "required": bool(field.get("required", False)),
                    }
                    for field in fields
                    if isinstance(field, dict)
                ],
                "rows": [value for value in raw_rows if isinstance(value, dict)],
                "text": answer,
            }
        limitations: list[str] = []
        if not citations:
            limitations.append("No claim-level source references were recorded.")
        if gaps:
            limitations.append("Some required capabilities or stages did not complete.")
        return {
            "schema": PRESENTATION_SCHEMA,
            "version": {
                "id": f"{run_id}:v1",
                "number": 1,
                "parent_id": None,
                "immutable_original": True,
                "corrections": [],
            },
            "deliverable": deliverable,
            "sources": sources,
            "citations": citations,
            "checks": [{**deepcopy(check), "version_id": f"{run_id}:v1"} for check in checks],
            "limitations": limitations,
            "next_actions": ["inspect_sources", "run_again"],
        }

    @staticmethod
    def _grounding_text(value: Any) -> str:
        return "".join(character for character in str(value).casefold() if character.isalnum())

    def _result_checks(
        self,
        *,
        kind: str,
        presentation: dict[str, Any],
        inputs: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Apply conservative checks to the user-facing candidate, not just its envelope."""

        evidence_ids = list(
            dict.fromkeys(
                str(artifact_id)
                for value in inputs
                for artifact_id in value.get("evidence_ids", [])
            )
        )
        source_ids = {
            str(artifact_id)
            for value in inputs
            for artifact_id in [
                *value.get("artifact_ids", []),
                *value.get("evidence_ids", []),
            ]
        }
        citation_ids = {
            str(citation.get("artifact_id"))
            for citation in presentation.get("citations", [])
            if isinstance(citation, dict) and citation.get("artifact_id")
        }
        checks: list[dict[str, Any]] = []
        if evidence_ids:
            cited_sources = citation_ids & source_ids
            checks.append(
                {
                    "name": "source_grounding",
                    "passed": bool(cited_sources),
                    "failures": []
                    if cited_sources
                    else ["result does not cite any supplied source"],
                }
            )

            evidence_text = "\n".join(
                self.fleet.controller.artifacts.read_text(artifact_id)
                for artifact_id in evidence_ids
            )
            answer = str(presentation.get("deliverable", {}).get("text", ""))
            stopwords = {
                "about",
                "after",
                "also",
                "from",
                "have",
                "into",
                "only",
                "source",
                "that",
                "their",
                "there",
                "these",
                "this",
                "using",
                "were",
                "with",
            }
            evidence_terms = {
                term.casefold()
                for term in re.findall(r"[A-Za-z0-9][A-Za-z0-9_.:/+-]{3,}", evidence_text)
                if term.casefold() not in stopwords
            }
            answer_terms = {
                term.casefold()
                for term in re.findall(r"[A-Za-z0-9][A-Za-z0-9_.:/+-]{3,}", answer)
                if term.casefold() not in stopwords
            }
            minimum_overlap = min(2, len(evidence_terms))
            overlap = sorted(evidence_terms & answer_terms)
            grounded = bool(answer.strip()) and len(overlap) >= minimum_overlap
            checks.append(
                {
                    "name": "evidence_overlap",
                    "passed": grounded,
                    "failures": []
                    if grounded
                    else ["result does not contain enough verifiable source detail"],
                }
            )
            quoted_failures: list[str] = []
            for quoted in re.findall(r'["“]([^"”\n]{4,})["”]', answer):
                if "`" in quoted:
                    continue
                normalized_quote = self._grounding_text(quoted)
                if normalized_quote and normalized_quote not in self._grounding_text(evidence_text):
                    quoted_failures.append(
                        f"quoted text is not present in supplied evidence: {quoted[:80]}"
                    )
            checks.append(
                {
                    "name": "quoted_source_grounding",
                    "passed": not quoted_failures,
                    "failures": quoted_failures,
                }
            )

        if kind != "document":
            return checks

        deliverable = presentation.get("deliverable", {})
        columns = deliverable.get("columns", []) if isinstance(deliverable, dict) else []
        rows = deliverable.get("rows", []) if isinstance(deliverable, dict) else []
        field_ids = {
            str(column.get("id"))
            for column in columns
            if isinstance(column, dict) and column.get("id")
        }
        required_ids = {
            str(column.get("id"))
            for column in columns
            if isinstance(column, dict) and column.get("id") and column.get("required")
        }
        contract_failures: list[str] = []
        grounding_failures: list[str] = []
        if deliverable.get("type") == "table" and not rows:
            contract_failures.append("no structured rows were returned")
        evidence_text = "\n".join(
            self.fleet.controller.artifacts.read_text(artifact_id)
            for artifact_id in evidence_ids
        )
        normalized_evidence = self._grounding_text(evidence_text)
        for row_number, row in enumerate(rows, 1):
            if not isinstance(row, dict):
                contract_failures.append(f"row {row_number}: row is not an object")
                continue
            unknown = set(map(str, row)) - field_ids - {"_citations"}
            for field_id in sorted(unknown):
                contract_failures.append(f"row {row_number}: unknown field {field_id}")
            for field_id in sorted(required_ids):
                if row.get(field_id) in (None, ""):
                    contract_failures.append(f"row {row_number}: {field_id} is required")
            field_citations = row.get("_citations", {})
            if not isinstance(field_citations, dict):
                contract_failures.append(f"row {row_number}: _citations must be an object")
                field_citations = {}
            for field_id in sorted(field_ids):
                value = row.get(field_id)
                if value in (None, ""):
                    continue
                normalized_value = self._grounding_text(value)
                if normalized_value and normalized_value not in normalized_evidence:
                    grounding_failures.append(
                        f"row {row_number}: {field_id} is not found in source text"
                    )
                reference = str(field_citations.get(field_id, ""))
                if not reference or reference not in source_ids:
                    grounding_failures.append(
                        f"row {row_number}: {field_id} has no valid source reference"
                    )
        checks.extend(
            [
                {
                    "name": "result_contract",
                    "passed": not contract_failures,
                    "failures": contract_failures,
                },
                {
                    "name": "document_value_grounding",
                    "passed": not grounding_failures,
                    "failures": grounding_failures,
                },
            ]
        )
        return checks

    def _stage_prompt(
        self,
        prompt: str,
        stage: dict[str, Any],
        context: CoordinationContext,
    ) -> tuple[str, bool]:
        role = str(stage.get("role", "model work"))
        stage_id = str(stage.get("id", role))
        instructions = {
            "extraction": (
                "Extract only facts present in the evidence. Preserve exact values and cite the "
                "evidence chunk identifier beside each fact."
            ),
            "repair": (
                "Diagnose the supplied material directly. Do not infer unseen code. "
                "Return a complete proposed answer with evidence citations."
            ),
            "diagnosis": (
                "Rank causes using the supplied evidence, distinguish observations "
                "from hypotheses, and return reversible next steps with evidence citations. "
                "Do not invent literal commands, paths, flags, identifiers, or error messages."
            ),
            "interpretation": (
                "Use only the supplied document evidence. Do not invent missing values. "
                "Preserve exact source wording for extracted values and cite evidence identifiers."
            ),
            "summary": (
                "Summarize the supplied transcript evidence into decisions, owners, "
                "dates, risks, and actions. Do not invent absent details; "
                "cite evidence identifiers."
            ),
            "synthesis": (
                "Synthesize the evidence, explicitly separate agreement, disagreement, and limits, "
                "and cite evidence identifiers for every substantive claim."
            ),
            "critique": (
                "Audit the prior candidate against the supplied evidence. Begin with exactly "
                "VERDICT: PASS when every claim is supported, otherwise VERDICT: FAIL. "
                "Then list unsupported or missing claims concisely; do not introduce new facts."
            ),
            "validation": (
                "Validate the prior candidate against the supplied evidence. Begin with exactly "
                "VERDICT: PASS when every claim is supported, otherwise VERDICT: FAIL. "
                "Then list exact discrepancies and missing citations; do not introduce new facts."
            ),
            "reconciliation": (
                "Return a corrected complete replacement answer using the evidence and "
                "prior candidate. Retain evidence identifiers beside supported claims."
            ),
        }
        default_instruction = "Complete this bounded stage using only supplied evidence."
        instruction = instructions.get(role, instructions.get(stage_id, default_instruction))
        value = f"{prompt}\n\nStage role: {role}.\n{instruction}"
        handoff, truncated = context.model_handoff(maximum_characters=1600)
        return f"{value}\n\n{handoff}", truncated

    def _bounded_evidence(
        self,
        endpoint: EndpointDefinition,
        prompt: str,
        evidence_ids: tuple[str, ...],
    ) -> tuple[tuple[str, ...], bool]:
        input_token_budget = endpoint.context_size - endpoint.max_output_tokens
        character_limit = endpoint.max_input_characters or input_token_budget * 2
        remaining = max(0, character_limit - len(prompt) - 300)
        query_terms = {
            term.casefold()
            for term in re.findall(r"[A-Za-z0-9][A-Za-z0-9_.:/+-]{2,}", prompt)
        }
        candidates: list[tuple[int, int, str, int]] = []
        for ordinal, artifact_id in enumerate(evidence_ids):
            record = self.fleet.controller.artifacts.get(artifact_id)
            evidence_text = self.fleet.controller.artifacts.read_text(artifact_id)
            evidence_terms = {
                term.casefold()
                for term in re.findall(
                    r"[A-Za-z0-9][A-Za-z0-9_.:/+-]{2,}", evidence_text
                )
            }
            overhead = 100 + len(str(record.metadata).encode("utf-8"))
            required = len(evidence_text) + overhead
            candidates.append((len(query_terms & evidence_terms), ordinal, artifact_id, required))
        candidates.sort(key=lambda candidate: (-candidate[0], candidate[1]))
        selected: list[str] = []
        for _score, _ordinal, artifact_id, required in candidates:
            if required > remaining:
                continue
            selected.append(artifact_id)
            remaining -= required
        selected.sort(key=evidence_ids.index)
        return tuple(selected), len(selected) != len(evidence_ids)

    def _run_models(
        self,
        run_id: str,
        template: dict[str, Any],
        payload: dict[str, Any],
        inputs: list[dict[str, Any]],
        images: list[Path],
        timeline: list[dict[str, Any]],
        cancellation: threading.Event | None,
    ) -> tuple[list[dict[str, Any]], list[str], list[str]]:
        flags = {str(value) for value in payload.get("flags", [])}
        if str(template.get("kind")) == "batch":
            for record in payload.get("records", []):
                if not isinstance(record, dict):
                    flags.add("malformed")
                    continue
                for flag in ("visual", "ambiguous", "conflicting", "malformed"):
                    if record.get(flag):
                        flags.add(flag)
        has_audio = any(value["kind"] in MEDIA_KINDS for value in inputs)
        has_transcript = any(value["kind"] == "transcript" for value in inputs)
        stages = [
            deepcopy(value)
            for value in self._configured_stages(template, payload)
            if self._condition_matches(
                str(value.get("condition", "always")),
                has_visual=bool(images),
                has_audio=has_audio,
                has_transcript=has_transcript,
                flags=flags,
            )
        ][: self.maximum_model_stages]
        if images and not any(
            str(value.get("capability", ""))
            in {"visual_understanding", "document_ocr", "document_vision"}
            for value in stages
        ):
            stages.insert(
                0,
                {
                    "id": "visual_evidence",
                    "role": "visual_interpretation",
                    "capability": "visual_understanding",
                    "preferred": [
                        "gemma-e2b-vision",
                        "gemma4-12b-qat",
                    ],
                },
            )
            stages = stages[: self.maximum_model_stages]
        routes: list[dict[str, Any]] = []
        outputs: list[str] = []
        gaps: list[str] = []
        evidence = list(
            dict.fromkeys(
                artifact_id
                for value in inputs
                for artifact_id in [
                    *value.get("evidence_ids", []),
                    *value.get("artifact_ids", []),
                ]
                if self.fleet.controller.artifacts.get(artifact_id).kind == "evidence_chunk"
            )
        )
        execute = not bool(payload.get("plan_only", False))
        prompt = str(payload.get("prompt", template.get("description", ""))).strip()
        source_catalog: list[str] = []
        for value in inputs:
            for artifact_id in value.get("artifact_ids", []):
                record = self.fleet.controller.artifacts.get(str(artifact_id))
                source_name = record.metadata.get("source_name") or value.get("kind")
                location = (
                    f"; page {record.metadata['page_number']}"
                    if record.metadata.get("page_number") is not None
                    else ""
                )
                source_catalog.append(f"[{record.id}; {source_name}{location}]")
        if source_catalog:
            prompt += (
                "\n\nSaved source identifiers:\n"
                + "\n".join(source_catalog)
                + "\nCite the relevant saved source identifier beside supported claims."
            )
        if str(template.get("kind")) == "document" and payload.get("output_intent") == "table":
            fields = [
                {
                    "id": str(value.get("id", value.get("label", "field"))),
                    "type": str(value.get("type", "string")),
                    "required": bool(value.get("required", False)),
                }
                for value in payload.get("document_fields", [])
                if isinstance(value, dict)
            ]
            prompt += (
                "\n\nReturn only a JSON array with one row object per source document. "
                "Each row must use the field IDs below as its keys. "
                "Do not return id/value wrappers. "
                "Use null for missing values and preserve values exactly as written in the source. "
                "Add a _citations object mapping each populated field ID to its evidence chunk ID. "
                "Field contract: " + json.dumps(fields, sort_keys=True)
            )
        context = CoordinationContext(
            objective=prompt,
            trigger=(",".join(sorted(flags)) if flags else None),
            source_evidence=list(evidence),
        )
        for stage in stages:
            if cancellation is not None and cancellation.is_set():
                raise RequestFailedError("use-case run cancelled")
            endpoint = self._resolve_endpoint(stage)
            route: dict[str, Any] = {
                "stage": str(stage["id"]),
                "role": str(stage.get("role", "unspecified")),
                "capability": str(stage.get("capability", "")),
                "endpoint": endpoint.id if endpoint else None,
                "status": "planned" if endpoint else "unavailable",
                **(
                    {"ocr_mode": str(stage["ocr_mode"])}
                    if stage.get("ocr_mode")
                    else {}
                ),
            }
            if endpoint is None:
                gaps.append(str(stage.get("capability", stage["id"])))
                routes.append(route)
                continue
            if execute:
                try:
                    configured_ocr_prompt = endpoint.runtime.get("ocr_prompt")
                    base_prompt = (
                        str(configured_ocr_prompt)
                        if str(stage.get("role")) == "ocr"
                        and isinstance(configured_ocr_prompt, str)
                        and configured_ocr_prompt
                        else prompt
                    )
                    stage_prompt, context_truncated = self._stage_prompt(
                        base_prompt, stage, context
                    )
                    stage_evidence, truncated = self._bounded_evidence(
                        endpoint, stage_prompt, tuple(evidence)
                    )
                    route["context"] = {
                        "schema": context.schema,
                        "input_stage_references": [
                            value.answer_reference
                            for value in context.stages
                            if value.answer_reference
                        ],
                        "input_evidence_references": list(stage_evidence),
                        "handoff_truncated": context_truncated,
                    }
                    if evidence and not stage_evidence:
                        raise RequestFailedError(
                            f"source evidence does not fit the input budget for stage {stage['id']}"
                        )
                    if truncated:
                        route["evidence_truncated"] = True
                    result = self.fleet.submit(
                        endpoint_id=endpoint.id,
                        prompt=stage_prompt,
                        images=tuple(images) if "image" in endpoint.modalities else (),
                        evidence_references=stage_evidence,
                        cancellation=cancellation,
                        timeout_seconds=float(
                            self.config.data.get("controller", {}).get(
                                "request_timeout_seconds", 300
                            )
                        ),
                    )
                    route["status"] = result.envelope.status
                    route["request_id"] = result.envelope.request_id
                    if result.envelope.answer_reference:
                        outputs.append(result.envelope.answer_reference)
                        route["answer_reference"] = result.envelope.answer_reference
                        answer_text = self.fleet.controller.artifacts.read_text(
                            result.envelope.answer_reference
                        )
                        context.add_stage(
                            StageHandoff(
                                stage=str(stage["id"]),
                                role=str(stage.get("role", "unspecified")),
                                endpoint=endpoint.id,
                                answer_reference=result.envelope.answer_reference,
                                evidence_references=stage_evidence,
                                answer=answer_text,
                            )
                        )
                        if str(stage.get("role")) == "ocr" and answer_text.strip():
                            source_parents = [
                                str(artifact_id)
                                for value in inputs
                                for artifact_id in value.get("artifact_ids", [])
                            ]
                            extracted = self.fleet.controller.artifacts.put_text(
                                request_id=run_id,
                                kind="extracted_ocr_text",
                                text=answer_text,
                                provenance={
                                    "type": "model_ocr_extraction",
                                    "parents": [
                                        result.envelope.answer_reference,
                                        *source_parents,
                                    ],
                                },
                                metadata={
                                    "derived": True,
                                    "source_name": f"{endpoint.display_name} OCR output",
                                    "endpoint": endpoint.id,
                                },
                            )
                            ocr_evidence = self._text_evidence(run_id, extracted.id)
                            evidence.extend(ocr_evidence)
                            context.add_evidence(ocr_evidence)
                            inputs.append(
                                {
                                    "ordinal": len(inputs),
                                    "kind": "ocr_text",
                                    "artifact_ids": [extracted.id],
                                    "evidence_ids": ocr_evidence,
                                    "metadata": {"endpoint": endpoint.id},
                                }
                            )
                            route["ocr_evidence_references"] = ocr_evidence
                except Exception as exc:
                    route["status"] = "failed"
                    route["error_type"] = type(exc).__name__
                    gaps.append(f"stage_failed:{stage['id']}")
            routes.append(route)
            timeline.append(
                {
                    "sequence": len(timeline) + 1,
                    "class": "inferred",
                    "event": "model_stage",
                    "stage": route["stage"],
                    "endpoint": route["endpoint"],
                    "status": route["status"],
                }
            )
        return routes, outputs, gaps

    def _model_review_checks(self, routes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        last_reconciliation = max(
            (
                index
                for index, route in enumerate(routes)
                if route.get("role") == "reconciliation" and route.get("answer_reference")
            ),
            default=-1,
        )
        checks: list[dict[str, Any]] = []
        for index, route in enumerate(routes):
            if index < last_reconciliation or route.get("role") not in {
                "critique",
                "validation",
            }:
                continue
            answer_reference = route.get("answer_reference")
            if not answer_reference or route.get("endpoint") == "mock-echo":
                continue
            review = self.fleet.controller.artifacts.read_text(str(answer_reference)).strip()
            verdict = re.match(r"(?:\*\*)?VERDICT:\s*(PASS|FAIL)", review, re.IGNORECASE)
            passed = verdict is not None and verdict.group(1).upper() == "PASS"
            failure = (
                "model review rejected the candidate"
                if verdict is not None
                else "model review did not return a parseable verdict"
            )
            checks.append(
                {
                    "name": f"model_review:{route.get('stage')}",
                    "passed": passed,
                    "failures": [] if passed else [failure],
                    "review_artifact_id": str(answer_reference),
                }
            )
        return checks

    @staticmethod
    def _schema_errors(record: Any, schema: dict[str, Any]) -> list[str]:
        if not isinstance(record, dict):
            return ["record must be an object"]
        errors: list[str] = []
        for key in schema.get("required", []):
            if key not in record:
                errors.append(f"missing required field: {key}")
        type_map: dict[str, type[Any] | tuple[type[Any], ...]] = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "object": dict,
            "array": list,
        }
        for key, definition in schema.get("properties", {}).items():
            expected = type_map.get(definition.get("type"))
            if key in record and expected is not None and not isinstance(record[key], expected):
                errors.append(f"field {key} has the wrong type")
        return errors

    def _batch(
        self,
        run_id: str,
        payload: dict[str, Any],
    ) -> tuple[dict[str, Any], list[str], list[dict[str, Any]]]:
        records = payload.get("records", [])
        if not isinstance(records, list) or len(records) > self.maximum_batch_records:
            raise ValueError(
                f"records must be a list of at most {self.maximum_batch_records} items"
            )
        schema = payload.get("record_schema", {})
        if not isinstance(schema, dict):
            raise ValueError("record_schema must be an object")
        maximum_retries = min(2, max(0, int(payload.get("maximum_retries", 1))))
        accepted: list[dict[str, Any]] = []
        dead_letters: list[dict[str, Any]] = []
        traces: list[dict[str, Any]] = []
        seen: set[str] = set()
        escalated = 0
        started = monotonic()
        for ordinal, value in enumerate(records):
            if not isinstance(value, dict):
                value = {"value": value}
            record_id = str(value.get("id", f"record-{ordinal:06d}"))
            if record_id in seen:
                dead_letters.append({"id": record_id, "errors": ["duplicate record id"]})
                continue
            seen.add(record_id)
            data = value.get("data", value)
            errors = self._schema_errors(data, schema)
            exceptional = bool(
                value.get("visual") or value.get("ambiguous") or value.get("conflicting") or errors
            )
            escalated += int(exceptional)
            attempts = 1 + (maximum_retries if errors else 0)
            trace = {
                "id": record_id,
                "ordinal": ordinal,
                "attempts": attempts,
                "escalated": exceptional,
                "status": "dead_letter" if errors else "accepted",
                "errors": errors,
            }
            traces.append(trace)
            if errors:
                dead_letters.append({"id": record_id, "data": data, "errors": errors})
            else:
                accepted.append({"id": record_id, "data": data})
        duration = max(monotonic() - started, 0.000001)
        outputs: list[str] = []
        for kind, values in (("accepted_records", accepted), ("dead_letter_records", dead_letters)):
            artifact = self.fleet.controller.artifacts.put_text(
                request_id=run_id,
                kind=kind,
                text=json.dumps(values, indent=2, sort_keys=True),
                provenance={"type": "batch_processing", "parents": []},
                metadata={"record_count": len(values)},
            )
            outputs.append(artifact.id)
        metrics = {
            "total": len(records),
            "accepted": len(accepted),
            "validation_failed": len(dead_letters),
            "retried": sum(trace["attempts"] - 1 for trace in traces),
            "escalated": escalated,
            "dead_letter": len(dead_letters),
            "throughput_records_per_second": round(len(records) / duration, 2),
        }
        return metrics, outputs, traces

    def _research_retrieval(
        self,
        run_id: str,
        template: dict[str, Any],
        payload: dict[str, Any],
        inputs: list[dict[str, Any]],
    ) -> tuple[dict[str, Any] | None, list[str]]:
        if not bool(payload.get("index_sources", False)):
            return None, []
        endpoint_id = str(
            payload.get(
                "retrieval_endpoint",
                template.get(
                    "retrieval_endpoint",
                    self.config.data.get("retrieval", {}).get(
                        "default_endpoint", "harrier-0.6b"
                    ),
                ),
            )
        )
        backend, index = create_vector_index(self.config, self.registry, endpoint_id)
        ingested: list[dict[str, Any]] = []
        try:
            backend.start()
            for value in inputs:
                for artifact_id in value.get("artifact_ids", []):
                    artifact = self.fleet.controller.artifacts.get(artifact_id)
                    if not (
                        artifact.media_type.startswith("text/")
                        or Path(artifact.path).suffix.casefold() in TEXT_SUFFIXES
                    ):
                        continue
                    result = index.ingest(Path(artifact.path))
                    ingested.append(
                        {
                            "document_artifact_id": result["document_artifact_id"],
                            "chunks_added": result["chunks_added"],
                        }
                    )
            query = str(payload.get("query", payload.get("prompt", ""))).strip()
            if not query:
                raise ValueError("research indexing requires a query or prompt")
            search_result = index.search(
                query, top_k=min(20, max(1, int(payload.get("top_k", 5))))
            )
        finally:
            backend.stop()
        evidence_ids = list(search_result.evidence_references)
        return (
            {
                "schema": "sparse-network-use-case-retrieval.v1",
                "index_id": search_result.index_id,
                "embedding_endpoint": search_result.embedding_endpoint,
                "embedding_revision": search_result.embedding_revision,
                "query": query,
                "ingested": ingested,
                "hits": [
                    {
                        "artifact_id": hit.artifact_id,
                        "document_artifact_id": hit.document_artifact_id,
                        "source_name": hit.source_name,
                        "line_start": hit.line_start,
                        "line_end": hit.line_end,
                        "semantic_score": hit.semantic_score,
                        "rerank_score": hit.rerank_score,
                    }
                    for hit in search_result.hits
                ],
            },
            evidence_ids,
        )

    def _run_tools(
        self,
        run_id: str,
        template: dict[str, Any],
        payload: dict[str, Any],
        permissions: set[str],
        timeline: list[dict[str, Any]],
        cancellation: threading.Event | None,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        requested = payload.get("tools", [])
        if not isinstance(requested, list):
            raise ValueError("tools must be a list")
        admitted = {str(value) for value in template.get("allowed_tools", [])}
        working = self.config.root
        if payload.get("working_directory"):
            candidate = self._safe_source(str(payload["working_directory"]))
            if not candidate.is_dir():
                raise ValueError("working_directory must be a directory")
            working = candidate
        results: list[dict[str, Any]] = []
        outputs: list[str] = []
        diagnostic_completed = False
        for requested_id in requested:
            if cancellation is not None and cancellation.is_set():
                raise RequestFailedError("use-case run cancelled")
            tool_id = str(requested_id)
            if tool_id not in admitted:
                raise PermissionError(f"tool is not admitted by this template: {tool_id}")
            tool = self.tools[tool_id]
            mode = str(tool["mode"])
            required_permission = "tool:diagnose" if mode == "read_only" else "tool:remediate"
            authorized = required_permission in permissions
            if mode == "state_changing":
                authorized = authorized and (
                    payload.get("confirmation") == "AUTHORIZE REMEDIATION"
                    and bool(str(payload.get("rollback_information", "")).strip())
                    and diagnostic_completed
                )
            self.fleet.event_log.emit(
                event="tool_event",
                endpoint="controller",
                request_id=run_id,
                details={"tool": tool_id, "state": "authorized" if authorized else "proposed"},
            )
            if not authorized:
                results.append(
                    {
                        "tool": tool_id,
                        "mode": mode,
                        "status": "blocked",
                        "required_permission": required_permission,
                        "reason": (
                            "read-only diagnostics must complete first"
                            if mode == "state_changing" and not diagnostic_completed
                            else "authorization policy not satisfied"
                        ),
                    }
                )
                timeline.append(
                    {
                        "sequence": len(timeline) + 1,
                        "class": "proposed",
                        "event": "tool_blocked",
                        "tool": tool_id,
                    }
                )
                continue
            argv = [
                sys.executable if str(value) == "{python}" else str(value) for value in tool["argv"]
            ]
            self.fleet.event_log.emit(
                event="tool_event",
                endpoint="controller",
                request_id=run_id,
                details={"tool": tool_id, "state": "running"},
            )
            started = monotonic()
            try:
                completed = subprocess.run(
                    argv,
                    cwd=working,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=float(tool.get("timeout_seconds", 120)),
                )
                status = "completed" if completed.returncode == 0 else "failed"
                output = (completed.stdout + completed.stderr)[-100_000:]
                artifact = self.fleet.controller.artifacts.put_text(
                    request_id=run_id,
                    kind="tool_result",
                    text=output,
                    provenance={"type": "allowlisted_tool", "parents": []},
                    metadata={
                        "tool": tool_id,
                        "mode": mode,
                        "returncode": completed.returncode,
                    },
                )
                outputs.append(artifact.id)
                result = {
                    "tool": tool_id,
                    "mode": mode,
                    "status": status,
                    "returncode": completed.returncode,
                    "elapsed_ms": round((monotonic() - started) * 1000),
                    "artifact_id": artifact.id,
                }
            except subprocess.TimeoutExpired:
                result = {"tool": tool_id, "mode": mode, "status": "failed_timeout"}
            results.append(result)
            if mode == "read_only" and result["status"] == "completed":
                diagnostic_completed = True
            self.fleet.event_log.emit(
                event="tool_event",
                endpoint="controller",
                request_id=run_id,
                details={"tool": tool_id, "state": result["status"]},
            )
            timeline.append(
                {
                    "sequence": len(timeline) + 1,
                    "class": "action",
                    "event": "tool_result",
                    "tool": tool_id,
                    "status": result["status"],
                }
            )
        return results, outputs

    def _validate_claims(
        self,
        values: Any,
        known_artifacts: set[str],
    ) -> tuple[list[dict[str, Any]], list[str]]:
        if not isinstance(values, list):
            raise ValueError("claims must be a list")
        claims: list[dict[str, Any]] = []
        unsupported: list[str] = []
        for ordinal, raw in enumerate(values):
            if not isinstance(raw, dict):
                raise ValueError("each claim must be an object")
            claim_id = str(raw.get("id", f"claim-{ordinal + 1}"))
            evidence = [str(value) for value in raw.get("evidence_references", [])]
            supported = bool(evidence) and set(evidence).issubset(known_artifacts)
            classification = str(raw.get("class", "inferred"))
            if classification not in FACT_CLASSES:
                raise ValueError(f"invalid claim class: {classification}")
            claims.append(
                {
                    "id": claim_id,
                    "class": classification,
                    "text": str(raw.get("text", "")),
                    "evidence_references": evidence,
                    "page": raw.get("page"),
                    "timestamp_start": raw.get("timestamp_start"),
                    "timestamp_end": raw.get("timestamp_end"),
                    "supported": supported,
                }
            )
            if not supported:
                unsupported.append(claim_id)
        return claims, unsupported

    def _meeting_outputs(
        self,
        run_id: str,
        payload: dict[str, Any],
        parents: list[str],
    ) -> list[str]:
        outputs: list[str] = []
        segments = payload.get("transcript_segments", [])
        if segments:
            if not isinstance(segments, list) or any(
                not isinstance(value, dict)
                or not isinstance(value.get("start"), (int, float))
                or not isinstance(value.get("end"), (int, float))
                or float(value["end"]) < float(value["start"])
                for value in segments
            ):
                raise ValueError("transcript segments need ordered numeric timestamps")
            transcript = self.fleet.controller.artifacts.put_text(
                request_id=run_id,
                kind="timestamped_transcript",
                text=json.dumps(segments, indent=2),
                provenance={"type": "transcription", "parents": parents},
                metadata={"segment_count": len(segments), "immutable_original": True},
            )
            outputs.append(transcript.id)
            corrections = payload.get("speaker_corrections", [])
            if corrections:
                correction = self.fleet.controller.artifacts.put_text(
                    request_id=run_id,
                    kind="speaker_correction",
                    text=json.dumps(corrections, indent=2),
                    provenance={"type": "human_correction", "parents": [transcript.id]},
                    metadata={"rewrites_original": False},
                )
                outputs.append(correction.id)
        return outputs

    def run(
        self,
        payload: dict[str, Any],
        *,
        actor: str,
        permissions: set[str],
        run_id: str | None = None,
        cancellation: threading.Event | None = None,
        progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        template_id = str(payload.get("template_id", ""))
        try:
            template = self.templates[template_id]
        except KeyError as exc:
            raise ValueError(f"unknown use-case template: {template_id}") from exc
        parent = self._continuation_parent(payload, template_id)
        run_id = run_id or f"usecase-{uuid4()}"
        if not run_id.startswith("usecase-") or Path(run_id).name != run_id:
            raise ValueError("invalid use-case run identifier")
        created_at = datetime.now(UTC).isoformat()
        timeline: list[dict[str, Any]] = [
            {
                "sequence": 1,
                "class": "observed",
                "event": "run_created",
                "actor": actor,
                "timestamp": created_at,
            }
        ]
        def report(stage_id: str, label: str, **counters: int) -> None:
            if progress is not None:
                progress({"stage_id": stage_id, "label": label, **counters})

        raw_inputs = payload.get("inputs", [])
        report(
            "reading_sources",
            "Reading sources",
            current=0,
            total=len(raw_inputs) if isinstance(raw_inputs, list) else 0,
        )
        inputs, images = self._ingest_inputs(run_id, template, raw_inputs)
        prior_result_id: str | None = None
        if parent is not None:
            inherited, inherited_images, prior_result_id = self._continuation_inputs(
                run_id, parent
            )
            for value in inputs:
                value["ordinal"] = len(inherited) + int(value.get("ordinal", 0))
            inputs = [*inherited, *inputs]
            images = [*inherited_images, *images]
            timeline.append(
                {
                    "sequence": len(timeline) + 1,
                    "class": "observed",
                    "event": "continuation_context_attached",
                    "parent_run_id": parent["id"],
                    "prior_result_artifact_id": prior_result_id,
                }
            )
        report("reading_sources", "Reading sources", current=len(inputs), total=len(inputs))
        if cancellation is not None and cancellation.is_set():
            raise RequestFailedError("use-case run cancelled")
        input_ids = [item for value in inputs for item in value["artifact_ids"]]
        timeline.append(
            {
                "sequence": len(timeline) + 1,
                "class": "observed",
                "event": "inputs_ingested",
                "artifact_ids": input_ids,
            }
        )
        retrieval: dict[str, Any] | None = None
        if str(template.get("kind")) == "research":
            retrieval, retrieval_evidence = self._research_retrieval(
                run_id, template, payload, inputs
            )
            if retrieval_evidence:
                inputs.append(
                    {
                        "ordinal": len(inputs),
                        "kind": "retrieval_result",
                        "artifact_ids": retrieval_evidence,
                        "metadata": {"index_id": retrieval["index_id"] if retrieval else None},
                    }
                )
                timeline.append(
                    {
                        "sequence": len(timeline) + 1,
                        "class": "observed",
                        "event": "retrieval_completed",
                        "index_id": retrieval["index_id"] if retrieval else None,
                        "evidence_references": retrieval_evidence,
                    }
                )
                input_ids = [item for value in inputs for item in value.get("artifact_ids", [])]
        trigger = str(payload.get("escalation_trigger", ""))
        frozen = {str(value) for value in template.get("frozen_escalation_triggers", [])}
        if trigger and trigger not in frozen:
            raise PermissionError(f"escalation trigger is not frozen for {template_id}: {trigger}")
        report("extracting_fields", "Extracting fields")
        routes, model_outputs, gaps = self._run_models(
            run_id, template, payload, inputs, images, timeline, cancellation
        )
        tools, tool_outputs = self._run_tools(
            run_id, template, payload, permissions, timeline, cancellation
        )
        outputs = [*model_outputs, *tool_outputs]
        metrics: dict[str, Any] = {}
        record_traces: list[dict[str, Any]] = []
        batch_outputs: list[str] = []
        meeting_outputs: list[str] = []
        if str(template.get("kind")) == "batch":
            metrics, batch_outputs, record_traces = self._batch(run_id, payload)
            outputs.extend(batch_outputs)
        if str(template.get("kind")) == "meeting":
            meeting_outputs = self._meeting_outputs(run_id, payload, input_ids)
            outputs.extend(meeting_outputs)
        if str(template.get("kind")) == "experiment":
            experiment_metrics = payload.get("metrics", {})
            if not isinstance(experiment_metrics, dict):
                raise ValueError("experiment metrics must be an object")
            metrics = deepcopy(experiment_metrics)
        known_artifacts = {
            str(artifact_id)
            for value in inputs
            for artifact_id in [
                *value.get("artifact_ids", []),
                *value.get("evidence_ids", []),
            ]
        } | set(outputs)
        raw_claims = deepcopy(payload.get("claims", []))
        if isinstance(raw_claims, list):
            for raw_claim in raw_claims:
                if not isinstance(raw_claim, dict):
                    continue
                references = raw_claim.get("evidence_references", [])
                if not isinstance(references, list):
                    continue
                resolved: list[str] = []
                for reference in references:
                    text = str(reference)
                    if text.startswith("input:") and text[6:].isdigit():
                        ordinal = int(text[6:])
                        if ordinal < len(inputs) and inputs[ordinal]["artifact_ids"]:
                            text = str(inputs[ordinal]["artifact_ids"][0])
                    elif text.startswith("output:") and text[7:].isdigit():
                        ordinal = int(text[7:])
                        if ordinal < len(outputs):
                            text = outputs[ordinal]
                    resolved.append(text)
                raw_claim["evidence_references"] = resolved
        report("checking_references", "Checking references")
        claims, unsupported = self._validate_claims(raw_claims, known_artifacts)
        high_risk = bool(payload.get("high_risk", False))
        blocked_tools = [value["tool"] for value in tools if value["status"] == "blocked"]
        failed_tools = [
            value["tool"] for value in tools if str(value["status"]).startswith("failed")
        ]
        checks = [
            {"name": "supported_claims", "passed": not unsupported, "failures": unsupported},
            {"name": "required_capabilities", "passed": not gaps, "failures": gaps},
            {"name": "tool_policy", "passed": not blocked_tools, "failures": blocked_tools},
            {"name": "tool_results", "passed": not failed_tools, "failures": failed_tools},
            {
                "name": "human_review",
                "passed": not high_risk or bool(payload.get("human_review_required", False)),
                "failures": ["human review is required"]
                if high_risk and not payload.get("human_review_required", False)
                else [],
            },
        ]
        checks.extend(self._model_review_checks(routes))
        excluded_candidate_roles = {"critique", "validation", "transcription", "diarization"}
        candidate_outputs = [
            str(route["answer_reference"])
            for route in routes
            if route.get("answer_reference")
            and str(route.get("role")) not in excluded_candidate_roles
        ]
        kind = str(template.get("kind", ""))
        if kind == "batch" and batch_outputs:
            selected = 0 if int(metrics.get("accepted", 0)) else 1
            presentation_outputs = [batch_outputs[selected]]
        else:
            presentation_outputs = candidate_outputs or model_outputs or tool_outputs
        report("preparing_output", "Preparing output")
        presentation = self._result_presentation(
            run_id=run_id,
            payload=payload,
            inputs=inputs,
            outputs=presentation_outputs,
            claims=claims,
            checks=checks,
            gaps=gaps,
        )
        if any(route.get("evidence_truncated") for route in routes):
            presentation["limitations"].append(
                "Only the highest-relevance source chunks that fit the model context were used."
            )
        if not payload.get("plan_only"):
            checks.extend(
                self._result_checks(kind=kind, presentation=presentation, inputs=inputs)
            )
        presentation["checks"] = [
            {**deepcopy(check), "version_id": f"{run_id}:v1"} for check in checks
        ]
        accepted = all(bool(value["passed"]) for value in checks)
        state = "accepted" if accepted else "rejected"
        if high_risk and payload.get("human_review_required", False):
            state = "needs_human_review"
        manifest: dict[str, Any] = {
            "schema": RUN_SCHEMA,
            "id": run_id,
            "template_id": template_id,
            "template_version": self.version,
            "kind": kind,
            "title": str(payload.get("title", template.get("title", template_id))),
            "prompt": str(payload.get("prompt", template.get("description", ""))),
            "actor": actor,
            "created_at": created_at,
            "completed_at": datetime.now(UTC).isoformat(),
            "state": state,
            "production_effect": False,
            "inputs": inputs,
            "routes": routes,
            "tools": tools,
            "claims": claims,
            "timeline": timeline,
            "record_traces": record_traces,
            "outputs": outputs,
            "metrics": metrics,
            "escalation": {
                "trigger": trigger or None,
                "frozen": not trigger or trigger in frozen,
                "large_model_executed": False,
            },
            "controls": {
                "language": payload.get("language"),
                "hotwords": payload.get("hotwords", []),
                "ocr_mode": self._ocr_mode(template, payload),
                "retention_policy": payload.get("retention_policy", "artifact_default"),
                "seed": int(payload.get("seed", 0)),
            },
            "verification": {"accepted": accepted, "checks": checks},
            "presentation": presentation,
            "parent_run_id": parent.get("id") if parent else None,
            "thread_id": (
                str(parent.get("thread_id") or parent["id"]) if parent else run_id
            ),
            "turn_index": int(parent.get("turn_index", 1)) + 1 if parent else 1,
        }
        if retrieval is not None:
            manifest["retrieval"] = retrieval
        if str(template.get("kind")) == "experiment":
            manifest["experiment"] = {
                "dataset_revision": payload.get("dataset_revision"),
                "prompt_revision": payload.get("prompt_revision"),
                "settings": deepcopy(payload.get("settings", {})),
                "seed": int(payload.get("seed", 0)),
                "baselines": deepcopy(payload.get("baselines", [])),
                "endpoint_utilization": deepcopy(payload.get("endpoint_utilization", {})),
                "unnecessary_escalations": int(payload.get("unnecessary_escalations", 0)),
                "scheduling_modes": deepcopy(
                    payload.get("scheduling_modes", ["resident", "load_on_demand"])
                ),
                "repetition_regressions": deepcopy(payload.get("repetition_regressions", [])),
            }
        self.runs_root.mkdir(parents=True, exist_ok=True)
        destination = self.runs_root / f"{run_id}.json"
        destination.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        self.fleet.event_log.emit(
            event="use_case_completed",
            endpoint="controller",
            request_id=run_id,
            details={
                "template_id": template_id,
                "state": state,
                "input_count": len(inputs),
                "output_count": len(outputs),
            },
        )
        return manifest

    def get(self, run_id: str) -> dict[str, Any]:
        if Path(run_id).name != run_id:
            raise FileNotFoundError(run_id)
        path = (self.runs_root / f"{run_id.removesuffix('.json')}.json").resolve(strict=True)
        if not path.is_relative_to(self.runs_root.resolve(strict=False)):
            raise FileNotFoundError(run_id)
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or value.get("schema") != RUN_SCHEMA:
            raise ValueError("invalid use-case run manifest")
        versions = self.runs_root / "versions"
        candidates = sorted(
            versions.glob(f"{run_id}-v*.json"),
            key=lambda candidate: candidate.stat().st_mtime_ns,
            reverse=True,
        ) if versions.exists() else []
        if candidates:
            reviewed = json.loads(candidates[0].read_text(encoding="utf-8"))
            if isinstance(reviewed, dict) and reviewed.get("schema") == RUN_SCHEMA:
                return reviewed
        return value

    def correct_result(
        self,
        run_id: str,
        payload: dict[str, Any],
        *,
        actor: str,
    ) -> dict[str, Any]:
        current = self.get(run_id)
        presentation = current.get("presentation")
        if not isinstance(presentation, dict):
            raise ValueError("run has no reviewable result")
        version = presentation.get("version", {})
        if payload.get("version_id") != version.get("id"):
            raise ValueError("result version changed; reload before saving corrections")
        raw_changes = payload.get("changes", [])
        if not isinstance(raw_changes, list) or not raw_changes:
            raise ValueError("corrections require a non-empty changes list")
        reviewed = deepcopy(current)
        reviewed_presentation = reviewed["presentation"]
        deliverable = reviewed_presentation.get("deliverable", {})
        corrections: list[dict[str, Any]] = []
        created_at = datetime.now(UTC).isoformat()
        for raw in raw_changes:
            if not isinstance(raw, dict):
                raise ValueError("each correction must be an object")
            field_id = str(raw.get("field_id", ""))
            row_index = int(raw.get("row_index", 0))
            replacement = raw.get("value")
            if field_id == "$text":
                original = deliverable.get("text", "")
                deliverable["text"] = str(replacement)
            elif deliverable.get("type") == "table":
                rows = deliverable.get("rows", [])
                columns = {
                    str(column.get("id"))
                    for column in deliverable.get("columns", [])
                    if isinstance(column, dict)
                }
                if field_id not in columns or not 0 <= row_index < len(rows):
                    raise ValueError("correction references an unknown table cell")
                if not isinstance(rows[row_index], dict):
                    raise ValueError("result table row is invalid")
                original = rows[row_index].get(field_id)
                rows[row_index][field_id] = replacement
            else:
                raise ValueError("correction references an unsupported result field")
            corrections.append(
                {
                    "field_id": field_id,
                    "row_index": row_index if field_id != "$text" else None,
                    "original": original,
                    "value": replacement,
                    "source_reference": raw.get("source_reference"),
                    "actor": actor,
                    "created_at": created_at,
                }
            )
        number = int(version.get("number", 1)) + 1
        version_id = f"{run_id}:v{number}"
        reviewed_presentation["version"] = {
            "id": version_id,
            "number": number,
            "parent_id": version.get("id"),
            "immutable_original": False,
            "corrections": corrections,
        }
        stale_checks = []
        for check in reviewed_presentation.get("checks", []):
            stale_checks.append(
                {
                    **deepcopy(check),
                    "original_passed": check.get("passed"),
                    "passed": False,
                    "status": "stale",
                    "version_id": version_id,
                }
            )
        references_resolve = all(
            self.fleet.controller.artifacts.get(str(citation["artifact_id"]))
            for citation in reviewed_presentation.get("citations", [])
            if citation.get("artifact_id")
        )
        rerun_checks = [
            {
                "name": "source_references",
                "passed": references_resolve,
                "failures": [] if references_resolve else ["source reference missing"],
                "status": "rerun",
                "version_id": version_id,
            }
        ]
        if deliverable.get("type") == "table":
            required = [
                str(column["id"])
                for column in deliverable.get("columns", [])
                if isinstance(column, dict) and column.get("required")
            ]
            missing = [
                f"row {index + 1}: {field}"
                for index, row in enumerate(deliverable.get("rows", []))
                for field in required
                if not isinstance(row, dict) or row.get(field) in {None, ""}
            ]
            rerun_checks.append(
                {
                    "name": "required_fields",
                    "passed": not missing,
                    "failures": missing,
                    "status": "rerun",
                    "version_id": version_id,
                }
            )
        reviewed_presentation["checks"] = [*stale_checks, *rerun_checks]
        reviewed["reviewed_at"] = created_at
        reviewed["reviewed_by"] = actor
        versions = self.runs_root / "versions"
        versions.mkdir(parents=True, exist_ok=True)
        destination = versions / f"{run_id}-v{number}.json"
        if destination.exists():
            raise FileExistsError(f"result version already exists: {version_id}")
        destination.write_text(
            json.dumps(reviewed, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self.fleet.event_log.emit(
            event="use_case_result_corrected",
            endpoint="controller",
            request_id=run_id,
            details={"version_id": version_id, "actor": actor},
        )
        return reviewed

    @staticmethod
    def _safe_csv_value(value: Any) -> str:
        text = "" if value is None else str(value)
        return f"'{text}" if text.startswith(("=", "+", "-", "@")) else text

    def export_result(
        self,
        run_id: str,
        payload: dict[str, Any],
        *,
        actor: str,
    ) -> dict[str, Any]:
        run = self.get(run_id)
        presentation = run.get("presentation", {})
        version = presentation.get("version", {})
        requested_version = payload.get("version_id")
        if requested_version and requested_version != version.get("id"):
            raise ValueError("export version does not match the current reviewed result")
        export_format = str(payload.get("format", "report"))
        deliverable = presentation.get("deliverable", {})
        if export_format == "csv":
            if deliverable.get("type") != "table":
                raise ValueError("CSV export requires a table result")
            columns = deliverable.get("columns", [])
            buffer = io.StringIO(newline="")
            writer = csv.writer(buffer, lineterminator="\n")
            writer.writerow([str(column.get("label", "Field")) for column in columns])
            for row in deliverable.get("rows", []):
                writer.writerow(
                    [
                        self._safe_csv_value(
                            row.get(str(column.get("id"))) if isinstance(row, dict) else None
                        )
                        for column in columns
                    ]
                )
            text = buffer.getvalue()
            kind = "reviewed_csv"
            suffix = ".csv"
            media_type = "text/csv; charset=utf-8"
        elif export_format == "report":
            citations = presentation.get("citations", [])
            limitations = presentation.get("limitations", [])
            text = (
                f"# {run.get('title', 'Sparse result')}\n\n"
                f"Version: {version.get('id', 'unknown')}\n\n"
                f"{deliverable.get('text', '')}\n\n"
                "## Sources\n\n"
                + "\n".join(
                    f"- {citation.get('label', citation.get('artifact_id'))}"
                    f"{f' — page {citation.get('page')}' if citation.get('page') else ''}"
                    for citation in citations
                )
                + "\n\n## Limitations\n\n"
                + "\n".join(f"- {value}" for value in limitations)
                + "\n"
            )
            kind = "reviewed_report"
            suffix = ".md"
            media_type = "text/markdown; charset=utf-8"
        else:
            raise ValueError("export format must be csv or report")
        record = self.fleet.controller.artifacts.put_bytes(
            request_id=run_id,
            kind=kind,
            payload=text.encode("utf-8-sig" if export_format == "csv" else "utf-8"),
            suffix=suffix,
            media_type=media_type,
            provenance={
                "type": "reviewed_result_export",
                "parents": [str(value) for value in run.get("outputs", [])],
            },
            metadata={
                "version_id": version.get("id"),
                "exported_by": actor,
                "source_name": f"{run_id}-{version.get('number', 1)}{suffix}",
            },
        )
        return {
            "schema": "sparse-network-result-export.v1",
            "version_id": version.get("id"),
            "format": export_format,
            "artifact": record.to_dict(),
        }

    def recent(self, limit: int = 50) -> list[dict[str, Any]]:
        if not self.runs_root.exists():
            return []
        values: list[dict[str, Any]] = []
        for path in sorted(
            self.runs_root.glob("usecase-*.json"),
            key=lambda value: value.stat().st_mtime_ns,
            reverse=True,
        )[:limit]:
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(value, dict):
                try:
                    values.append(self.get(str(value["id"])))
                except (KeyError, OSError, ValueError):
                    values.append(value)
        return values

    def replay_batch(
        self,
        run_id: str,
        payload: dict[str, Any],
        *,
        actor: str,
        permissions: set[str],
    ) -> dict[str, Any]:
        original = self.get(run_id)
        if original.get("kind") != "batch":
            raise ValueError("only batch runs support replay")
        if payload.get("confirmation") != f"REPLAY {run_id}":
            raise PermissionError(f"batch replay requires confirmation: REPLAY {run_id}")
        dead_letter = next(
            (
                artifact_id
                for artifact_id in original.get("outputs", [])
                if self.fleet.controller.artifacts.get(artifact_id).kind == "dead_letter_records"
            ),
            None,
        )
        records = (
            json.loads(self.fleet.controller.artifacts.read_text(dead_letter))
            if dead_letter
            else []
        )
        accepted_ids = {
            trace["id"]
            for trace in original.get("record_traces", [])
            if trace.get("status") == "accepted"
        }
        replay_records = [value for value in records if value.get("id") not in accepted_ids]
        return self.run(
            {
                **payload,
                "template_id": original["template_id"],
                "title": f"Replay of {run_id}",
                "records": replay_records,
                "inputs": [],
                "plan_only": True,
            },
            actor=actor,
            permissions=permissions,
        )

    def admission_decision(
        self,
        run_id: str,
        payload: dict[str, Any],
        *,
        actor: str,
    ) -> dict[str, Any]:
        run = self.get(run_id)
        if run.get("kind") != "experiment":
            raise ValueError("admission decisions require an experiment run")
        decision = str(payload.get("decision", ""))
        if decision not in {"admit", "reject"}:
            raise ValueError("decision must be admit or reject")
        if payload.get("confirmation") != f"{decision.upper()} {run_id}":
            raise PermissionError(
                f"admission decision requires confirmation: {decision.upper()} {run_id}"
            )
        reason = str(payload.get("reason", "")).strip()
        if not reason:
            raise ValueError("admission decision requires a reason")
        value = {
            "schema": "sparse-network-admission-decision.v1",
            "id": f"admission-{uuid4()}",
            "run_id": run_id,
            "decision": decision,
            "reason": reason,
            "actor": actor,
            "created_at": datetime.now(UTC).isoformat(),
            "production_configuration_changed": False,
        }
        self.runs_root.mkdir(parents=True, exist_ok=True)
        (self.runs_root / f"{value['id']}.json").write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        self.fleet.event_log.emit(
            event="use_case_admission_decision",
            endpoint="controller",
            request_id=run_id,
            details={"decision": decision, "actor": actor, "id": value["id"]},
        )
        return value
