from __future__ import annotations

import threading
from datetime import date
from pathlib import Path
from time import monotonic, sleep

import pytest
import yaml
from PIL import Image
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from sparse_network.config import AppConfig
from sparse_network.errors import ResourceAdmissionError
from sparse_network.fleet import FleetManager
from sparse_network.models import ModelRegistry
from sparse_network.service import ControllerService
from sparse_network.usecases import UseCaseManager


def _endpoint() -> dict:
    return {
        "display_name": "Mock all-capability endpoint",
        "role": "mock",
        "family": "mock",
        "source": {"type": "builtin"},
        "runtime": {"adapter": "mock"},
        "modalities": ["text", "image", "audio"],
        "capabilities": [
            "classification",
            "extraction",
            "coding",
            "plan_review",
            "reasoning",
            "verification",
            "visual_understanding",
            "document_ocr",
            "text_generation",
            "audio_transcription",
            "speaker_diarization",
        ],
        "context_size": 4096,
        "max_output_tokens": 512,
        "max_input_characters": 3584,
        "proposed_budget": {
            "process_ram_bytes": 0,
            "incremental_vram_bytes": 0,
            "kv_cache_bytes": 0,
            "request_workspace_bytes": 0,
            "vram_reclaim_tolerance_bytes": 0,
        },
        "admission": {"state": "admitted"},
        "license": {"reviewed_on": date(2026, 8, 24)},
    }


def _manager(
    tmp_path: Path, *, include_remediation_probe: bool = False
) -> tuple[FleetManager, UseCaseManager]:
    configs = tmp_path / "configs"
    (configs / "rosters").mkdir(parents=True)
    (configs / "hardware").mkdir()
    (configs / "models.yaml").write_text(
        yaml.safe_dump(
            {
                "schema": "sparse-network-model-registry.v1",
                "endpoints": {"mock-echo": _endpoint()},
            }
        ),
        encoding="utf-8",
    )
    (configs / "rosters" / "test.yaml").write_text(
        yaml.safe_dump({"id": "test", "hardware_profile": "test", "resident": ["mock-echo"]}),
        encoding="utf-8",
    )
    (configs / "hardware" / "test.yaml").write_text(
        yaml.safe_dump(
            {
                "scheduler": {"initial_parallel_generations": 1},
                "budgets": {"minimum_vram_reserve_gb": 0, "total_kv_cache_gb": 1},
            }
        ),
        encoding="utf-8",
    )
    use_cases = Path(__file__).parents[1] / "configs" / "usecases.yaml"
    if include_remediation_probe:
        use_case_data = yaml.safe_load(use_cases.read_text(encoding="utf-8"))
        use_case_data["tools"]["remediation_gate_probe"] = {
            "label": "Remediation gate probe",
            "mode": "state_changing",
            "argv": ["{python}", "-c", "print('remediation authorization accepted')"],
            "timeout_seconds": 30,
        }
        for template_id in ("incident-assistant", "troubleshooting-assistant"):
            use_case_data["templates"][template_id]["allowed_tools"].append(
                "remediation_gate_probe"
            )
        use_cases = configs / "usecases.yaml"
        use_cases.write_text(yaml.safe_dump(use_case_data), encoding="utf-8")
    config = AppConfig(
        root=tmp_path,
        data={
            "paths": {
                "model_cache": None,
                "artifacts": str(tmp_path / "artifacts"),
                "logs": str(tmp_path / "logs"),
                "runs": str(tmp_path / "runs"),
                "indexes": str(tmp_path / "indexes"),
            },
            "runtime_executables": {"llama_cpp": "llama-server"},
            "model_registry": str(configs / "models.yaml"),
            "endpoint_overrides": {},
            "controller": {
                "request_timeout_seconds": 5,
                "telemetry_interval_seconds": 0.01,
            },
            "fleet": {
                "roster": str(configs / "rosters" / "test.yaml"),
                "queue_capacity": 8,
                "require_gpu_telemetry": False,
            },
            "routing": {"use_cases": str(use_cases)},
        },
        sources=(),
    )
    registry = ModelRegistry.load(config)
    fleet = FleetManager(config, registry)
    fleet.load_all()
    return fleet, UseCaseManager(config, registry, fleet)


def _text_pdf(path: Path, text: str) -> None:
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    resources = DictionaryObject(
        {
            NameObject("/Font"): DictionaryObject(
                {NameObject("/F1"): writer._add_object(font)}
            )
        }
    )
    stream = DecodedStreamObject()
    escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream.set_data(f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET".encode("latin-1"))
    page[NameObject("/Resources")] = resources
    page[NameObject("/Contents")] = writer._add_object(stream)
    with path.open("wb") as destination:
        writer.write(destination)


def test_catalog_exposes_all_versioned_bounded_use_cases(tmp_path: Path) -> None:
    fleet, manager = _manager(tmp_path)
    try:
        catalog = manager.catalog()
        assert catalog["schema"] == "sparse-network-use-case-catalog.v1"
        assert catalog["maximum_model_stages"] == 3
        assert {value["kind"] for value in catalog["templates"]} == {
            "developer",
            "document",
            "incident",
            "batch",
            "research",
            "troubleshooting",
            "meeting",
            "experiment",
        }
        assert all(value["dashboard_panels"] for value in catalog["templates"])
        assert all(value["task_ui"]["action"] for value in catalog["templates"])
        assert {
            value["task_ui"]["category"] for value in catalog["templates"]
        } == {"everyday", "advanced"}
        with pytest.raises(PermissionError, match="not frozen"):
            manager.run(
                {
                    "template_id": "developer-workstation",
                    "plan_only": True,
                    "inputs": [],
                    "escalation_trigger": "because-the-model-asked",
                },
                actor="operator",
                permissions={"usecase:run"},
            )
        with pytest.raises(PermissionError, match="outside"):
            manager.run(
                {
                    "template_id": "developer-workstation",
                    "plan_only": True,
                    "inputs": [{"kind": "source_file", "path": str(__file__)}],
                },
                actor="operator",
                permissions={"usecase:run"},
            )
    finally:
        fleet.shutdown()


def test_developer_workflow_routes_visual_and_text_work_without_losing_artifacts(
    tmp_path: Path,
) -> None:
    fleet, manager = _manager(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = workspace / "app.py"
    source.write_text("print('ok')\n", encoding="utf-8")
    screenshot = workspace / "screen.png"
    Image.new("RGB", (8, 8), "navy").save(screenshot)
    try:
        visual = manager.run(
            {
                "template_id": "developer-workstation",
                "prompt": "Diagnose the screenshot and verify the repair",
                "inputs": [
                    {"kind": "source_file", "path": str(source)},
                    {"kind": "screenshot", "path": str(screenshot)},
                ],
                "working_directory": str(workspace),
                "tools": ["python_compile"],
                "flags": ["failed"],
            },
            actor="operator",
            permissions={"usecase:run", "tool:diagnose"},
        )
        assert visual["state"] == "accepted"
        assert visual["routes"][0]["stage"] == "visual_evidence"
        assert visual["routes"][1]["role"] == "repair"
        assert visual["tools"][0]["status"] == "completed"
        assert any(
            fleet.controller.artifacts.get(value).kind == "tool_result"
            for value in visual["outputs"]
        )

        text_only = manager.run(
            {
                "template_id": "developer-workstation",
                "prompt": "Repair this text-only function",
                "inputs": [{"kind": "source_file", "path": str(source)}],
            },
            actor="operator",
            permissions={"usecase:run"},
        )
        assert text_only["state"] == "accepted"
        assert all(route["stage"] != "visual_evidence" for route in text_only["routes"])
        assert text_only["routes"][0]["role"] == "repair"
    finally:
        fleet.shutdown()


def test_text_sources_are_evidence_and_later_stages_use_prior_answers(
    tmp_path: Path,
) -> None:
    fleet, manager = _manager(tmp_path)
    log = tmp_path / "service.log"
    log.write_text(
        "2026-10-08 cache-sync ERROR Redis timeout on port 6380\n", encoding="utf-8"
    )
    try:
        result = manager.run(
            {
                "template_id": "incident-assistant",
                "prompt": "Explain the failure and propose a reversible diagnostic.",
                "inputs": [
                    {
                        "kind": "log",
                        "path": str(log),
                    }
                ],
            },
            actor="operator",
            permissions={"usecase:run"},
        )
        evidence_ids = result["inputs"][0]["evidence_ids"]
        assert evidence_ids
        assert all(
            fleet.controller.artifacts.get(artifact_id).kind == "evidence_chunk"
            for artifact_id in evidence_ids
        )
        assert result["state"] == "accepted"
        assert "Redis timeout" in result["presentation"]["deliverable"]["text"]
        assert "Stage role: diagnosis" in result["presentation"]["deliverable"]["text"]
        assert "Stage role: critique" not in result["presentation"]["deliverable"]["text"]
        diagnosis = next(route for route in result["routes"] if route["role"] == "diagnosis")
        diagnosis_text = fleet.controller.artifacts.read_text(diagnosis["answer_reference"])
        assert "Prior stage symptoms" in diagnosis_text
        assert diagnosis["context"]["input_stage_references"]
        assert diagnosis["context"]["input_evidence_references"] == evidence_ids
        assert isinstance(diagnosis["context"]["handoff_truncated"], bool)
    finally:
        fleet.shutdown()


def test_follow_up_inherits_verified_sources_and_prior_result(tmp_path: Path) -> None:
    fleet, manager = _manager(tmp_path)
    note = tmp_path / "service.log"
    note.write_text("gateway timeout after 30 seconds\n", encoding="utf-8")
    try:
        first = manager.run(
            {
                "template_id": "incident-assistant",
                "title": "Gateway incident",
                "prompt": "Identify the observed symptom.",
                "inputs": [{"kind": "log", "path": str(note)}],
            },
            actor="operator",
            permissions={"usecase:run"},
        )
        follow_up = manager.run(
            {
                "template_id": "incident-assistant",
                "title": "Gateway incident",
                "prompt": "What reversible check should I try next?",
                "inputs": [],
                "parent_run_id": first["id"],
            },
            actor="operator",
            permissions={"usecase:run"},
        )
        assert follow_up["parent_run_id"] == first["id"]
        assert follow_up["thread_id"] == first["id"]
        assert follow_up["turn_index"] == 2
        assert follow_up["inputs"][0]["artifact_ids"] == first["inputs"][0][
            "artifact_ids"
        ]
        prior = next(value for value in follow_up["inputs"] if value["kind"] == "prior_result")
        assert prior["evidence_ids"]
        assert fleet.controller.artifacts.get(prior["artifact_ids"][0]).metadata[
            "parent_run_id"
        ] == first["id"]
        assert any(
            value["event"] == "continuation_context_attached"
            for value in follow_up["timeline"]
        )
        assert set(follow_up["routes"][0]["context"]["input_evidence_references"]).issuperset(
            first["inputs"][0]["evidence_ids"]
        )

        with pytest.raises(ValueError, match="same task template"):
            manager.readiness(
                {
                    "template_id": "private-document-analysis",
                    "inputs": [],
                    "parent_run_id": first["id"],
                }
            )
    finally:
        fleet.shutdown()


def test_digital_pdf_uses_embedded_text_without_requiring_renderer(tmp_path: Path) -> None:
    fleet, manager = _manager(tmp_path)
    pdf = tmp_path / "invoice.pdf"
    _text_pdf(pdf, "Invoice INV-204 Total USD 1,250.00 Due 2026-11-15")
    try:
        payload = {
            "template_id": "private-document-analysis",
            "prompt": "State the invoice number, total, and due date with a source citation.",
            "inputs": [{"kind": "pdf", "path": str(pdf), "render_pages": True}],
        }
        readiness = manager.readiness(payload)
        assert all(stage["role"] != "ocr" for stage in readiness["required"])
        result = manager.run(
            payload,
            actor="evaluator",
            permissions={"usecase:run"},
        )
        assert result["state"] == "accepted"
        assert result["inputs"][0]["evidence_ids"]
        assert len(result["inputs"][0]["artifact_ids"]) == 1
        evidence = fleet.controller.artifacts.get(result["inputs"][0]["evidence_ids"][0])
        assert evidence.metadata["page_number"] == 1
        assert "INV-204" in fleet.controller.artifacts.read_text(evidence.id)
        assert all(route["role"] != "ocr" for route in result["routes"])
    finally:
        fleet.shutdown()


def test_scanned_pdf_is_rendered_and_ocr_output_becomes_evidence(tmp_path: Path) -> None:
    fleet, manager = _manager(tmp_path)
    pdf = tmp_path / "scan.pdf"
    Image.new("RGB", (400, 180), "white").save(pdf, "PDF", resolution=150)
    try:
        payload = {
            "template_id": "private-document-analysis",
            "prompt": "Transcribe the scanned page and cite the OCR evidence.",
            "ocr_mode": "complex",
            "inputs": [{"kind": "pdf", "path": str(pdf), "render_pages": True}],
        }
        readiness = manager.readiness(payload)
        assert next(stage for stage in readiness["required"] if stage["role"] == "ocr")[
            "status"
        ] == "ready"
        result = manager.run(
            payload,
            actor="evaluator",
            permissions={"usecase:run"},
        )
        assert result["state"] == "accepted"
        assert any(value["kind"] == "ocr_text" for value in result["inputs"])
        ocr_route = next(route for route in result["routes"] if route["role"] == "ocr")
        assert ocr_route["ocr_evidence_references"]
        assert any(
            fleet.controller.artifacts.get(artifact_id).kind == "page_image"
            for artifact_id in result["inputs"][0]["artifact_ids"]
        )
    finally:
        fleet.shutdown()


def test_document_rows_without_valid_value_sources_are_rejected(tmp_path: Path) -> None:
    fleet, manager = _manager(tmp_path)
    try:
        result = manager.run(
            {
                "template_id": "private-document-analysis",
                "prompt": "Extract the invoice total.",
                "inputs": [{"kind": "text_document", "content": "Invoice total: 42"}],
                "output_intent": "table",
                "document_fields": [
                    {"id": "total", "label": "Total", "type": "string", "required": True}
                ],
                "extracted_rows": [{"total": "999"}],
            },
            actor="evaluator",
            permissions={"usecase:run"},
        )
        assert result["state"] == "rejected"
        grounding = next(
            check
            for check in result["verification"]["checks"]
            if check["name"] == "document_value_grounding"
        )
        assert not grounding["passed"]
        assert any("not found in source text" in failure for failure in grounding["failures"])
        assert any("no valid source reference" in failure for failure in grounding["failures"])
    finally:
        fleet.shutdown()


def test_document_and_research_claims_require_source_evidence(tmp_path: Path) -> None:
    fleet, manager = _manager(tmp_path)
    first_page = tmp_path / "page-1.png"
    second_page = tmp_path / "page-2.png"
    Image.new("RGB", (8, 8), "white").save(first_page)
    Image.new("RGB", (8, 8), "gray").save(second_page)
    try:
        supported = manager.run(
            {
                "template_id": "private-document-analysis",
                "plan_only": True,
                "inputs": [{"kind": "text_document", "content": "Invoice total: 42"}],
                "claims": [
                    {
                        "id": "total",
                        "text": "The total is 42",
                        "class": "observed",
                        "evidence_references": ["input:0"],
                        "page": 1,
                    }
                ],
            },
            actor="evaluator",
            permissions={"usecase:run"},
        )
        assert supported["verification"]["accepted"]
        assert supported["claims"][0]["supported"]
        assert supported["presentation"]["schema"] == "sparse-network-result-view.v1"
        assert supported["presentation"]["deliverable"]["text"] == "The total is 42"
        assert supported["presentation"]["citations"][0]["page"] == 1
        assert supported["presentation"]["version"]["immutable_original"]
        corrected = manager.correct_result(
            supported["id"],
            {
                "version_id": supported["presentation"]["version"]["id"],
                "changes": [
                    {
                        "field_id": "$text",
                        "value": "The reviewed total is 42",
                        "source_reference": supported["claims"][0][
                            "evidence_references"
                        ][0],
                    }
                ],
            },
            actor="evaluator",
        )
        assert corrected["presentation"]["version"]["parent_id"].endswith(":v1")
        assert corrected["presentation"]["deliverable"]["text"] == (
            "The reviewed total is 42"
        )
        assert any(
            check.get("status") == "stale"
            for check in corrected["presentation"]["checks"]
        )
        report = manager.export_result(
            supported["id"],
            {
                "version_id": corrected["presentation"]["version"]["id"],
                "format": "report",
            },
            actor="evaluator",
        )
        assert report["format"] == "report"
        assert "reviewed total" in fleet.controller.artifacts.read_text(
            report["artifact"]["id"]
        )

        table = manager.run(
            {
                "template_id": "private-document-analysis",
                "plan_only": True,
                "inputs": [{"kind": "text_document", "content": "Invoice"}],
                "output_intent": "table",
                "document_fields": [
                    {"id": "total", "label": "Total", "type": "string"}
                ],
                "extracted_rows": [{"total": "=2+2"}],
            },
            actor="evaluator",
            permissions={"usecase:run"},
        )
        csv_export = manager.export_result(
            table["id"],
            {
                "version_id": table["presentation"]["version"]["id"],
                "format": "csv",
            },
            actor="evaluator",
        )
        assert "'=2+2" in fleet.controller.artifacts.read_text(
            csv_export["artifact"]["id"]
        )

        multipage = manager.run(
            {
                "template_id": "private-document-analysis",
                "plan_only": True,
                "inputs": [
                    {"kind": "page_image", "path": str(first_page), "metadata": {"page_number": 1}},
                    {
                        "kind": "page_image",
                        "path": str(second_page),
                        "metadata": {"page_number": 2},
                    },
                ],
                "claims": [
                    {
                        "id": "p1",
                        "text": "Page one",
                        "class": "observed",
                        "evidence_references": ["input:0"],
                        "page": 1,
                    },
                    {
                        "id": "p2",
                        "text": "Page two",
                        "class": "observed",
                        "evidence_references": ["input:1"],
                        "page": 2,
                    },
                ],
                "ocr_mode": "complex",
                "high_risk": True,
                "human_review_required": True,
            },
            actor="evaluator",
            permissions={"usecase:run"},
        )
        assert multipage["state"] == "needs_human_review"
        assert [value["page"] for value in multipage["claims"]] == [1, 2]
        assert multipage["controls"]["ocr_mode"] == "complex"
        assert next(
            route for route in multipage["routes"] if route["stage"] == "extract"
        )["ocr_mode"] == "complex"
        assert all(
            fleet.controller.artifacts.get(value["artifact_ids"][0]).kind == "page_image"
            for value in multipage["inputs"]
        )

        indexed = manager.run(
            {
                "template_id": "research-synthesis",
                "plan_only": True,
                "inputs": [{"kind": "note", "content": "The measured fleet uses 17 GB VRAM."}],
                "index_sources": True,
                "retrieval_endpoint": "hash-embedding",
                "query": "measured fleet VRAM",
                "claims": [
                    {
                        "id": "retrieved",
                        "text": "The fleet measurement is recorded",
                        "class": "observed",
                        "evidence_references": ["input:1"],
                    }
                ],
            },
            actor="evaluator",
            permissions={"usecase:run"},
        )
        assert indexed["retrieval"]["embedding_endpoint"] == "hash-embedding"
        assert indexed["retrieval"]["hits"]
        assert indexed["claims"][0]["supported"]

        unsupported = manager.run(
            {
                "template_id": "research-synthesis",
                "plan_only": True,
                "inputs": [{"kind": "note", "content": "Measured result"}],
                "claims": [
                    {
                        "id": "invented",
                        "text": "Unsupported conclusion",
                        "class": "inferred",
                        "evidence_references": [],
                    }
                ],
            },
            actor="evaluator",
            permissions={"usecase:run"},
        )
        assert unsupported["state"] == "rejected"
        assert not unsupported["claims"][0]["supported"]
    finally:
        fleet.shutdown()


def test_request_specific_readiness_separates_required_and_optional_stages(
    tmp_path: Path,
) -> None:
    fleet, manager = _manager(tmp_path)
    try:
        readiness = manager.readiness(
            {
                "template_id": "private-document-analysis",
                "inputs": [{"kind": "text_document"}],
            }
        )
        assert readiness["schema"] == "sparse-network-use-case-readiness.v1"
        assert readiness["state"] == "ready"
        assert [stage["id"] for stage in readiness["required"]] == ["interpret"]
        assert {stage["id"] for stage in readiness["optional"]} == {"extract", "challenge"}
    finally:
        fleet.shutdown()


def test_incident_and_troubleshooting_remediation_is_policy_gated(tmp_path: Path) -> None:
    fleet, manager = _manager(tmp_path, include_remediation_probe=True)
    screenshot = tmp_path / "incident.png"
    Image.new("RGB", (8, 8), "red").save(screenshot)
    try:
        blocked = manager.run(
            {
                "template_id": "incident-assistant",
                "plan_only": True,
                "inputs": [
                    {"kind": "log", "content": "service unavailable"},
                    {"kind": "screenshot", "path": str(screenshot)},
                ],
                "tools": ["remediation_gate_probe"],
            },
            actor="operator",
            permissions={"usecase:run", "tool:diagnose"},
        )
        assert blocked["tools"][0]["status"] == "blocked"
        assert blocked["state"] == "rejected"
        assert blocked["routes"][0]["stage"] == "visual_evidence"
        assert any(value["event"] == "tool_blocked" for value in blocked["timeline"])

        authorized = manager.run(
            {
                "template_id": "troubleshooting-assistant",
                "plan_only": True,
                "inputs": [{"kind": "diagnostic", "content": "read-only check"}],
                "tools": ["python_compile", "remediation_gate_probe"],
                "confirmation": "AUTHORIZE REMEDIATION",
                "rollback_information": "No state is changed by the gate probe",
            },
            actor="administrator",
            permissions={"usecase:run", "tool:diagnose", "tool:remediate"},
        )
        assert [value["status"] for value in authorized["tools"]] == [
            "completed",
            "completed",
        ]
    finally:
        fleet.shutdown()


def test_batch_preserves_identity_and_replays_only_dead_letters(tmp_path: Path) -> None:
    fleet, manager = _manager(tmp_path)
    image = tmp_path / "record.png"
    Image.new("RGB", (8, 8), "green").save(image)
    try:
        original = manager.run(
            {
                "template_id": "structured-batch",
                "plan_only": True,
                "inputs": [{"kind": "image_record", "path": str(image)}],
                "record_schema": {
                    "required": ["name"],
                    "properties": {"name": {"type": "string"}},
                },
                "records": [
                    {"id": "good", "data": {"name": "alpha"}},
                    {"id": "bad", "data": {"value": 2}, "visual": True},
                ],
            },
            actor="operator",
            permissions={"usecase:run"},
        )
        assert original["metrics"]["accepted"] == 1
        assert original["metrics"]["dead_letter"] == 1
        assert original["metrics"]["escalated"] == 1
        assert [value["role"] for value in original["routes"]] == [
            "visual_interpretation",
            "extraction",
            "exception",
        ]
        assert [value["id"] for value in original["record_traces"]] == ["good", "bad"]

        replay = manager.replay_batch(
            original["id"],
            {
                "confirmation": f"REPLAY {original['id']}",
                "record_schema": {},
            },
            actor="operator",
            permissions={"usecase:run"},
        )
        assert replay["metrics"]["total"] == 1
        assert replay["metrics"]["accepted"] == 1
        assert replay["record_traces"][0]["id"] == "bad"
    finally:
        fleet.shutdown()


def test_meeting_preserves_original_transcript_and_separate_corrections(
    tmp_path: Path,
) -> None:
    fleet, manager = _manager(tmp_path)
    try:
        run = manager.run(
            {
                "template_id": "meeting-analysis",
                "plan_only": True,
                "inputs": [
                    {
                        "kind": "audio",
                        "content_base64": "UklGRg==",
                        "suffix": ".wav",
                        "media_type": "audio/wav",
                        "retention": {"class": "session"},
                    }
                ],
                "transcript_segments": [
                    {"start": 0.0, "end": 1.5, "speaker": "A", "text": "Hello"}
                ],
                "speaker_corrections": [{"segment": 0, "speaker": "Alex"}],
                "language": "en",
                "hotwords": ["Sparse"],
                "claims": [
                    {
                        "id": "greeting",
                        "text": "A greeting occurred",
                        "class": "observed",
                        "evidence_references": ["input:0"],
                        "timestamp_start": 0.0,
                        "timestamp_end": 1.5,
                    }
                ],
            },
            actor="operator",
            permissions={"usecase:run"},
        )
        kinds = [fleet.controller.artifacts.get(value).kind for value in run["outputs"]]
        assert "timestamped_transcript" in kinds
        assert "speaker_correction" in kinds
        correction_id = run["outputs"][kinds.index("speaker_correction")]
        correction = fleet.controller.artifacts.get(correction_id)
        assert correction.metadata["rewrites_original"] is False
        assert run["controls"]["hotwords"] == ["Sparse"]
    finally:
        fleet.shutdown()


def test_experiment_is_reproducible_and_cannot_silently_change_routing(
    tmp_path: Path,
) -> None:
    fleet, manager = _manager(tmp_path)
    try:
        run = manager.run(
            {
                "template_id": "routing-antidoom-lab",
                "plan_only": True,
                "inputs": [{"kind": "dataset", "content": "case-1"}],
                "dataset_revision": "sha256:dataset",
                "prompt_revision": "sha256:prompt",
                "seed": 7,
                "settings": {"temperature": 0},
                "baselines": [
                    {"id": "top-1", "quality": 80, "ci_low": 77, "ci_high": 83},
                    {"id": "top-2", "quality": 88, "ci_low": 85, "ci_high": 91},
                    {"id": "cascade", "quality": 90},
                    {"id": "oracle", "quality": 95},
                    {"id": "always-endpoint", "quality": 84},
                ],
            },
            actor="evaluator",
            permissions={"usecase:run"},
        )
        reloaded = manager.get(run["id"])
        assert reloaded["experiment"]["seed"] == 7
        assert reloaded["production_effect"] is False
        decision = manager.admission_decision(
            run["id"],
            {
                "decision": "admit",
                "confirmation": f"ADMIT {run['id']}",
                "reason": "Frozen suite passed",
            },
            actor="administrator",
        )
        assert decision["production_configuration_changed"] is False
    finally:
        fleet.shutdown()


def test_service_runs_and_cancels_use_cases_as_controller_jobs(tmp_path: Path) -> None:
    fleet, manager = _manager(tmp_path)
    config = manager.config
    registry = manager.registry
    fleet.shutdown()
    service = ControllerService(config, registry)
    service.start()
    try:
        submitted = service.submit_use_case(
            {
                "template_id": "developer-workstation",
                "plan_only": True,
                "inputs": [{"kind": "source_file", "content": "print('ok')"}],
            },
            actor="operator",
            permissions={"usecase:run"},
        )
        deadline = monotonic() + 3
        while (
            service.use_case_jobs[submitted["id"]].state
            not in {
                "completed",
                "failed",
            }
            and monotonic() < deadline
        ):
            sleep(0.01)
        assert service.use_case_jobs[submitted["id"]].state == "completed"
        stage_events = [
            event["details"]["stage_id"]
            for event in service.fleet.event_log.read()
            if event["event"] == "use_case_stage_changed"
            and event.get("request_id") == submitted["id"]
        ]
        assert stage_events == [
            "reading_sources",
            "reading_sources",
            "extracting_fields",
            "checking_references",
            "preparing_output",
        ]
        assert service.use_case_run(submitted["id"])["schema"] == ("sparse-network-use-case-run.v1")

        cancellable = service.submit_use_case(
            {
                "template_id": "developer-workstation",
                "prompt": "[mock:slow]",
                "inputs": [{"kind": "source_file", "content": "print('slow')"}],
            },
            actor="operator",
            permissions={"usecase:run"},
        )
        service.cancel_use_case(
            cancellable["id"],
            confirmation=f"CANCEL {cancellable['id']}",
            actor="operator",
        )
        deadline = monotonic() + 3
        while (
            service.use_case_jobs[cancellable["id"]].state
            not in {
                "cancelled",
                "failed",
            }
            and monotonic() < deadline
        ):
            sleep(0.01)
        assert service.use_case_jobs[cancellable["id"]].state == "cancelled"
        assert any(
            event["event"] == "use_case_cancellation_requested"
            for event in service.fleet.event_log.read()
        )
    finally:
        service.fleet.shutdown()


def test_service_prepares_installed_models_for_authorized_task(tmp_path: Path) -> None:
    fleet, manager = _manager(tmp_path)
    config = manager.config
    registry = manager.registry
    fleet.shutdown()
    service = ControllerService(config, registry)
    payload = {
        "template_id": "developer-workstation",
        "plan_only": True,
        "inputs": [{"kind": "source_file", "content": "print('ok')"}],
    }
    try:
        readiness = service.use_case_readiness(
            payload, permissions={"usecase:run", "fleet:operate"}
        )
        assert readiness["state"] == "needs_preparation"
        assert readiness["preparation"]["permitted"] is True
        with pytest.raises(PermissionError, match="authorized role"):
            service.submit_use_case(
                payload,
                actor="operator",
                permissions={"usecase:run"},
            )

        submitted = service.submit_use_case(
            payload,
            actor="administrator",
            permissions={"usecase:run", "fleet:operate"},
        )
        deadline = monotonic() + 3
        while (
            service.use_case_jobs[submitted["id"]].state
            not in {"completed", "failed"}
            and monotonic() < deadline
        ):
            sleep(0.01)
        assert service.use_case_jobs[submitted["id"]].state == "completed"
        assert any(
            event["event"] == "use_case_preparing_models"
            and event.get("request_id") == submitted["id"]
            for event in service.fleet.event_log.read()
        )
    finally:
        service.fleet.shutdown()


def test_service_coordinates_concurrent_preparation_and_preserves_failures(
    tmp_path: Path,
) -> None:
    fleet, manager = _manager(tmp_path)
    config = manager.config
    registry = manager.registry
    fleet.shutdown()
    service = ControllerService(config, registry)
    payload = {
        "template_id": "developer-workstation",
        "plan_only": True,
        "inputs": [{"kind": "source_file", "content": "print('ok')"}],
    }
    permissions = {"usecase:run", "fleet:operate"}
    original_load = service.fleet.load_endpoints
    loading_before = sum(
        event["event"] == "state_transition"
        and event["details"].get("to") == "loading"
        for event in service.fleet.event_log.read()
    )
    entered = threading.Event()
    release = threading.Event()

    def delayed_load(endpoint_ids: tuple[str, ...] | list[str]) -> dict[str, object]:
        entered.set()
        assert release.wait(2)
        return original_load(endpoint_ids)

    service.fleet.load_endpoints = delayed_load  # type: ignore[method-assign]
    try:
        first = service.submit_use_case(
            payload, actor="administrator", permissions=permissions
        )
        assert entered.wait(2)
        second = service.submit_use_case(
            payload, actor="administrator", permissions=permissions
        )
        release.set()
        deadline = monotonic() + 3
        while (
            any(
                service.use_case_jobs[value["id"]].state
                not in {"completed", "failed"}
                for value in (first, second)
            )
            and monotonic() < deadline
        ):
            sleep(0.01)
        assert [
            service.use_case_jobs[value["id"]].state for value in (first, second)
        ] == ["completed", "completed"]
        loading_events = [
            event
            for event in service.fleet.event_log.read()
            if event["event"] == "state_transition"
            and event["details"].get("to") == "loading"
        ]
        assert len(loading_events) == loading_before + 1

        service.fleet.unload_all()
        cancel_entered = threading.Event()
        cancel_release = threading.Event()

        def cancellable_load(endpoint_ids: tuple[str, ...] | list[str]) -> dict[str, object]:
            cancel_entered.set()
            assert cancel_release.wait(2)
            return original_load(endpoint_ids)

        service.fleet.load_endpoints = cancellable_load  # type: ignore[method-assign]
        cancelled = service.submit_use_case(
            payload, actor="administrator", permissions=permissions
        )
        assert cancel_entered.wait(2)
        service.cancel_use_case(
            cancelled["id"],
            confirmation=f"CANCEL {cancelled['id']}",
            actor="administrator",
        )
        cancel_release.set()
        deadline = monotonic() + 3
        while (
            service.use_case_jobs[cancelled["id"]].state
            not in {"cancelled", "failed"}
            and monotonic() < deadline
        ):
            sleep(0.01)
        assert service.use_case_jobs[cancelled["id"]].state == "cancelled"

        service.fleet.unload_all()

        def insufficient_resources(
            endpoint_ids: tuple[str, ...] | list[str],
        ) -> dict[str, object]:
            raise ResourceAdmissionError(f"insufficient resources for {endpoint_ids}")

        service.fleet.load_endpoints = insufficient_resources  # type: ignore[method-assign]
        failed = service.submit_use_case(
            payload, actor="administrator", permissions=permissions
        )
        deadline = monotonic() + 3
        while (
            service.use_case_jobs[failed["id"]].state
            not in {"completed", "failed"}
            and monotonic() < deadline
        ):
            sleep(0.01)
        assert service.use_case_jobs[failed["id"]].state == "failed"
        assert service.use_case_jobs[failed["id"]].error_type == "ResourceAdmissionError"
        assert "insufficient resources" in str(service.use_case_run(failed["id"])["error"])
        assert service.use_case_run(failed["id"])["id"] == failed["id"]
    finally:
        service.fleet.shutdown()
