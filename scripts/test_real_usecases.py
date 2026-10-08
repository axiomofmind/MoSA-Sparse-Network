"""Run evidence-grounded use-case checks against installed local models and real files."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic
from typing import Any

from sparse_network.config import load_config
from sparse_network.fleet import FleetManager
from sparse_network.models import ModelRegistry
from sparse_network.usecases import UseCaseManager


def _cases(root: Path) -> list[dict[str, Any]]:
    inputs = root / ".sparse-data" / "test-inputs"
    retrieval = root / ".sparse-data" / "fixtures" / "retrieval"
    return [
        {
            "id": "developer-real-source",
            "expect": ["mock:slow", "timeout"],
            "payload": {
                "template_id": "developer-workstation",
                "prompt": (
                    "In the supplied Python source, identify the exact marker that simulates a "
                    "timeout and explain the behavior. Cite the source. "
                    "Do not propose unrelated code."
                ),
                "inputs": [
                    {
                        "kind": "source_file",
                        "path": str(root / "src" / "sparse_network" / "runtimes" / "mock.py"),
                    }
                ],
            },
        },
        {
            "id": "document-real-pdf",
            "expect": ["w-9", "march 2024", "requester"],
            "payload": {
                "template_id": "private-document-analysis",
                "prompt": (
                    "Identify the form, its revision, and whether the completed form should be "
                    "sent to the IRS or given to the requester. "
                    "Cite the relevant PDF page evidence."
                ),
                "inputs": [
                    {
                        "kind": "pdf",
                        "path": str(inputs / "irs-form-w9.pdf"),
                        "render_pages": True,
                    }
                ],
            },
        },
        {
            "id": "incident-real-log",
            "expect": ["cors", "api key"],
            "payload": {
                "template_id": "incident-assistant",
                "prompt": (
                    "What concrete security warning appears in this server log, and what is the "
                    "smallest reversible mitigation? Separate the observation from the proposal."
                ),
                "inputs": [
                    {
                        "kind": "log",
                        "path": str(root / ".sparse-data" / "logs" / "qwen35-4b.log"),
                    }
                ],
            },
        },
        {
            "id": "batch-deterministic",
            "expect": ["valid-1", "ada"],
            "payload": {
                "template_id": "structured-batch",
                "prompt": "Validate each record against the schema.",
                "inputs": [],
                "record_schema": {
                    "required": ["name", "count"],
                    "properties": {"name": {"type": "string"}, "count": {"type": "integer"}},
                },
                "records": [
                    {"id": "valid-1", "data": {"name": "Ada", "count": 3}},
                    {"id": "invalid-1", "data": {"name": "Lin"}},
                    {"id": "invalid-2", "data": {"name": "Mia", "count": "four"}},
                ],
            },
        },
        {
            "id": "research-real-files",
            "expect": ["cobalt", "8443"],
            "payload": {
                "template_id": "research-synthesis",
                "prompt": (
                    "What color is the emergency cooling valve, what happens when it is pulled, "
                    "and which port accepts encrypted client traffic? Cite each answer."
                ),
                "query": "emergency cooling valve color reserve pump encrypted client port",
                "index_sources": True,
                "retrieval_endpoint": "harrier-0.6b",
                "inputs": [
                    {"kind": "text_document", "path": str(retrieval / "operations.md")},
                    {"kind": "text_document", "path": str(retrieval / "research.md")},
                ],
            },
        },
        {
            "id": "troubleshooting-real-config",
            "expect": ["gpu index 2", "hf_hub_offline", "transformers_offline"],
            "payload": {
                "template_id": "troubleshooting-assistant",
                "prompt": (
                    "Which GPU is assigned to qwen3-8b-fp8 and which offline environment flags "
                    "are set? Give a reversible check for a model-loading problem and cite "
                    "the config."
                ),
                "inputs": [
                    {"kind": "configuration", "path": str(root / "config.local.yaml")}
                ],
            },
        },
        {
            "id": "meeting-real-transcript-file",
            "expect": ["priya", "friday", "maintenance"],
            "payload": {
                "template_id": "meeting-analysis",
                "prompt": (
                    "Summarize the decision, owner, timing, risk, and follow-up action. "
                    "Cite the transcript evidence."
                ),
                "inputs": [
                    {"kind": "transcript", "path": str(inputs / "meeting-transcript.txt")}
                ],
            },
        },
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", action="append", dest="case_ids")
    arguments = parser.parse_args()
    root = Path(__file__).parents[1]
    config = load_config(root=root, local_config=root / "config.local.yaml")
    registry = ModelRegistry.load(config)
    fleet = FleetManager(config, registry)
    manager = UseCaseManager(config, registry, fleet)
    results: list[dict[str, Any]] = []
    started = monotonic()
    try:
        fleet.load_all()
        cases = _cases(root)
        if arguments.case_ids:
            selected = set(arguments.case_ids)
            cases = [case for case in cases if case["id"] in selected]
        for case in cases:
            case_started = monotonic()
            try:
                run = manager.run(
                    case["payload"],
                    actor="local-evaluator",
                    permissions={"usecase:run"},
                )
                answer = str(run.get("presentation", {}).get("deliverable", {}).get("text", ""))
                expected = [str(value).casefold() for value in case["expect"]]
                checks = run.get("verification", {}).get("checks", [])
                results.append(
                    {
                        "id": case["id"],
                        "run_id": run["id"],
                        "state": run["state"],
                        "accepted": run.get("verification", {}).get("accepted", False),
                        "quality_terms_found": {
                            value: value in answer.casefold() for value in expected
                        },
                        "quality_passed": all(value in answer.casefold() for value in expected),
                        "failed_checks": [
                            check for check in checks if not bool(check.get("passed"))
                        ],
                        "routes": run.get("routes", []),
                        "elapsed_seconds": round(monotonic() - case_started, 2),
                        "answer": answer,
                    }
                )
            except Exception as exc:
                results.append(
                    {
                        "id": case["id"],
                        "state": "error",
                        "accepted": False,
                        "quality_passed": False,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "elapsed_seconds": round(monotonic() - case_started, 2),
                    }
                )
    finally:
        fleet.shutdown()
    report = {
        "schema": "sparse-network-real-usecase-evaluation.v1",
        "created_at": datetime.now(UTC).isoformat(),
        "elapsed_seconds": round(monotonic() - started, 2),
        "passed": all(
            result.get("accepted") and result.get("quality_passed") for result in results
        ),
        "results": results,
    }
    destination = root / ".sparse-data" / "runs" / "real-usecase-evaluation.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
