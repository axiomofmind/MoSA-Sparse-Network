"""Exercise the complex OCR route with an image-only PDF and installed local models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import monotonic

from PIL import Image

from sparse_network.config import load_config
from sparse_network.fleet import FleetManager
from sparse_network.models import ModelRegistry
from sparse_network.usecases import UseCaseManager


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("fast", "complex"), default="complex")
    arguments = parser.parse_args()
    root = Path(__file__).parents[1]
    source = root / ".sparse-data" / "fixtures" / "vision" / "document.png"
    scanned_pdf = root / ".sparse-data" / "test-inputs" / "scanned-invoice.pdf"
    scanned_pdf.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        image.convert("RGB").save(scanned_pdf, "PDF", resolution=150)

    config = load_config(root=root, local_config=root / "config.local.yaml")
    registry = ModelRegistry.load(config)
    fleet = FleetManager(config, registry)
    manager = UseCaseManager(config, registry, fleet)
    started = monotonic()
    try:
        payload = {
            "template_id": "private-document-analysis",
            "title": "Scanned invoice OCR verification",
            "prompt": (
                "Extract the invoice number, customer, line item, and total from this scanned "
                "invoice. Preserve exact values and cite the OCR evidence."
            ),
            "ocr_mode": arguments.mode,
            "inputs": [{"kind": "pdf", "path": str(scanned_pdf), "render_pages": True}],
        }
        before = manager.readiness(payload)
        endpoints = tuple(before["preparation"]["endpoints"])
        if endpoints:
            fleet.load_endpoints(endpoints)
        after = manager.readiness(payload)
        run = manager.run(payload, actor="local-evaluator", permissions={"usecase:run"})
        answer = str(run["presentation"]["deliverable"]["text"])
        expected = ["inv-204", "ada example", "local inference service", "$42.50"]
        report = {
            "schema": "sparse-network-real-ocr-evaluation.v1",
            "ocr_mode": arguments.mode,
            "elapsed_seconds": round(monotonic() - started, 2),
            "readiness_before": before,
            "readiness_after": after,
            "run_id": run["id"],
            "state": run["state"],
            "accepted": run["verification"]["accepted"],
            "quality_terms_found": {
                value: value in answer.casefold() for value in expected
            },
            "routes": run["routes"],
            "failed_checks": [
                check for check in run["verification"]["checks"] if not check["passed"]
            ],
            "answer": answer,
        }
    finally:
        fleet.shutdown()
    destination = (
        root
        / ".sparse-data"
        / "runs"
        / f"real-ocr-evaluation-{arguments.mode}.json"
    )
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["accepted"] and all(report["quality_terms_found"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
