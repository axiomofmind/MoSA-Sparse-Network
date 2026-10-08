"""Model-free Milestone 14 acceptance smoke through the authenticated API."""

from __future__ import annotations

import json
import tempfile
import threading
from copy import deepcopy
from pathlib import Path
from time import monotonic, sleep

from sparse_network.api import build_server
from sparse_network.client import ControllerClient
from sparse_network.config import AppConfig, load_config
from sparse_network.models import ModelRegistry


def _payloads() -> list[dict[str, object]]:
    return [
        {
            "template_id": "developer-workstation",
            "plan_only": True,
            "inputs": [{"kind": "source_file", "content": "print('ok')"}],
        },
        {
            "template_id": "private-document-analysis",
            "plan_only": True,
            "inputs": [{"kind": "text_document", "content": "field: value"}],
        },
        {
            "template_id": "incident-assistant",
            "plan_only": True,
            "inputs": [{"kind": "log", "content": "service recovered"}],
        },
        {
            "template_id": "structured-batch",
            "plan_only": True,
            "inputs": [],
            "record_schema": {"required": ["value"]},
            "records": [{"id": "one", "data": {"value": "ok"}}],
        },
        {
            "template_id": "research-synthesis",
            "plan_only": True,
            "inputs": [{"kind": "note", "content": "source fact"}],
        },
        {
            "template_id": "troubleshooting-assistant",
            "plan_only": True,
            "inputs": [{"kind": "diagnostic", "content": "read-only result"}],
        },
        {
            "template_id": "meeting-analysis",
            "plan_only": True,
            "inputs": [{"kind": "transcript", "content": "00:00 Speaker A: hello"}],
            "transcript_segments": [{"start": 0, "end": 1, "speaker": "A", "text": "hello"}],
        },
        {
            "template_id": "routing-antidoom-lab",
            "plan_only": True,
            "inputs": [{"kind": "dataset", "content": "case-1"}],
            "dataset_revision": "sha256:fixture",
            "prompt_revision": "sha256:fixture",
            "baselines": [{"id": "top-1", "quality": 100}],
        },
    ]


def main() -> int:
    root = Path(__file__).parents[1]
    base = load_config(root=root, local_config=root / "configs" / "mock.yaml")
    with tempfile.TemporaryDirectory(prefix="sparse-usecase-smoke-") as temporary:
        temp = Path(temporary)
        data = deepcopy(base.data)
        data["paths"].update(
            {
                "artifacts": str(temp / "artifacts"),
                "logs": str(temp / "logs"),
                "runs": str(temp / "runs"),
                "indexes": str(temp / "indexes"),
            }
        )
        config = AppConfig(root=root, data=data, sources=base.sources)
        token = "milestone-14-administrator-token"
        viewer_token = "milestone-14-viewer-token"
        server = build_server(
            config,
            ModelRegistry.load(config),
            port=0,
            token=token,
            role_tokens={"viewer": viewer_token},
        )
        server.service.start()
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        client = ControllerClient(f"http://127.0.0.1:{server.server_port}", token)
        viewer = ControllerClient(f"http://127.0.0.1:{server.server_port}", viewer_token)
        checks: dict[str, bool] = {}
        try:
            catalog = client.request("GET", "/v1/use-cases")
            checks["eight_versioned_templates"] = len(catalog["templates"]) == 8
            checks["bounded_model_stages"] = catalog["maximum_model_stages"] == 3
            try:
                viewer.request("POST", "/v1/use-cases/runs", _payloads()[0])
            except RuntimeError as exc:
                checks["viewer_submission_blocked"] = "403" in str(exc)
            else:
                checks["viewer_submission_blocked"] = False
            for payload in _payloads():
                job = client.request("POST", "/v1/use-cases/runs", payload)
                deadline = monotonic() + 5
                result = client.request("GET", f"/v1/use-cases/runs/{job['id']}")
                while result.get("schema") == "sparse-network-use-case-job.v1":
                    if result.get("state") in {"failed", "cancelled"} or monotonic() > deadline:
                        break
                    sleep(0.02)
                    result = client.request("GET", f"/v1/use-cases/runs/{job['id']}")
                checks[str(payload["template_id"])] = (
                    result.get("schema") == "sparse-network-use-case-run.v1"
                    and result.get("verification", {}).get("accepted") is True
                    and result.get("production_effect") is False
                )
            snapshot = client.request("GET", "/v1/dashboard")
            checks["dashboard_catalog"] = (
                len(snapshot.get("use_cases", {}).get("catalog", {}).get("templates", [])) == 8
            )
            checks["audit_reconstructable"] = (
                sum(
                    event.get("event") == "use_case_completed"
                    for event in snapshot.get("events", [])
                )
                == 8
            )
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)
            server.service.fleet.shutdown()
        report = {
            "schema": "sparse-network-milestone-14-smoke.v1",
            "passed": all(checks.values()),
            "checks": checks,
        }
        print(json.dumps(report, indent=2))
        return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
