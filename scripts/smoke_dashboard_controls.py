"""Exercise Milestone 13 roles and controlled operations on the mock fleet."""

from __future__ import annotations

import argparse
import json
import threading
from pathlib import Path
from time import monotonic, sleep
from typing import Any

from sparse_network.api import build_server
from sparse_network.client import ControllerClient
from sparse_network.config import load_config
from sparse_network.models import ModelRegistry


def _wait_for_request(
    client: ControllerClient, request_id: str, *, timeout_seconds: float = 10
) -> dict[str, Any]:
    deadline = monotonic() + timeout_seconds
    while monotonic() < deadline:
        request = client.request_status(request_id)
        if request["state"] in {"completed", "cancelled", "failed"}:
            return request
        sleep(0.05)
    raise TimeoutError(request_id)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/mock.yaml"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".sparse-data/mock/runs/milestone-13-dashboard.json"),
    )
    args = parser.parse_args()

    admin_token = "milestone-13-admin-token"
    viewer_token = "milestone-13-viewer-token"
    config = load_config(local_config=args.config)
    server = build_server(
        config,
        ModelRegistry.load(config),
        port=0,
        token=admin_token,
        role_tokens={"viewer": viewer_token},
    )
    server.service.start()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"
    admin = ControllerClient(base_url, admin_token)
    viewer = ControllerClient(base_url, viewer_token)
    checks: dict[str, bool] = {}
    try:
        viewer_snapshot = viewer.request("GET", "/v1/dashboard")
        checks["viewer_is_read_only"] = viewer_snapshot["session"]["read_only"] is True
        try:
            viewer.operation("smoke", "mock-echo")
        except RuntimeError as exc:
            checks["viewer_operation_denied"] = "403" in str(exc)
        else:
            checks["viewer_operation_denied"] = False

        admin_snapshot = admin.request("GET", "/v1/dashboard")
        checks["administrator_controls"] = all(
            permission in admin_snapshot["session"]["permissions"]
            for permission in ("fleet:operate", "profile:apply", "configuration:write")
        )
        smoke = admin.operation("smoke", "mock-echo")
        checks["controlled_smoke"] = smoke["envelope"]["status"] == "answer"

        slow = admin.submit("mock-echo", "[mock:slow] cancellation sentinel")
        cancelled = admin.cancel(
            str(slow["id"]), confirmation=f"CANCEL {slow['id']}"
        )
        terminal = _wait_for_request(admin, str(slow["id"]))
        checks["confirmed_cancellation"] = cancelled["id"] == slow["id"] and terminal[
            "state"
        ] == "cancelled"

        config_update = admin.request(
            "PATCH",
            "/v1/config",
            {
                "changes": {"fleet.queue_capacity": 17},
                "confirmation": "APPLY CONFIG",
            },
        )
        checks["versioned_config"] = str(config_update["id"]).startswith("config-update-")

        evaluation = admin.request(
            "POST",
            "/v1/evaluations",
            {
                "suite": "milestone-13-smoke",
                "baselines": [
                    {"id": "top-1", "metrics": {"acceptance_rate": 82}},
                    {"id": "top-2", "metrics": {"acceptance_rate": 90}},
                ],
            },
        )
        checks["evaluation_manifest"] = len(evaluation["baselines"]) == 2

        plan = admin.request(
            "POST", "/v1/profiles/plan", {"profile_id": "minimal-mock"}
        )
        checks["profile_preflight"] = plan["valid"] is True
        try:
            admin.request(
                "POST",
                "/v1/profiles/apply",
                {"plan_id": plan["id"], "confirmation": "wrong"},
            )
        except RuntimeError as exc:
            checks["profile_confirmation_required"] = "confirmation phrase" in str(exc)
        else:
            checks["profile_confirmation_required"] = False
        switched = admin.request(
            "POST",
            "/v1/profiles/apply",
            {"plan_id": plan["id"], "confirmation": plan["confirmation_phrase"]},
        )
        checks["profile_apply_and_smoke"] = switched["status"] == "completed" and all(
            state == "answer" for state in switched["smoke_results"].values()
        )
        events = admin.events()
        checks["audit_events"] = all(
            name in events
            for name in (
                "service_request_cancel_requested",
                "configuration_updated",
                "evaluation_created",
                "profile_switch_completed",
            )
        )
    finally:
        server.shutdown()
        server.service.fleet.shutdown()
        server.server_close()
        thread.join(timeout=2)

    report = {
        "schema": "sparse-network-dashboard-controls-smoke.v1",
        "passed": all(checks.values()),
        "checks": checks,
    }
    output = args.output if args.output.is_absolute() else config.root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
