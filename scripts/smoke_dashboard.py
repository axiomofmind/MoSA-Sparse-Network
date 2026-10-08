"""Run the Milestone 12 dashboard against the portable mock controller."""

from __future__ import annotations

import argparse
import json
import threading
from pathlib import Path
from time import monotonic, sleep
from typing import Any
from urllib.error import HTTPError
from urllib.request import urlopen

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
    raise TimeoutError(f"Dashboard smoke request did not finish: {request_id}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/mock.yaml"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".sparse-data/mock/runs/milestone-12-dashboard.json"),
    )
    args = parser.parse_args()

    token = "milestone-12-smoke-token"
    config = load_config(local_config=args.config)
    server = build_server(config, ModelRegistry.load(config), port=0, token=token)
    server.service.start()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"
    client = ControllerClient(base_url, token)

    checks: dict[str, bool] = {}
    try:
        with urlopen(base_url + "/v1/health", timeout=5) as response:  # noqa: S310
            checks["health"] = json.loads(response.read())["ok"] is True

        with urlopen(base_url + "/dashboard/", timeout=5) as response:  # noqa: S310
            html = response.read().decode("utf-8")
            checks["application_shell"] = "Sparse — Local AI workspace" in html and "app.js" in html
            checks["use_case_navigation"] = all(
                value in html
                for value in ('data-view="home"', 'data-view="work"', 'data-view="settings"')
            )
            checks["content_security_policy"] = bool(
                response.headers.get("Content-Security-Policy")
            )

        try:
            urlopen(base_url + "/v1/dashboard", timeout=5)  # noqa: S310
        except HTTPError as exc:
            checks["authentication"] = exc.code == 401
        else:
            checks["authentication"] = False

        created = client.submit("mock-echo", "milestone 12 private sentinel")
        completed = _wait_for_request(client, str(created["id"]))
        snapshot = client.request("GET", "/v1/dashboard")
        profiles = client.request("GET", "/v1/profiles")
        events = client.events()
        serialized = json.dumps(snapshot)

        dashboard_request = next(
            item for item in snapshot["requests"] if item["id"] == created["id"]
        )
        checks.update(
            {
                "mock_request": completed["state"] == "completed",
                "role_aware_session": snapshot["session"]["role"] == "administrator"
                and snapshot["session"]["read_only"] is False
                and "profile:apply" in snapshot["session"]["permissions"],
                "prompt_redaction": dashboard_request["prompt"] == "[redacted]",
                "answer_redaction": dashboard_request["result"]["answer"] == "[redacted]",
                "portable_profiles": any(
                    profile["selection_key"] == "16" for profile in profiles["profiles"]
                )
                and any(
                    profile["selection_key"] == "24" for profile in profiles["profiles"]
                )
                and any(
                    profile["selection_key"] == "32" for profile in profiles["profiles"]
                ),
                "mock_profile": profiles["recommended_selection"] == "mock",
                "use_case_catalog": len(
                    snapshot.get("use_cases", {}).get("catalog", {}).get("templates", [])
                )
                == 8,
                "event_replay": "service_request_finished" in events,
                "no_host_paths": str(config.root) not in serialized,
            }
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        disconnected_result = server.service.fleet.submit(
            endpoint_id="mock-echo",
            prompt="dashboard disconnected sentinel",
            timeout_seconds=5,
        )
        checks["dashboard_disconnect_isolated"] = (
            disconnected_result.envelope.status == "answer"
        )
        server.service.fleet.shutdown()

    report = {
        "schema": "sparse-network-dashboard-smoke.v1",
        "passed": all(checks.values()),
        "checks": checks,
    }
    output = args.output
    if not output.is_absolute():
        output = config.root / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
