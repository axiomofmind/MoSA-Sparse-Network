"""Run repeatable real-model top-2 and MoSA workflow smoke cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import monotonic

from sparse_network.config import load_config
from sparse_network.execution import ExecutionRequest
from sparse_network.fleet import FleetManager
from sparse_network.models import ModelRegistry
from sparse_network.routing import RouteRequest, StaticRouter
from sparse_network.workflows import WorkflowExecutor, WorkflowRequest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config.local.yaml"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".sparse-data/runs/workflow-smoke.json"),
    )
    arguments = parser.parse_args()
    config = load_config(local_config=arguments.config)
    registry = ModelRegistry.load(config)
    manager = FleetManager(config, registry)
    router = StaticRouter(config, registry, resident_endpoint_ids=set(manager.entries))
    workflows = WorkflowExecutor(manager, router)
    cases = (
        ("top-2", "disputed", "general_generation", "TOP2_READY"),
        ("mosa", "maximum_quality", "maximum_quality", "MOSA_READY"),
    )
    required = sorted(
        {
            endpoint
            for mode, _trigger, lane, _sentinel in cases
            for endpoint in (workflows.top2 if mode == "top-2" else workflows.mosa)[lane]
        }
    )
    started = monotonic()
    results: list[dict[str, object]] = []
    try:
        manager.load_endpoints(required)
        for mode, trigger, lane, sentinel in cases:
            result = workflows.run(
                WorkflowRequest(
                    execution=ExecutionRequest(
                        route=RouteRequest(
                            prompt=f"Reply with exactly: {sentinel}",
                            explicit_lane=lane,
                        ),
                        expected_contains=sentinel,
                    ),
                    mode=mode,
                    trigger=trigger,
                )
            )
            results.append(result.to_dict())
        payload = {
            "schema": "sparse-network-workflow-smoke.v1",
            "passed": all(result["status"] == "accepted" for result in results),
            "required_endpoints": required,
            "elapsed_ms": round((monotonic() - started) * 1000),
            "results": results,
        }
    finally:
        manager.shutdown()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
