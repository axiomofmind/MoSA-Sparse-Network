"""Run repeated real Qwen3.8-27B swap/restoration cycles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import monotonic

from sparse_network.config import load_config
from sparse_network.fleet import FleetManager
from sparse_network.models import ModelRegistry
from sparse_network.swap import SwapCoordinator, SwapRequest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config.local.yaml"))
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument(
        "--roster",
        type=Path,
        help="Optional roster whose declared exclusive-swap endpoint should be tested",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".sparse-data/runs/milestone-10-swap-cycles.json"),
    )
    args = parser.parse_args()
    if args.cycles < 2:
        raise ValueError("The restoration profile requires at least two cycles")
    config = load_config(local_config=args.config)
    manager = FleetManager(config, ModelRegistry.load(config), roster_path=args.roster)
    started = monotonic()
    results: list[dict[str, object]] = []
    try:
        initial = manager.load_all()
        coordinator = SwapCoordinator(manager)
        for index in range(args.cycles):
            result = coordinator.run(
                SwapRequest(
                    prompt=f"Reply with exactly: SWAP_CYCLE_{index + 1}",
                    expected_contains=f"SWAP_CYCLE_{index + 1}",
                    timeout_seconds=120,
                )
            )
            results.append(result.to_dict())
        final = manager.status()
        payload = {
            "schema": "sparse-network-swap-profile.v1",
            "passed": all(
                item["status"] == "answer" and item["fleet_restored"] for item in results
            ),
            "cycles": results,
            "initial": initial,
            "final": final,
            "elapsed_ms": round((monotonic() - started) * 1000),
        }
    finally:
        manager.shutdown()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
