"""Load and smoke a configured resident roster for hardware-tier profiling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sparse_network.config import load_config
from sparse_network.fleet import FleetManager
from sparse_network.models import ModelRegistry


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config.local.yaml"))
    parser.add_argument("--roster", type=Path, required=True)
    parser.add_argument("--smoke-endpoint", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(local_config=args.config)
    manager = FleetManager(
        config,
        ModelRegistry.load(config),
        roster_path=args.roster,
        maximum_parallel_generations=1,
    )
    try:
        loaded = manager.load_all()
        smoke = manager.submit(
            endpoint_id=args.smoke_endpoint,
            prompt="Reply with exactly: ROSTER_OK",
            timeout_seconds=120,
        )
        final = manager.status()
        payload = {
            "schema": "sparse-network-roster-profile.v1",
            "passed": smoke.answer == "ROSTER_OK",
            "roster": str(args.roster),
            "loaded": loaded,
            "smoke": smoke.to_dict(),
            "final": final,
        }
    finally:
        manager.shutdown()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
