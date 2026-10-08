"""Measure a resident endpoint with a substantial synthetic working context."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sparse_network.config import load_config
from sparse_network.fleet import FleetManager
from sparse_network.models import ModelRegistry


def _prompt(target_characters: int) -> str:
    lines: list[str] = []
    index = 1
    total = 0
    while total < target_characters:
        line = f"Evidence item {index}: local processing remains enabled for case {index}."
        lines.append(line)
        total += len(line) + 1
        index += 1
    body = "\n".join(lines)
    instruction = "\n\nIgnore repetition in the evidence. Reply with exactly: CONTEXT_OK"
    return body[: max(1, target_characters - len(instruction))] + instruction


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config.local.yaml"))
    parser.add_argument("--roster", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--target-characters", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.target_characters <= 0:
        parser.error("--target-characters must be positive")

    config = load_config(local_config=args.config)
    manager = FleetManager(
        config,
        ModelRegistry.load(config),
        roster_path=args.roster,
        maximum_parallel_generations=1,
    )
    prompt = _prompt(args.target_characters)
    try:
        loaded = manager.load_all()
        result = manager.submit(
            endpoint_id=args.endpoint,
            prompt=prompt,
            timeout_seconds=180,
        )
        final = manager.status()
        payload = {
            "schema": "sparse-network-context-profile.v1",
            "passed": result.envelope.status == "answer"
            and "CONTEXT_OK" in (result.answer or ""),
            "roster": manager.roster_id,
            "endpoint": args.endpoint,
            "context_size": manager.entries[args.endpoint].endpoint.context_size,
            "max_output_tokens": manager.entries[args.endpoint].endpoint.max_output_tokens,
            "prompt_characters": len(prompt),
            "input_tokens": result.envelope.resource_usage.input_tokens,
            "output_tokens": result.envelope.resource_usage.output_tokens,
            "peak_vram_bytes": result.envelope.resource_usage.peak_vram_bytes,
            "loaded": loaded,
            "result": result.to_dict(),
            "final": final,
        }
    finally:
        manager.shutdown()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                key: payload[key]
                for key in (
                    "passed",
                    "roster",
                    "endpoint",
                    "context_size",
                    "max_output_tokens",
                    "prompt_characters",
                    "input_tokens",
                    "output_tokens",
                    "peak_vram_bytes",
                )
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
