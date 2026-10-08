"""Aggregate coordination quality and cost metrics from workflow trace JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sparse_network.evaluation import evaluate_coordination_results


def _read_trace(path: Path) -> dict[str, Any] | None:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"workflow trace must be an object: {path}")
    if value.get("schema") not in {
        "sparse-network-workflow-trace.v1",
        "sparse-network-workflow-result.v1",
    }:
        return None
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="Trace files or directories")
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    arguments = parser.parse_args()
    trace_paths: list[tuple[Path, bool]] = []
    for path in arguments.paths:
        if path.is_dir():
            trace_paths.extend((candidate, False) for candidate in sorted(path.glob("*.json")))
        else:
            trace_paths.append((path, True))
    traces: list[dict[str, Any]] = []
    for path, strict in trace_paths:
        trace = _read_trace(path)
        if trace is None:
            if strict:
                raise ValueError(f"unsupported workflow trace schema: {path}")
            continue
        traces.append(trace)
    report = evaluate_coordination_results(traces)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
