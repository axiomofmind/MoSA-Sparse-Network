"""Benchmark frozen retrieval cases against one embedding endpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import monotonic

import yaml

from sparse_network.config import load_config
from sparse_network.models import ModelRegistry
from sparse_network.retrieval import create_vector_index
from sparse_network.telemetry import process_tree_rss_bytes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("endpoint")
    parser.add_argument(
        "--suite",
        type=Path,
        default=Path("configs/admission/milestone5-retrieval.yaml"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = load_config()
    registry = ModelRegistry.load(config)
    suite_path = args.suite if args.suite.is_absolute() else config.root / args.suite
    suite = yaml.safe_load(suite_path.read_text(encoding="utf-8"))
    backend, index = create_vector_index(config, registry, args.endpoint)
    load_started = monotonic()
    backend.start()
    load_ms = round((monotonic() - load_started) * 1000)
    worker_pid = backend.pid
    peak_process_ram_bytes = process_tree_rss_bytes(worker_pid)
    try:
        indexing_started = monotonic()
        indexed = []
        for document in suite["documents"]:
            path = Path(document["path"])
            if not path.is_absolute():
                path = config.root / path
            indexed.append(index.ingest(path))
            peak_process_ram_bytes = max(
                peak_process_ram_bytes, process_tree_rss_bytes(worker_pid)
            )
        indexing_ms = round((monotonic() - indexing_started) * 1000)
        cases = []
        for case in suite["queries"]:
            result = index.search(str(case["text"]), top_k=3)
            peak_process_ram_bytes = max(
                peak_process_ram_bytes, process_tree_rss_bytes(worker_pid)
            )
            top_source = result.hits[0].source_name if result.hits else None
            cases.append(
                {
                    "id": case["id"],
                    "query": case["text"],
                    "expected_document": case["expected_document"],
                    "top_document": top_source,
                    "passed_at_1": top_source == case["expected_document"],
                    "elapsed_ms": result.elapsed_ms,
                    "evidence_references": list(result.evidence_references),
                    "hits": [hit.to_dict() for hit in result.hits],
                }
            )
    finally:
        backend.stop()
    passed = sum(1 for case in cases if case["passed_at_1"])
    value = {
        "schema": "sparse-network-retrieval-benchmark.v1",
        "endpoint": args.endpoint,
        "revision": backend.revision,
        "dimension": backend.dimension,
        "index_id": index.index_id,
        "load_ms": load_ms,
        "indexing_ms": indexing_ms,
        "peak_process_ram_bytes": peak_process_ram_bytes,
        "documents": indexed,
        "recall_at_1": passed / len(cases),
        "passed": passed,
        "total": len(cases),
        "cases": cases,
    }
    rendered = json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True)
    if args.output is not None:
        output = args.output if args.output.is_absolute() else config.root / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
