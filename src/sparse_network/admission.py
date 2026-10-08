"""Frozen endpoint admission suites and objective output checks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .config import AppConfig
from .controller import Controller
from .errors import ConfigurationError
from .models import ModelRegistry


@dataclass(frozen=True)
class AdmissionCase:
    id: str
    prompt: str
    images: tuple[Path, ...]
    checks: tuple[dict[str, Any], ...]


def load_cases(config: AppConfig, endpoint_id: str, suite_path: Path) -> list[AdmissionCase]:
    suite = yaml.safe_load(suite_path.read_text(encoding="utf-8")) or {}
    endpoints = suite.get("endpoints", {})
    raw_cases = endpoints.get(endpoint_id)
    if not isinstance(raw_cases, list):
        raise ConfigurationError(f"Admission suite has no cases for {endpoint_id}")
    cases: list[AdmissionCase] = []
    for raw in raw_cases:
        if not isinstance(raw, dict):
            raise ConfigurationError(f"Invalid admission case for {endpoint_id}")
        images = tuple(
            (config.root / str(value)).resolve(strict=False) for value in raw.get("images", [])
        )
        checks = raw.get("checks", [])
        if not isinstance(checks, list):
            raise ConfigurationError(f"Admission checks must be a list: {raw.get('id')}")
        cases.append(
            AdmissionCase(
                id=str(raw["id"]),
                prompt=str(raw["prompt"]),
                images=images,
                checks=tuple(check for check in checks if isinstance(check, dict)),
            )
        )
    return cases


def evaluate_answer(answer: str, checks: tuple[dict[str, Any], ...]) -> list[dict[str, Any]]:
    outcomes: list[dict[str, Any]] = []
    lowered = answer.casefold().replace("\\", "")
    for check in checks:
        if "contains_all" in check:
            expected = [str(value) for value in check["contains_all"]]
            passed = all(value.casefold() in lowered for value in expected)
            outcomes.append({"check": "contains_all", "expected": expected, "passed": passed})
        elif "contains_any" in check:
            expected = [str(value) for value in check["contains_any"]]
            passed = any(value.casefold() in lowered for value in expected)
            outcomes.append({"check": "contains_any", "expected": expected, "passed": passed})
        elif "maximum_words" in check:
            maximum = int(check["maximum_words"])
            count = len(answer.split())
            outcomes.append(
                {
                    "check": "maximum_words",
                    "maximum": maximum,
                    "actual": count,
                    "passed": count <= maximum,
                }
            )
        elif "json_keys" in check:
            expected = [str(value) for value in check["json_keys"]]
            try:
                decoded = json.loads(answer)
            except json.JSONDecodeError:
                decoded = None
            passed = isinstance(decoded, dict) and all(key in decoded for key in expected)
            outcomes.append({"check": "json_keys", "expected": expected, "passed": passed})
        else:
            outcomes.append({"check": "unknown", "passed": False})
    return outcomes


def run_admission(
    config: AppConfig,
    registry: ModelRegistry,
    *,
    endpoint_id: str,
    suite_path: Path,
    case_ids: set[str] | None = None,
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    selected = [
        case
        for case in load_cases(config, endpoint_id, suite_path)
        if case_ids is None or case.id in case_ids
    ]
    if not selected:
        raise ConfigurationError("No admission cases selected")
    results: list[dict[str, Any]] = []
    for case in selected:
        smoke = Controller(config, registry).smoke(
            endpoint_id=endpoint_id,
            prompt=case.prompt,
            images=case.images,
            timeout_seconds=timeout_seconds,
        )
        checks = evaluate_answer(smoke.answer or "", case.checks)
        passed = smoke.envelope.status == "answer" and all(
            bool(check["passed"]) for check in checks
        )
        results.append(
            {
                "case": case.id,
                "passed": passed,
                "checks": checks,
                "result": smoke.to_dict(),
            }
        )
    return {
        "schema": "sparse-network-admission-result.v1",
        "endpoint": endpoint_id,
        "passed": all(bool(result["passed"]) for result in results),
        "cases": results,
    }
