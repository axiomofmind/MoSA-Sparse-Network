from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from sparse_network.config import deep_merge, load_config


def test_deep_merge_preserves_nested_defaults() -> None:
    merged = deep_merge(
        {"paths": {"models": "default", "logs": "logs"}, "items": [1]},
        {"paths": {"models": "local"}, "items": [2]},
    )
    assert merged == {
        "paths": {"models": "local", "logs": "logs"},
        "items": [2],
    }


def test_explicit_local_configuration_overrides_defaults(tmp_path: Path) -> None:
    local = tmp_path / "local.yaml"
    local.write_text("paths:\n  model_cache: portable-test-cache\n", encoding="utf-8")
    config = load_config(local_config=local)
    assert config.paths["model_cache"] == config.root / "portable-test-cache"
    assert config.sources[-1] == local


def test_tracked_portable_configuration_has_no_machine_paths() -> None:
    root = Path(__file__).resolve().parents[1]
    tracked_config_files = [
        root / "config.example.yaml",
        *sorted((root / "configs").rglob("*.yaml")),
    ]
    for path in tracked_config_files:
        content = path.read_text(encoding="utf-8")
        assert re.search(r"[A-Za-z]:[\\/]", content) is None


@pytest.mark.parametrize(
    ("roster_name", "roster_id", "resident", "optional", "exclusive"),
    [
        (
            "reference-16gb.yaml",
            "nvidia-16gb-multimodal",
            {"lfm25-1.2b", "qwen35-4b", "nemotron-4b", "gemma-e2b-vision"},
            {"gemma4-12b-qat", "pp-ocrv6-medium", "paddleocr-vl-1.6"},
            set(),
        ),
        (
            "reference-24gb.yaml",
            "nvidia-24gb-multimodal",
            {
                "lfm25-1.2b",
                "qwen35-4b",
                "nemotron-4b",
                "gemma-e2b-vision",
                "qwen3-8b-fp8",
            },
            {"gemma4-12b-qat", "pp-ocrv6-medium", "paddleocr-vl-1.6"},
            set(),
        ),
        (
            "reference-32gb.yaml",
            "nvidia-32gb-multimodal",
            {
                "lfm25-1.2b",
                "qwen35-4b",
                "nemotron-4b",
                "gemma-e2b-vision",
                "qwen3-8b-fp8",
            },
            {"pp-ocrv6-medium", "paddleocr-vl-1.6"},
            {"qwen38-27b-q4"},
        ),
        (
            "reference-32gb-gemma12.yaml",
            "nvidia-32gb-multimodal-gemma12",
            {
                "lfm25-1.2b",
                "qwen35-4b",
                "nemotron-4b",
                "gemma-e2b-vision",
                "qwen3-8b-fp8",
                "gemma4-12b-qat",
            },
            {"pp-ocrv6-medium", "paddleocr-vl-1.6"},
            {"qwen38-27b"},
        ),
    ],
)
def test_release_profile_allocations_are_explicit_and_disjoint(
    roster_name: str,
    roster_id: str,
    resident: set[str],
    optional: set[str],
    exclusive: set[str],
) -> None:
    root = Path(__file__).resolve().parents[1]
    roster = yaml.safe_load(
        (root / "configs" / "rosters" / roster_name).read_text(encoding="utf-8")
    )

    assert roster["id"] == roster_id
    assert set(roster["cpu"]) == {"harrier-0.6b"}
    assert set(roster["resident"]) == resident
    assert set(roster.get("optional", [])) == optional
    assert set(roster.get("exclusive_swap", [])) == exclusive
    assert resident.isdisjoint(optional | exclusive)
    assert optional.isdisjoint(exclusive)
