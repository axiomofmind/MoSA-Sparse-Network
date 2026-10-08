from __future__ import annotations

import re
from pathlib import Path

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
