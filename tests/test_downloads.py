from __future__ import annotations

from pathlib import Path

import yaml

from sparse_network.config import AppConfig
from sparse_network.downloads import build_download_plan, pull_download_plan
from sparse_network.models import ModelRegistry


def _registry(tmp_path: Path) -> ModelRegistry:
    definitions = {}
    for endpoint_id, size in (("large", 200), ("small", 100)):
        definitions[endpoint_id] = {
            "display_name": endpoint_id.title(),
            "role": "fixture",
            "family": "fixture",
            "source": {
                "type": "huggingface",
                "repository": f"fixture/{endpoint_id}",
                "revision": f"{endpoint_id}-revision",
                "download_size_bytes": size,
                "files": {"config": "config.json"},
            },
            "runtime": {"adapter": "mock"},
            "modalities": ["text"],
            "capabilities": ["text_generation"],
            "context_size": 128,
            "max_output_tokens": 16,
            "admission": {"state": "admitted"},
            "license": {"id": "apache-2.0", "review_required": False},
        }
    definitions["bundle"] = {
        "display_name": "Bundle",
        "role": "fixture",
        "family": "fixture",
        "source": {
            "type": "huggingface_bundle",
            "artifacts": {
                "first_config": {
                    "repository": "fixture/bundle-first",
                    "revision": "first-revision",
                    "path": "config.json",
                    "download_size_bytes": 20,
                    "files": ["config.json"],
                },
                "second_config": {
                    "repository": "fixture/bundle-second",
                    "revision": "second-revision",
                    "path": "config.json",
                    "download_size_bytes": 30,
                    "files": ["config.json"],
                },
            },
        },
        "runtime": {"adapter": "mock"},
        "modalities": ["text"],
        "capabilities": ["text_generation"],
        "context_size": 128,
        "max_output_tokens": 16,
        "admission": {"state": "admitted"},
        "license": {"id": "apache-2.0", "review_required": False},
    }
    registry_path = tmp_path / "models.yaml"
    registry_path.write_text(
        yaml.safe_dump(
            {"schema": "sparse-network-model-registry.v1", "endpoints": definitions}
        ),
        encoding="utf-8",
    )
    config = AppConfig(
        root=tmp_path,
        data={"model_registry": str(registry_path), "endpoint_overrides": {}},
        sources=(),
    )
    return ModelRegistry.load(config)


def test_download_plan_is_smallest_first(tmp_path: Path) -> None:
    plan = build_download_plan(_registry(tmp_path))
    assert [item.endpoint for item in plan.items] == ["bundle", "small", "large"]
    assert plan.total_size_bytes == 350


def test_pull_download_plan_runs_serially_and_validates(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    plan = build_download_plan(registry)
    cache = tmp_path / "cache"
    calls: list[str] = []

    def fake_download(**kwargs: object) -> str:
        repository = str(kwargs["repo_id"])
        revision = str(kwargs["revision"])
        calls.append(repository)
        snapshot = (
            cache
            / ("models--" + repository.replace("/", "--"))
            / "snapshots"
            / revision
        )
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").write_text('{"fixture": true}', encoding="utf-8")
        return str(snapshot)

    result = pull_download_plan(
        registry,
        plan,
        cache_dir=cache,
        downloader=fake_download,
    )

    assert calls == [
        "fixture/bundle-first",
        "fixture/bundle-second",
        "fixture/small",
        "fixture/large",
    ]
    assert result["passed"]
    assert [item["status"] for item in result["items"]] == [
        "downloaded",
        "downloaded",
        "downloaded",
    ]

    repeated = pull_download_plan(
        registry,
        plan,
        cache_dir=cache,
        downloader=fake_download,
    )
    assert calls == [
        "fixture/bundle-first",
        "fixture/bundle-second",
        "fixture/small",
        "fixture/large",
    ]
    assert [item["status"] for item in repeated["items"]] == [
        "already_present",
        "already_present",
        "already_present",
    ]
