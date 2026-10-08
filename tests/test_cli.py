from __future__ import annotations

from pathlib import Path

from sparse_network.cli import main


def test_doctor_runs_without_mutation(capsys: object) -> None:
    exit_code = main(["doctor", "--json"])
    assert exit_code == 0
    captured = capsys.readouterr()  # type: ignore[attr-defined]
    assert '"mutated": false' in captured.out


def test_mock_smoke_cli(capsys: object) -> None:
    exit_code = main(["smoke", "mock-echo", "--prompt", "cli", "--json"])
    assert exit_code == 0
    captured = capsys.readouterr()  # type: ignore[attr-defined]
    assert '"status": "answer"' in captured.out
    assert '"runtime_state": "stopped"' in captured.out


def test_pull_all_dry_run_uses_canonical_smallest_first(
    tmp_path: Path, capsys: object
) -> None:
    exit_code = main(
        [
            "models",
            "pull-all",
            "--cache-dir",
            str(tmp_path / "models"),
            "--dry-run",
            "--json",
        ]
    )
    assert exit_code == 0
    output = capsys.readouterr().out  # type: ignore[attr-defined]
    assert output.index('"endpoint": "minilm-l6-v2"') < output.index(
        '"endpoint": "lfm25-1.2b"'
    )
    assert output.index('"endpoint": "qwen38-27b-q4"') < output.index(
        '"endpoint": "qwen38-27b"'
    )


def test_static_route_plan_cli(capsys: object) -> None:
    exit_code = main(
        [
            "route",
            "plan",
            "--prompt",
            "@route:classify Assign a label",
            "--json",
        ]
    )
    assert exit_code == 0
    captured = capsys.readouterr()  # type: ignore[attr-defined]
    assert '"lane": "classification"' in captured.out
    assert '"endpoint": "lfm25-1.2b"' in captured.out
    assert '"mode": "top-1"' in captured.out


def test_hash_retrieval_cli(tmp_path: Path, capsys: object) -> None:
    document = tmp_path / "note.md"
    document.write_text("The recovery code is amber-seven.", encoding="utf-8")
    config = tmp_path / "local.yaml"
    config.write_text(
        "paths:\n"
        f"  artifacts: '{(tmp_path / 'artifacts').as_posix()}'\n"
        f"  indexes: '{(tmp_path / 'indexes').as_posix()}'\n",
        encoding="utf-8",
    )
    exit_code = main(
        [
            "--config",
            str(config),
            "retrieval",
            "index",
            str(document),
            "--endpoint",
            "hash-embedding",
            "--json",
        ]
    )
    assert exit_code == 0
    assert '"chunks_added"' in capsys.readouterr().out  # type: ignore[attr-defined]
    exit_code = main(
        [
            "--config",
            str(config),
            "retrieval",
            "search",
            "--query",
            "recovery code",
            "--endpoint",
            "hash-embedding",
            "--top-k",
            "1",
            "--json",
        ]
    )
    assert exit_code == 0
    assert "amber-seven" in capsys.readouterr().out  # type: ignore[attr-defined]
