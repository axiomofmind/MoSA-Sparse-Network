"""Trusted host and GPU telemetry."""

from __future__ import annotations

import csv
import io
import shutil
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from time import monotonic

import psutil


def _nvidia_smi() -> str | None:
    return shutil.which("nvidia-smi")


def gpu_memory_used_bytes(index: int | None) -> int:
    executable = _nvidia_smi()
    if executable is None or index is None:
        return 0
    try:
        completed = subprocess.run(
            [
                executable,
                f"--id={index}",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        mebibytes = int(completed.stdout.strip().splitlines()[0])
        return mebibytes * 1024 * 1024
    except (OSError, ValueError, subprocess.SubprocessError, IndexError):
        return 0


def process_tree_rss_bytes(pid: int | None) -> int:
    if pid is None:
        return 0
    try:
        process = psutil.Process(pid)
        processes = [process, *process.children(recursive=True)]
        return sum(item.memory_info().rss for item in processes if item.is_running())
    except (psutil.Error, OSError):
        return 0


@dataclass(frozen=True)
class TelemetrySnapshot:
    elapsed_ms: int
    peak_vram_bytes: int
    peak_ram_bytes: int
    baseline_vram_bytes: int
    final_vram_bytes: int


class ResourceSampler:
    """Periodically sample process RAM and physical GPU memory."""

    def __init__(self, *, pid: int | None, gpu_index: int | None, interval: float) -> None:
        self.pid = pid
        self.gpu_index = gpu_index
        self.interval = max(0.05, interval)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._start = 0.0
        self._baseline_vram = 0
        self._peak_vram = 0
        self._peak_ram = 0

    def start(self) -> None:
        self._start = monotonic()
        self._baseline_vram = gpu_memory_used_bytes(self.gpu_index)
        self._peak_vram = self._baseline_vram
        self._peak_ram = process_tree_rss_bytes(self.pid)
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def _sample_loop(self) -> None:
        while not self._stop.wait(self.interval):
            self._peak_vram = max(self._peak_vram, gpu_memory_used_bytes(self.gpu_index))
            self._peak_ram = max(self._peak_ram, process_tree_rss_bytes(self.pid))

    def stop(self) -> TelemetrySnapshot:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval * 3))
        final_vram = gpu_memory_used_bytes(self.gpu_index)
        self._peak_vram = max(self._peak_vram, final_vram)
        self._peak_ram = max(self._peak_ram, process_tree_rss_bytes(self.pid))
        return TelemetrySnapshot(
            elapsed_ms=round((monotonic() - self._start) * 1000),
            peak_vram_bytes=self._peak_vram,
            peak_ram_bytes=self._peak_ram,
            baseline_vram_bytes=self._baseline_vram,
            final_vram_bytes=final_vram,
        )


def query_nvidia_gpus() -> list[dict[str, int | str]]:
    executable = _nvidia_smi()
    if executable is None:
        return []
    try:
        completed = subprocess.run(
            [
                executable,
                "--query-gpu=index,name,memory.total,memory.free,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    rows: list[dict[str, int | str]] = []
    for row in csv.reader(io.StringIO(completed.stdout)):
        if len(row) != 5:
            continue
        rows.append(
            {
                "index": int(row[0].strip()),
                "name": row[1].strip(),
                "memory_total_mib": int(row[2].strip()),
                "memory_free_mib": int(row[3].strip()),
                "driver_version": row[4].strip(),
            }
        )
    return rows


def disk_free_bytes(path: Path) -> int:
    anchor = path
    while not anchor.exists() and anchor.parent != anchor:
        anchor = anchor.parent
    return shutil.disk_usage(anchor).free
