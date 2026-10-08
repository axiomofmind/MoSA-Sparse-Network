"""Isolated JSON-lines worker for typed d1 decisions."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from time import monotonic
from typing import Any


def emit(value: dict[str, Any]) -> None:
    print(json.dumps(value, ensure_ascii=False), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--memory-fraction", type=float, default=0.4)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()
    try:
        import torch  # type: ignore[import-not-found]
        import transformers  # type: ignore[import-not-found]
        from PIL import Image
        from transformers import AutoModel

        if args.device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available to the decision worker")
            device_index = int(args.device.rsplit(":", maxsplit=1)[-1])
            torch.cuda.set_device(device_index)
            torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device_index)
            dtype = torch.bfloat16
            device_name = torch.cuda.get_device_name(device_index)
        else:
            dtype = torch.float32
            device_name = args.device
        model = AutoModel.from_pretrained(
            args.model,
            local_files_only=True,
            trust_remote_code=args.trust_remote_code,
            dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(args.device)
        model.eval()
        emit(
            {
                "event": "ready",
                "device": device_name,
                "torch_version": torch.__version__,
                "transformers_version": transformers.__version__,
            }
        )
        for line in sys.stdin:
            try:
                request = json.loads(line)
                if request.get("command") == "shutdown":
                    emit({"event": "stopped"})
                    return 0
                if request.get("command") != "decide":
                    raise ValueError("unknown worker command")
                questions = request.get("questions")
                if not isinstance(questions, dict) or not questions:
                    raise ValueError("decision request has no questions")
                image_paths = [Path(str(value)) for value in request.get("images", [])]
                images = []
                for path in image_paths:
                    with Image.open(path) as source:
                        images.append(source.convert("RGB").copy())
                started = monotonic()
                with torch.inference_mode():
                    result = model.system_one(request.get("state"), questions, images)
                emit(
                    {
                        "event": "decision",
                        "result": result,
                        "elapsed_ms": round((monotonic() - started) * 1000),
                    }
                )
            except Exception as exc:
                emit(
                    {
                        "event": "error",
                        "error": str(exc),
                        "traceback": traceback.format_exc(limit=12),
                    }
                )
    except Exception as exc:
        emit(
            {
                "event": "startup_error",
                "error": str(exc),
                "traceback": traceback.format_exc(limit=20),
            }
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
