"""Isolated JSON-lines worker for PP-OCRv6 detection and recognition."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from time import monotonic
from typing import Any


def emit(value: dict[str, Any]) -> None:
    print(json.dumps(value, ensure_ascii=False), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detection-model", required=True)
    parser.add_argument("--recognition-model", required=True)
    parser.add_argument("--device", default="gpu:0")
    args = parser.parse_args()
    try:
        import torch  # type: ignore[import-not-found]
        from paddleocr import PaddleOCR  # type: ignore[import-not-found]

        ocr = PaddleOCR(
            text_detection_model_dir=args.detection_model,
            text_recognition_model_dir=args.recognition_model,
            engine="transformers",
            device=args.device,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
        emit(
            {
                "event": "ready",
                "torch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "device": args.device,
            }
        )
        for line in sys.stdin:
            try:
                request = json.loads(line)
                if request.get("command") == "shutdown":
                    emit({"event": "stopped"})
                    return 0
                if request.get("command") != "generate":
                    raise ValueError("unknown worker command")
                started = monotonic()
                images = [str(value) for value in request.get("images", [])]
                if not images:
                    raise ValueError("OCR request requires at least one image")
                pages: list[str] = []
                line_count = 0
                for page_number, image in enumerate(images, 1):
                    page_lines: list[str] = []
                    for result in ocr.predict(image):
                        raw = result.json
                        payload = raw.get("res", raw) if isinstance(raw, dict) else {}
                        values = payload.get("rec_texts", []) if isinstance(payload, dict) else []
                        page_lines.extend(str(value) for value in values if str(value).strip())
                    line_count += len(page_lines)
                    pages.append(f"[Page {page_number}]\n" + "\n".join(page_lines))
                text = "\n\n".join(pages).strip()
                emit(
                    {
                        "event": "result",
                        "text": text,
                        "input_tokens": 0,
                        "output_tokens": max(1, len(text.split())),
                        "finish_reason": "stop",
                        "elapsed_ms": round((monotonic() - started) * 1000),
                        "pages": len(images),
                        "lines": line_count,
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
