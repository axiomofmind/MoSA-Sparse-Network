"""Isolated JSON-lines worker for CPU text embeddings."""

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
    parser.add_argument("--model", required=True)
    parser.add_argument("--pooling", choices=("last_token", "mean"), required=True)
    parser.add_argument("--dimension", type=int, required=True)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--query-instruction", default="")
    args = parser.parse_args()
    try:
        import torch  # type: ignore[import-not-found]
        import torch.nn.functional as functional  # type: ignore[import-not-found]
        import transformers  # type: ignore[import-not-found]
        from transformers import AutoModel, AutoTokenizer

        if args.threads > 0:
            torch.set_num_threads(args.threads)
        tokenizer = AutoTokenizer.from_pretrained(
            args.model,
            local_files_only=True,
            padding_side="left" if args.pooling == "last_token" else "right",
        )
        model = AutoModel.from_pretrained(
            args.model,
            local_files_only=True,
            dtype="auto",
            device_map={"": "cpu"},
            low_cpu_mem_usage=True,
        )
        model.eval()
        emit(
            {
                "event": "ready",
                "dimension": args.dimension,
                "device": "cpu",
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
                if request.get("command") != "embed":
                    raise ValueError("unknown worker command")
                mode = str(request.get("mode", "passage"))
                texts = [str(value) for value in request.get("texts", [])]
                if not texts:
                    raise ValueError("embedding request has no texts")
                if mode == "query" and args.query_instruction:
                    texts = [f"{args.query_instruction}{text}" for text in texts]
                started = monotonic()
                inputs = tokenizer(
                    texts,
                    max_length=args.max_length,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
                with torch.inference_mode():
                    outputs = model(**inputs)
                    hidden = outputs.last_hidden_state
                    mask = inputs["attention_mask"]
                    if args.pooling == "last_token":
                        if bool((mask[:, -1].sum() == mask.shape[0]).item()):
                            pooled = hidden[:, -1]
                        else:
                            sequence_lengths = mask.sum(dim=1) - 1
                            rows = torch.arange(hidden.shape[0])
                            pooled = hidden[rows, sequence_lengths]
                    else:
                        expanded = mask.unsqueeze(-1).expand(hidden.size()).float()
                        pooled = (hidden * expanded).sum(dim=1) / expanded.sum(dim=1).clamp(
                            min=1e-9
                        )
                    normalized = functional.normalize(pooled.float(), p=2, dim=1)
                if int(normalized.shape[1]) != args.dimension:
                    raise ValueError(
                        f"configured dimension {args.dimension} does not match "
                        f"model output {normalized.shape[1]}"
                    )
                emit(
                    {
                        "event": "embeddings",
                        "vectors": normalized.tolist(),
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
