"""Isolated JSON-lines worker for text-only EmbeddingGemma 2 retrieval."""

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
    parser.add_argument("--dimension", type=int, required=True)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--query-prompt", default="SearchQuery")
    parser.add_argument("--document-prompt", default="Document")
    parser.add_argument("--text-only", action="store_true")
    args = parser.parse_args()
    try:
        import torch  # type: ignore[import-not-found]
        import transformers  # type: ignore[import-not-found]
        from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]

        if args.threads > 0:
            torch.set_num_threads(args.threads)
        config_kwargs: dict[str, Any] = {}
        if args.text_only:
            config_kwargs = {"vision_config": None, "audio_config": None}
        model = SentenceTransformer(
            args.model,
            device="cpu",
            model_kwargs={"torch_dtype": torch.float32},
            config_kwargs=config_kwargs,
            local_files_only=True,
            trust_remote_code=False,
        )
        model.max_seq_length = args.max_length
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
                prompt_name = args.query_prompt if mode == "query" else args.document_prompt
                started = monotonic()
                vectors = model.encode(
                    texts,
                    prompt_name=prompt_name or None,
                    batch_size=args.batch_size,
                    truncate_dim=args.dimension,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
                if int(vectors.shape[1]) != args.dimension:
                    raise ValueError(
                        f"configured dimension {args.dimension} does not match "
                        f"model output {vectors.shape[1]}"
                    )
                emit(
                    {
                        "event": "embeddings",
                        "vectors": vectors.tolist(),
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
