"""Isolated JSON-lines worker for Transformers text-generation endpoints."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from contextlib import redirect_stdout
from time import monotonic
from typing import Any


def emit(value: dict[str, Any]) -> None:
    print(json.dumps(value, ensure_ascii=False), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--memory-fraction", type=float, default=0.75)
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--vision", action="store_true")
    args = parser.parse_args()
    try:
        import torch  # type: ignore[import-not-found]
        import transformers  # type: ignore[import-not-found]
        from transformers import (
            AutoModelForCausalLM,
            AutoModelForImageTextToText,
            AutoProcessor,
            AutoTokenizer,
        )

        # Some model processors emit schema diagnostics to stdout during loading.
        # Keep stdout reserved for this worker's JSON-lines protocol.
        transformers.logging.set_verbosity(60)

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available to the Transformers worker")
        device_index = int(args.device.rsplit(":", maxsplit=1)[-1])
        torch.cuda.set_device(device_index)
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device_index)
        processor = None
        with redirect_stdout(sys.stderr):
            if args.vision:
                processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
                model = AutoModelForImageTextToText.from_pretrained(
                    args.model,
                    local_files_only=True,
                    dtype="auto",
                    device_map={"": args.device},
                    low_cpu_mem_usage=True,
                )
                tokenizer = processor.tokenizer
            else:
                tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
                model = AutoModelForCausalLM.from_pretrained(
                    args.model,
                    local_files_only=True,
                    dtype="auto",
                    device_map={"": args.device},
                    low_cpu_mem_usage=True,
                )
        model.eval()
        emit(
            {
                "event": "ready",
                "torch_version": torch.__version__,
                "transformers_version": transformers.__version__,
                "device": torch.cuda.get_device_name(device_index),
                "allocated_bytes": torch.cuda.memory_allocated(device_index),
                "reserved_bytes": torch.cuda.memory_reserved(device_index),
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
                image_paths = [str(value) for value in request.get("images", [])]
                if args.vision:
                    if processor is None or not image_paths:
                        raise ValueError("vision generation requires at least one image")
                    content = [
                        {"type": "image", "url": image_path}
                        for image_path in image_paths
                    ]
                    content.append({"type": "text", "text": str(request["prompt"])})
                    inputs = processor.apply_chat_template(
                        [{"role": "user", "content": content}],
                        tokenize=True,
                        add_generation_prompt=True,
                        return_dict=True,
                        return_tensors="pt",
                    ).to(args.device)
                    inputs.pop("token_type_ids", None)
                else:
                    messages = [{"role": "user", "content": str(request["prompt"])}]
                    rendered = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=args.thinking,
                    )
                    inputs = tokenizer(rendered, return_tensors="pt").to(args.device)
                input_tokens = int(inputs["input_ids"].shape[-1])
                with torch.inference_mode():
                    output = model.generate(
                        **inputs,
                        max_new_tokens=int(request["max_tokens"]),
                        do_sample=False,
                        use_cache=True,
                    )
                generated = output[0, input_tokens:]
                text = tokenizer.decode(generated, skip_special_tokens=True)
                output_tokens = int(generated.shape[-1])
                del generated, output, inputs
                torch.cuda.empty_cache()
                emit(
                    {
                        "event": "result",
                        "text": text,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "finish_reason": "stop",
                        "elapsed_ms": round((monotonic() - started) * 1000),
                        "allocated_bytes": torch.cuda.memory_allocated(device_index),
                        "reserved_bytes": torch.cuda.memory_reserved(device_index),
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
