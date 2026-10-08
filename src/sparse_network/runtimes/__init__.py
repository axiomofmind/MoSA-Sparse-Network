"""Runtime adapter implementations."""

from .base import RuntimeAdapter, RuntimeResult, RuntimeState
from .llama_cpp import LlamaCppAdapter
from .mock import MockAdapter
from .paddle_ocr import PaddleOcrAdapter
from .transformers import TransformersAdapter

__all__ = [
    "LlamaCppAdapter",
    "MockAdapter",
    "PaddleOcrAdapter",
    "RuntimeAdapter",
    "RuntimeResult",
    "RuntimeState",
    "TransformersAdapter",
]
