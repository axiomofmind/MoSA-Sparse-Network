"""CPU exact-repetition detection derived from AntiDoom's Apache-2.0 detector.

This file adapts ``antidoom.repetition`` from the local AntiDoom project. The
configuration wrapper, diagnostic prefix, and streaming policy integration are
Sparse Network modifications. See THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .errors import RequestFailedError


@dataclass(frozen=True)
class RepeatHit:
    start: int
    end: int
    period: int
    repeats: int
    snippet: str

    @property
    def repeat_start(self) -> int:
        return self.start + self.period

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RepetitionPolicy:
    min_repeats: int = 4
    max_period: int = 1024
    min_period: int = 1
    min_total_repeated: int = 60
    sample_len: int = 16
    sample_interval: int = 128
    streaming_check_interval_characters: int = 128
    maximum_retries: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Any) -> RepetitionPolicy:
        if not isinstance(value, dict):
            return cls()
        policy = cls(
            min_repeats=int(value.get("min_repeats", 4)),
            max_period=int(value.get("max_period", 1024)),
            min_period=int(value.get("min_period", 1)),
            min_total_repeated=int(value.get("min_total_repeated", 60)),
            sample_len=int(value.get("sample_len", 16)),
            sample_interval=int(value.get("sample_interval", 128)),
            streaming_check_interval_characters=int(
                value.get("streaming_check_interval_characters", 128)
            ),
            maximum_retries=int(value.get("maximum_retries", 1)),
        )
        if (
            policy.min_repeats < 2
            or policy.min_period < 1
            or policy.max_period < policy.min_period
            or policy.min_total_repeated < 1
            or policy.sample_len < 1
            or policy.sample_interval < 1
            or policy.streaming_check_interval_characters < 1
            or not 0 <= policy.maximum_retries <= 2
        ):
            raise ValueError("Invalid repetition policy")
        return policy


class RepetitionDetectedError(RequestFailedError):
    def __init__(self, text: str, hit: RepeatHit, *, streaming: bool) -> None:
        super().__init__(
            f"Exact repetition detected at character {hit.start} with period "
            f"{hit.period} for {hit.repeats} repeats"
        )
        self.hit = hit
        self.diagnostic_prefix = text[: hit.start]
        self.streaming = streaming


def _verify_repetition_at(
    text: str,
    start_pos: int,
    period: int,
    min_repeats: int,
    min_total_repeated: int,
) -> tuple[bool, RepeatHit | None]:
    if period < 1 or start_pos < 0 or start_pos + period > len(text):
        return False, None
    pattern = text[start_pos : start_pos + period]
    reps = 0
    pos = start_pos
    while pos + period <= len(text) and text[pos : pos + period] == pattern:
        reps += 1
        pos += period
    end_pos = pos
    pos = start_pos - period
    while pos >= 0 and text[pos : pos + period] == pattern:
        reps += 1
        start_pos = pos
        pos -= period
    if reps >= min_repeats and reps * period >= min_total_repeated:
        snippet = pattern if len(pattern) <= 100 else pattern[:100] + "..."
        return True, RepeatHit(start_pos, end_pos, period, reps, snippet)
    return False, None


def find_inner_repetition(
    text: str,
    *,
    min_repeats: int = 4,
    max_period: int = 1024,
    min_period: int = 1,
    min_total_repeated: int = 60,
    sample_len: int = 16,
    sample_interval: int = 128,
) -> tuple[bool, RepeatHit | None]:
    """Return the first exact inner repetition using AntiDoom's fingerprint scan."""
    if not text or len(text) < min_total_repeated:
        return False, None
    for sample_pos in range(0, len(text) - sample_len, sample_interval):
        fingerprint = text[sample_pos : sample_pos + sample_len]
        other_pos = text.find(fingerprint, sample_pos + sample_len)
        if other_pos != -1:
            period = other_pos - sample_pos
            if min_period <= period <= max_period:
                found, hit = _verify_repetition_at(
                    text, sample_pos, period, min_repeats, min_total_repeated
                )
                if found:
                    return True, hit
        other_pos = text.rfind(fingerprint, 0, sample_pos)
        if other_pos != -1:
            period = sample_pos - other_pos
            if min_period <= period <= max_period:
                found, hit = _verify_repetition_at(
                    text, other_pos, period, min_repeats, min_total_repeated
                )
                if found:
                    return True, hit
    return False, None


def detect_repetition(text: str, policy: RepetitionPolicy) -> RepeatHit | None:
    found, hit = find_inner_repetition(
        text,
        min_repeats=policy.min_repeats,
        max_period=policy.max_period,
        min_period=policy.min_period,
        min_total_repeated=policy.min_total_repeated,
        sample_len=policy.sample_len,
        sample_interval=policy.sample_interval,
    )
    return hit if found else None
