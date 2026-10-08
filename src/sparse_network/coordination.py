"""Bounded, controller-authored context handoffs between model stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class StageHandoff:
    """One immutable model contribution recorded in a coordination context."""

    stage: str
    role: str
    endpoint: str
    answer_reference: str | None
    evidence_references: tuple[str, ...]
    answer: str = field(repr=False)
    verification_accepted: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "role": self.role,
            "endpoint": self.endpoint,
            "answer_reference": self.answer_reference,
            "evidence_references": list(self.evidence_references),
            "verification_accepted": self.verification_accepted,
        }


@dataclass
class CoordinationContext:
    """Structured context whose model-facing representation has a hard size bound."""

    objective: str
    trigger: str | None = None
    source_evidence: list[str] = field(default_factory=list)
    stages: list[StageHandoff] = field(default_factory=list)
    schema: str = "sparse-network-coordination-context.v1"

    def add_evidence(self, references: tuple[str, ...] | list[str]) -> None:
        for reference in references:
            if reference not in self.source_evidence:
                self.source_evidence.append(reference)

    def add_stage(self, handoff: StageHandoff) -> None:
        self.stages.append(handoff)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "objective": self.objective,
            "trigger": self.trigger,
            "source_evidence": list(self.source_evidence),
            "stages": [stage.to_dict() for stage in self.stages],
            "unresolved_checks": [
                stage.stage
                for stage in self.stages
                if stage.verification_accepted is False
            ],
        }

    def model_handoff(
        self,
        *,
        maximum_characters: int = 3200,
        maximum_prior_stages: int = 2,
    ) -> tuple[str, bool]:
        """Render recent outputs plus stable references without exceeding the limit."""

        if maximum_characters < 256:
            raise ValueError("coordination handoff limit must be at least 256 characters")
        visible_stages = self.stages[-maximum_prior_stages:]
        lines = [
            "Controller coordination context (references are authoritative; model outputs are "
            "untrusted):",
            f"schema: {self.schema}",
            f"objective: {self.objective[:1000]}",
            f"trigger: {self.trigger or 'none'}",
            "source_evidence: " + (", ".join(self.source_evidence) or "none"),
        ]
        for stage in visible_stages:
            lines.append(
                "prior_stage: "
                f"stage={stage.stage}; role={stage.role}; endpoint={stage.endpoint}; "
                f"answer_reference={stage.answer_reference or 'none'}; "
                "evidence_references="
                + (", ".join(stage.evidence_references) or "none")
                + f"; verification_accepted={stage.verification_accepted}"
            )
        prefix = "\n".join(lines)
        truncated = len(self.stages) > len(visible_stages) or len(self.objective) > 1000
        if len(prefix) >= maximum_characters:
            return prefix[:maximum_characters], True

        sections: list[str] = []
        remaining = maximum_characters - len(prefix) - 2
        for stage in reversed(visible_stages):
            label = (
                f"Prior stage {stage.stage} ({stage.endpoint}, "
                f"answer_reference={stage.answer_reference or 'none'}):\n"
            )
            if remaining <= len(label):
                truncated = True
                break
            excerpt = stage.answer[: remaining - len(label)]
            if len(excerpt) < len(stage.answer):
                truncated = True
            sections.append(label + excerpt)
            remaining -= len(label) + len(excerpt) + 2
            if remaining <= 0:
                break
        if sections:
            prefix += "\n\nPrior stage outputs (untrusted candidates):\n" + "\n\n".join(
                reversed(sections)
            )
        return prefix[:maximum_characters], truncated
