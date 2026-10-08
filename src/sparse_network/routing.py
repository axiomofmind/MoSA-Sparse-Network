"""Deterministic, policy-bounded static routing for the resident fleet."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from .config import AppConfig
from .errors import ConfigurationError
from .models import EndpointDefinition, ModelRegistry

MACRO_PATTERN = re.compile(r"^\s*@route:([a-z0-9_-]+)\b", re.IGNORECASE)


@dataclass(frozen=True)
class RouteRequest:
    prompt: str
    images: tuple[Path, ...] = ()
    audio: tuple[Path, ...] = ()
    explicit_lane: str | None = None
    high_risk: bool = False
    requested_tools: tuple[str, ...] = ()
    authorized_tools: tuple[str, ...] = ()
    router_scores: dict[str, float] | None = None

    @property
    def required_modalities(self) -> tuple[str, ...]:
        values = ["text"]
        if self.images:
            values.append("image")
        if self.audio:
            values.append("audio")
        return tuple(values)

    @property
    def runtime_prompt(self) -> str:
        """Remove a leading controller macro before model inference."""
        return MACRO_PATTERN.sub("", self.prompt, count=1).lstrip()


@dataclass(frozen=True)
class RouteDecision:
    lane: str
    endpoint: str | None
    reason: str
    source: str
    required_modalities: tuple[str, ...]
    verification_required: bool = False
    human_review_required: bool = False
    rejected: bool = False
    router_scores_used: bool = False
    schema: str = "sparse-network-route-decision.v1"

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["required_modalities"] = list(self.required_modalities)
        return value


@dataclass(frozen=True)
class RouteLane:
    id: str
    endpoint: str | None
    fallback_endpoints: tuple[str, ...]
    capability: str
    modalities: tuple[str, ...]
    keywords: tuple[str, ...]
    verification_required: bool
    human_review_required: bool


class StaticRouter:
    """Apply controller rules in a fixed order; model scores are advisory only."""

    def __init__(
        self,
        config: AppConfig,
        registry: ModelRegistry,
        *,
        resident_endpoint_ids: set[str] | None = None,
        routes_path: Path | None = None,
    ) -> None:
        self.config = config
        self.registry = registry
        routing = config.data.get("routing", {})
        configured_path = routes_path or Path(
            str(routing.get("routes", "configs/routes.yaml"))
        )
        if not configured_path.is_absolute():
            configured_path = config.root / configured_path
        self.routes_path = configured_path.resolve(strict=True)
        try:
            raw = yaml.safe_load(self.routes_path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            raise ConfigurationError(f"Invalid route configuration: {exc}") from exc
        if not isinstance(raw, dict) or raw.get("schema") != "sparse-network-route-config.v1":
            raise ConfigurationError("Unsupported or missing route configuration schema")
        defaults = raw.get("defaults", {})
        macros = raw.get("macros", {})
        lanes = raw.get("lanes", {})
        score_lanes = raw.get("router_score_lanes", [])
        if not all(isinstance(value, dict) for value in (defaults, macros, lanes)):
            raise ConfigurationError("Route defaults, macros, and lanes must be mappings")
        if not isinstance(score_lanes, list):
            raise ConfigurationError("router_score_lanes must be a list")
        self.default_lane = str(defaults.get("lane", "general_generation"))
        self.router_score_threshold = float(defaults.get("router_score_threshold", 0.7))
        self.maximum_graph_nodes = int(defaults.get("maximum_graph_nodes", 8))
        self.allowed_tools = frozenset(str(value) for value in defaults.get("allowed_tools", []))
        self.macros = {str(key): str(value) for key, value in macros.items()}
        self.router_score_lanes = frozenset(str(value) for value in score_lanes)
        self.lanes: dict[str, RouteLane] = {}
        for lane_id, lane_data in lanes.items():
            if not isinstance(lane_data, dict):
                raise ConfigurationError(f"Route lane {lane_id} must be a mapping")
            endpoint_value = lane_data.get("endpoint")
            endpoint = str(endpoint_value) if endpoint_value is not None else None
            self.lanes[str(lane_id)] = RouteLane(
                id=str(lane_id),
                endpoint=endpoint,
                fallback_endpoints=tuple(
                    str(value) for value in lane_data.get("fallback_endpoints", [])
                ),
                capability=str(lane_data.get("capability", "")),
                modalities=tuple(str(value) for value in lane_data.get("modalities", [])),
                keywords=tuple(str(value).lower() for value in lane_data.get("keywords", [])),
                verification_required=bool(lane_data.get("verification_required", False)),
                human_review_required=bool(lane_data.get("human_review_required", False)),
            )
        self.resident_endpoint_ids = resident_endpoint_ids
        self._validate_config()

    def _validate_config(self) -> None:
        if self.default_lane not in self.lanes:
            raise ConfigurationError(f"Unknown default route lane: {self.default_lane}")
        if self.maximum_graph_nodes < 4:
            raise ConfigurationError("maximum_graph_nodes must be at least four")
        if not 0 <= self.router_score_threshold <= 1:
            raise ConfigurationError("router_score_threshold must be between zero and one")
        for macro, macro_lane in self.macros.items():
            if macro_lane not in self.lanes:
                raise ConfigurationError(
                    f"Macro {macro} references unknown lane {macro_lane}"
                )
        for lane_id in self.router_score_lanes:
            if lane_id not in self.lanes:
                raise ConfigurationError(f"Router score lane is unknown: {lane_id}")
        known_endpoints = {endpoint.id for endpoint in self.registry.all()}
        for lane in self.lanes.values():
            for endpoint in (lane.endpoint, *lane.fallback_endpoints):
                if endpoint is not None and endpoint not in known_endpoints:
                    raise ConfigurationError(
                        f"Route lane {lane.id} references unknown endpoint {endpoint}"
                    )

    def _reject(self, request: RouteRequest, lane: str, reason: str, source: str) -> RouteDecision:
        return RouteDecision(
            lane=lane,
            endpoint=None,
            reason=reason,
            source=source,
            required_modalities=request.required_modalities,
            rejected=True,
            verification_required=request.high_risk,
            human_review_required=request.high_risk,
        )

    def _explicit_lane(self, request: RouteRequest) -> str | None:
        raw = request.explicit_lane
        if raw is None:
            match = MACRO_PATTERN.match(request.prompt)
            raw = match.group(1) if match else None
        if raw is None:
            return None
        normalized = raw.lower()
        return self.macros.get(normalized, normalized)

    def _visual_lane(self, request: RouteRequest, explicit: str | None) -> str:
        visual = {
            "visual_understanding",
            "visual_document_extraction",
            "gui_screen_grounding",
            "difficult_visual",
        }
        if explicit in visual:
            return explicit
        prompt = request.prompt.lower()
        if any(word in prompt for word in ("screenshot", "screen", "button", "gui", " ui ")):
            return "gui_screen_grounding"
        if any(word in prompt for word in ("document", "invoice", "receipt", "form", "ocr")):
            return "visual_document_extraction"
        return "visual_understanding"

    def _score_lane(self, scores: dict[str, float] | None) -> str | None:
        if not scores:
            return None
        bounded: list[tuple[float, str]] = []
        for lane, score in scores.items():
            if lane not in self.router_score_lanes or isinstance(score, bool):
                continue
            if not isinstance(score, (int, float)) or not 0 <= float(score) <= 1:
                continue
            bounded.append((float(score), lane))
        if not bounded:
            return None
        score, lane = max(bounded, key=lambda value: (value[0], value[1]))
        return lane if score >= self.router_score_threshold else None

    def _keyword_lane(self, prompt: str) -> str:
        lowered = prompt.lower()
        for lane in self.lanes.values():
            if lane.keywords and any(keyword in lowered for keyword in lane.keywords):
                return lane.id
        return self.default_lane

    def _endpoint_eligible(
        self,
        endpoint: EndpointDefinition,
        lane: RouteLane,
        request: RouteRequest,
        *,
        require_lane_capability: bool,
    ) -> bool:
        if self.resident_endpoint_ids is not None and endpoint.id not in self.resident_endpoint_ids:
            return False
        if not set(request.required_modalities).issubset(endpoint.modalities):
            return False
        if require_lane_capability and lane.capability not in endpoint.capabilities:
            return False
        return not (
            endpoint.max_input_characters
            and len(request.prompt) > endpoint.max_input_characters
        )

    def _choose_endpoint(self, lane: RouteLane, request: RouteRequest) -> str | None:
        if lane.endpoint is None:
            return None
        primary = self.registry.get(lane.endpoint)
        if self._endpoint_eligible(primary, lane, request, require_lane_capability=True):
            return primary.id
        for endpoint_id in lane.fallback_endpoints:
            endpoint = self.registry.get(endpoint_id)
            if self._endpoint_eligible(
                endpoint,
                lane,
                request,
                require_lane_capability=False,
            ):
                return endpoint.id
        return None

    def route(self, request: RouteRequest) -> RouteDecision:
        if not request.prompt.strip():
            return self._reject(request, self.default_lane, "Prompt is empty", "validation")
        requested_tools = set(request.requested_tools)
        unknown_tools = requested_tools - self.allowed_tools
        if unknown_tools:
            return self._reject(
                request,
                "tool_planning",
                f"Tools are not allowlisted: {', '.join(sorted(unknown_tools))}",
                "tool_policy",
            )
        unauthorized = requested_tools - set(request.authorized_tools)
        if unauthorized:
            return self._reject(
                request,
                "tool_planning",
                f"Tools are not authorized: {', '.join(sorted(unauthorized))}",
                "tool_policy",
            )

        explicit = self._explicit_lane(request)
        if explicit is not None and explicit not in self.lanes:
            return self._reject(
                request,
                explicit,
                f"Explicit route lane is unknown: {explicit}",
                "explicit_macro",
            )

        router_scores_used = False
        if request.audio:
            lane_id, source, reason = (
                "audio_transcription",
                "required_modality",
                "Audio input requires the audio transcription lane",
            )
        elif request.images:
            lane_id, source, reason = (
                self._visual_lane(request, explicit),
                "required_modality",
                "Image input requires a compatible vision lane",
            )
        elif request.high_risk:
            lane_id, source, reason = (
                "high_risk",
                "risk_policy",
                "High-risk work requires deterministic checks and human review",
            )
        elif requested_tools:
            lane_id, source, reason = (
                "tool_planning",
                "tool_policy",
                "Authorized tool use requires the tool-planning lane",
            )
        elif explicit is not None:
            lane_id, source, reason = explicit, "explicit_macro", "Explicit route macro selected"
        else:
            scored = self._score_lane(request.router_scores)
            if scored is not None:
                lane_id, source, reason = (
                    scored,
                    "bounded_router_score",
                    "Valid bounded router score exceeded the configured threshold",
                )
                router_scores_used = True
            else:
                lane_id = self._keyword_lane(request.prompt)
                source = "static_keyword" if lane_id != self.default_lane else "fallback"
                reason = (
                    "Static keyword rule selected"
                    if source == "static_keyword"
                    else "Deterministic general-generation fallback selected"
                )

        lane = self.lanes[lane_id]
        endpoint = self._choose_endpoint(lane, request)
        verification_required = lane.verification_required or request.high_risk
        human_review_required = lane.human_review_required or request.high_risk
        if endpoint is None:
            if lane.endpoint is None:
                reason = f"{reason}; no endpoint is admitted for this lane"
            else:
                reason = (
                    f"{reason}; no resident endpoint satisfies capability, modality, "
                    "and context limits"
                )
            return RouteDecision(
                lane=lane_id,
                endpoint=None,
                reason=reason,
                source=source,
                required_modalities=request.required_modalities,
                verification_required=verification_required,
                human_review_required=human_review_required,
                rejected=True,
                router_scores_used=router_scores_used,
            )
        return RouteDecision(
            lane=lane_id,
            endpoint=endpoint,
            reason=reason,
            source=source,
            required_modalities=request.required_modalities,
            verification_required=verification_required,
            human_review_required=human_review_required,
            router_scores_used=router_scores_used,
        )
