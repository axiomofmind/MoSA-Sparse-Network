from __future__ import annotations

from pathlib import Path

from sparse_network.config import load_config
from sparse_network.models import ModelRegistry
from sparse_network.routing import RouteRequest, StaticRouter


def _router(*, include_gemma12: bool = True) -> StaticRouter:
    config = load_config()
    residents = {
        "lfm25-1.2b",
        "qwen35-4b",
        "nemotron-4b",
        "gemma-e2b-vision",
        "qwen3-8b-fp8",
    }
    if include_gemma12:
        residents.add("gemma4-12b-qat")
    return StaticRouter(
        config,
        ModelRegistry.load(config),
        resident_endpoint_ids=residents,
    )


def test_explicit_macro_has_precedence_over_keywords() -> None:
    request = RouteRequest(prompt="@route:classify Write Python code to label this message")
    decision = _router().route(request)
    assert decision.lane == "classification"
    assert decision.endpoint == "lfm25-1.2b"
    assert decision.source == "explicit_macro"
    assert request.runtime_prompt == "Write Python code to label this message"


def test_required_image_modality_overrides_text_macro() -> None:
    decision = _router().route(
        RouteRequest(prompt="@route:code Describe this", images=(Path("fixture.png"),))
    )
    assert decision.lane == "visual_understanding"
    assert decision.endpoint == "gemma-e2b-vision"
    assert decision.source == "required_modality"


def test_visual_sublane_macro_is_preserved() -> None:
    decision = _router().route(
        RouteRequest(prompt="@route:document Extract fields", images=(Path("invoice.png"),))
    )
    assert decision.lane == "visual_document_extraction"


def test_high_risk_policy_requires_human_review() -> None:
    decision = _router().route(RouteRequest(prompt="Assess this", high_risk=True))
    assert decision.endpoint == "gemma4-12b-qat"
    assert decision.verification_required
    assert decision.human_review_required
    assert decision.source == "risk_policy"


def test_gemma12_has_large_verification_and_difficult_visual_duties() -> None:
    verifier = _router().route(
        RouteRequest(prompt="@route:verify-large Check the proposed answer")
    )
    assert verifier.lane == "large_verification"
    assert verifier.endpoint == "gemma4-12b-qat"
    assert verifier.verification_required

    visual = _router().route(
        RouteRequest(
            prompt="@route:vision-quality Inspect subtle details",
            images=(Path("fixture.png"),),
        )
    )
    assert visual.lane == "difficult_visual"
    assert visual.endpoint == "gemma4-12b-qat"


def test_lean_profile_falls_back_when_gemma12_is_not_resident() -> None:
    decision = _router(include_gemma12=False).route(
        RouteRequest(prompt="@route:verify-large Check this")
    )
    assert decision.endpoint == "qwen3-8b-fp8"


def test_tool_allowlist_and_authorization_are_controller_rules() -> None:
    router = _router()
    denied = router.route(
        RouteRequest(prompt="Use a shell", requested_tools=("shell",), authorized_tools=("shell",))
    )
    assert denied.rejected
    assert "not allowlisted" in denied.reason
    unauthorized = router.route(
        RouteRequest(prompt="Search", requested_tools=("retrieval_search",))
    )
    assert unauthorized.rejected
    allowed = router.route(
        RouteRequest(
            prompt="Search",
            requested_tools=("retrieval_search",),
            authorized_tools=("retrieval_search",),
        )
    )
    assert not allowed.rejected
    assert allowed.lane == "tool_planning"


def test_router_scores_are_bounded_and_have_deterministic_fallback() -> None:
    router = _router()
    scored = router.route(
        RouteRequest(prompt="Neutral request", router_scores={"classification": 0.9})
    )
    assert scored.lane == "classification"
    assert scored.router_scores_used
    invalid = router.route(
        RouteRequest(
            prompt="Neutral request",
            router_scores={"classification": 2.0, "invented": 0.99},
        )
    )
    assert invalid.lane == "general_generation"
    assert invalid.source == "fallback"
    assert not invalid.router_scores_used


def test_context_fallback_and_unavailable_audio_are_controlled() -> None:
    router = _router()
    context = router.route(RouteRequest(prompt="@route:general " + "x" * 6500))
    assert context.endpoint == "qwen3-8b-fp8"
    audio = router.route(RouteRequest(prompt="Transcribe", audio=(Path("audio.wav"),)))
    assert audio.rejected
    assert audio.lane == "audio_transcription"


def test_unknown_explicit_macro_is_rejected() -> None:
    decision = _router().route(RouteRequest(prompt="@route:invented hello"))
    assert decision.rejected
    assert decision.source == "explicit_macro"
