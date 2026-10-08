"""Command-line interface for the portable vertical slice."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from .admission import run_admission
from .api import serve
from .client import ControllerClient
from .config import load_config
from .controller import Controller
from .decision import choice_probabilities, create_decision_backend
from .doctor import run_doctor
from .downloads import build_download_plan, pull_download_plan
from .errors import SparseNetworkError
from .execution import ExecutionRequest, Top1Executor, build_top1_graph
from .fleet import FleetManager
from .models import ModelRegistry
from .retrieval import create_vector_index
from .routing import RouteRequest, StaticRouter
from .validation import validate_endpoint_artifacts


def _json_dump(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def _router_scores(values: list[str]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for value in values:
        lane, separator, score = value.partition("=")
        if not separator or not lane:
            raise ValueError(f"Router score must use lane=value: {value}")
        scores[lane] = float(score)
    return scores


def _execution_request(args: argparse.Namespace) -> ExecutionRequest:
    response_schema = None
    if args.response_schema is not None:
        loaded = json.loads(args.response_schema.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("Response schema must contain a JSON object")
        response_schema = loaded
    return ExecutionRequest(
        route=RouteRequest(
            prompt=args.prompt,
            images=tuple(args.image),
            audio=tuple(args.audio),
            explicit_lane=args.lane,
            high_risk=args.high_risk,
            requested_tools=tuple(args.tool),
            authorized_tools=tuple(args.authorize_tool),
            router_scores=_router_scores(args.router_score),
        ),
        evidence_references=tuple(args.evidence),
        priority=args.priority,
        timeout_seconds=args.timeout,
        expected_contains=args.expected_contains,
        response_schema=response_schema,
        require_citations=args.require_citations,
    )


def _apply_decision_router(
    config: Any,
    registry: ModelRegistry,
    router: StaticRouter,
    request: ExecutionRequest,
    endpoint_id: str | None,
) -> tuple[ExecutionRequest, dict[str, Any] | None]:
    route = request.route
    if endpoint_id is None or route.router_scores or route.explicit_lane is not None:
        return request, None
    if route.images or route.audio or route.requested_tools:
        return request, None
    criteria = {
        lane_id: "; ".join(
            part
            for part in (
                lane.capability.replace("_", " "),
                f"keywords: {', '.join(lane.keywords)}" if lane.keywords else "",
            )
            if part
        )
        for lane_id, lane in sorted(router.lanes.items())
        if lane_id in router.router_score_lanes
    }
    backend = create_decision_backend(config, registry, endpoint_id)
    backend.start()
    try:
        result = backend.decide(
            route.prompt,
            {
                "lane": {
                    "type": "choice",
                    "instructions": "Select the minimum sufficient capability lane.",
                    "criteria": criteria,
                }
            },
        )
    finally:
        backend.stop()
    scores = choice_probabilities(result, "lane")
    return replace(request, route=replace(route, router_scores=scores)), result.to_dict()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sparse-network")
    parser.add_argument(
        "--config",
        type=Path,
        help="Local configuration path; defaults to config.local.yaml when present",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="Inspect hardware and configuration")
    doctor.add_argument("--json", action="store_true", dest="as_json")

    models = subparsers.add_parser("models", help="Inspect model definitions")
    model_commands = models.add_subparsers(dest="models_command", required=True)
    verify = model_commands.add_parser("verify", help="Validate endpoint artifacts")
    verify.add_argument("endpoint", nargs="?", help="Endpoint ID; omit to verify all")
    verify.add_argument("--json", action="store_true", dest="as_json")
    pull_all = model_commands.add_parser(
        "pull-all", help="Download admitted Hugging Face models one at a time"
    )
    pull_all.add_argument("--cache-dir", type=Path)
    pull_all.add_argument("--endpoint", action="append", dest="endpoints", default=[])
    pull_all.add_argument("--dry-run", action="store_true")
    pull_all.add_argument("--force", action="store_true")
    pull_all.add_argument("--continue-on-error", action="store_true")
    pull_all.add_argument("--acknowledge-licenses", action="store_true")
    pull_all.add_argument("--json", action="store_true", dest="as_json")

    smoke = subparsers.add_parser("smoke", help="Run one start/request/stop lifecycle")
    smoke.add_argument("endpoint", help="Endpoint ID")
    smoke.add_argument("--prompt", required=True)
    smoke.add_argument("--image", action="append", type=Path, default=[])
    smoke.add_argument("--timeout", type=float, help="Request timeout in seconds")
    smoke.add_argument("--json", action="store_true", dest="as_json")

    admission = subparsers.add_parser("admission", help="Run frozen endpoint admission cases")
    admission.add_argument("endpoint", help="Endpoint ID")
    admission.add_argument("--suite", type=Path, default=Path("configs/admission/milestone4.yaml"))
    admission.add_argument("--case", action="append", dest="cases")
    admission.add_argument("--timeout", type=float)
    admission.add_argument("--output", type=Path, help="Write the JSON result to this path")
    admission.add_argument("--json", action="store_true", dest="as_json")

    retrieval = subparsers.add_parser("retrieval", help="Index and retrieve local evidence")
    retrieval_commands = retrieval.add_subparsers(dest="retrieval_command", required=True)
    retrieval_index = retrieval_commands.add_parser("index", help="Index UTF-8 documents")
    retrieval_index.add_argument("documents", nargs="+", type=Path)
    retrieval_index.add_argument("--endpoint", dest="embedding_endpoint")
    retrieval_index.add_argument("--output", type=Path)
    retrieval_index.add_argument("--json", action="store_true", dest="as_json")
    retrieval_search = retrieval_commands.add_parser("search", help="Search indexed evidence")
    retrieval_search.add_argument("--query", required=True)
    retrieval_search.add_argument("--endpoint", dest="embedding_endpoint")
    retrieval_search.add_argument("--top-k", type=int)
    retrieval_search.add_argument("--output", type=Path)
    retrieval_search.add_argument("--json", action="store_true", dest="as_json")
    retrieval_ask = retrieval_commands.add_parser(
        "ask", help="Retrieve evidence and submit it to an endpoint"
    )
    retrieval_ask.add_argument("endpoint", help="Generative endpoint ID")
    retrieval_ask.add_argument("--query", required=True)
    retrieval_ask.add_argument("--embedding-endpoint")
    retrieval_ask.add_argument("--top-k", type=int)
    retrieval_ask.add_argument("--timeout", type=float)
    retrieval_ask.add_argument("--output", type=Path)
    retrieval_ask.add_argument("--json", action="store_true", dest="as_json")

    decision = subparsers.add_parser("decision", help="Run a typed decision endpoint")
    decision.add_argument("endpoint", help="Decision endpoint ID")
    decision.add_argument("--state", required=True, help="Text or JSON state to evaluate")
    decision.add_argument("--questions", required=True, type=Path, help="Question JSON object")
    decision.add_argument("--image", action="append", type=Path, default=[])
    decision.add_argument("--json", action="store_true", dest="as_json")

    fleet = subparsers.add_parser("fleet", help="Run the resident model fleet")
    fleet_commands = fleet.add_subparsers(dest="fleet_command", required=True)
    fleet_run = fleet_commands.add_parser("run", help="Load the fleet and run one request")
    fleet_run.add_argument("endpoint")
    fleet_run.add_argument("--prompt", required=True)
    fleet_run.add_argument("--image", action="append", type=Path, default=[])
    fleet_run.add_argument("--priority", type=int, default=10)
    fleet_run.add_argument("--parallel", type=int)
    fleet_run.add_argument("--timeout", type=float)
    fleet_run.add_argument("--output", type=Path)
    fleet_run.add_argument("--json", action="store_true", dest="as_json")

    route = subparsers.add_parser("route", help="Plan or run deterministic top-1 routing")
    route_commands = route.add_subparsers(dest="route_command", required=True)

    def add_route_arguments(command: argparse.ArgumentParser) -> None:
        command.add_argument("--prompt", required=True)
        command.add_argument("--image", action="append", type=Path, default=[])
        command.add_argument("--audio", action="append", type=Path, default=[])
        command.add_argument("--lane", help="Explicit route lane or configured macro")
        command.add_argument("--high-risk", action="store_true")
        command.add_argument("--tool", action="append", default=[])
        command.add_argument("--authorize-tool", action="append", default=[])
        command.add_argument("--router-score", action="append", default=[])
        command.add_argument(
            "--decision-endpoint",
            help="Candidate decision model used to supply advisory route scores",
        )
        command.add_argument("--evidence", action="append", default=[])
        command.add_argument("--expected-contains")
        command.add_argument("--response-schema", type=Path)
        command.add_argument("--require-citations", action="store_true")
        command.add_argument("--priority", type=int, default=10)
        command.add_argument("--timeout", type=float)
        command.add_argument("--json", action="store_true", dest="as_json")

    route_plan = route_commands.add_parser("plan", help="Resolve and validate a static route")
    add_route_arguments(route_plan)
    route_run = route_commands.add_parser("run", help="Execute a persisted static top-1 graph")
    add_route_arguments(route_run)
    route_run.add_argument("--parallel", type=int)
    route_run.add_argument("--output", type=Path)

    api = subparsers.add_parser("api", help="Serve or operate the versioned controller API")
    api_commands = api.add_subparsers(dest="api_command", required=True)
    api_serve = api_commands.add_parser("serve", help="Run the loopback controller service")
    api_serve.add_argument("--host", default="127.0.0.1")
    api_serve.add_argument("--port", type=int, default=8765)
    api_serve.add_argument("--token")
    api_serve.add_argument("--viewer-token")
    api_serve.add_argument("--operator-token")
    api_serve.add_argument("--evaluator-token")
    api_serve.add_argument("--administrator-token")
    api_serve.add_argument("--load-fleet", action="store_true")

    def add_api_connection(command: argparse.ArgumentParser) -> None:
        command.add_argument("--url", default="http://127.0.0.1:8765")
        command.add_argument("--token")

    api_status = api_commands.add_parser("status", help="Read reconstructable controller state")
    add_api_connection(api_status)
    api_request = api_commands.add_parser("request", help="Submit a resident request")
    add_api_connection(api_request)
    api_request.add_argument("endpoint")
    api_request.add_argument("--prompt", required=True)
    api_get = api_commands.add_parser("request-status", help="Read request status or trace")
    add_api_connection(api_get)
    api_get.add_argument("request_id")
    api_get.add_argument("--trace", action="store_true")
    api_cancel = api_commands.add_parser("cancel", help="Cancel a queued or active request")
    add_api_connection(api_cancel)
    api_cancel.add_argument("request_id")
    api_cancel.add_argument("--confirmation", required=True)
    api_events = api_commands.add_parser("events", help="Replay controller events")
    add_api_connection(api_events)
    api_events.add_argument("--after", type=int, default=0)
    api_events.add_argument("--limit", type=int, default=1000)
    api_operation = api_commands.add_parser("operation", help="Run a safe fleet operation")
    add_api_connection(api_operation)
    api_operation.add_argument(
        "operation",
        choices=[
            "start",
            "smoke",
            "drain",
            "unload",
            "unload_all",
            "reload",
            "restore",
            "quarantine",
        ],
    )
    api_operation.add_argument("--endpoint")
    api_operation.add_argument("--confirmation", default="")
    api_profile_plan = api_commands.add_parser(
        "profile-plan", help="Validate and preview a hardware-profile switch"
    )
    add_api_connection(api_profile_plan)
    api_profile_plan.add_argument("profile_id")
    api_profile_plan.add_argument("--acknowledge-licenses", action="store_true")
    api_profile_apply = api_commands.add_parser(
        "profile-apply", help="Apply a previously validated hardware-profile plan"
    )
    add_api_connection(api_profile_apply)
    api_profile_apply.add_argument("plan_id")
    api_profile_apply.add_argument("--confirmation", required=True)
    api_profile_apply.add_argument("--acknowledge-licenses", action="store_true")
    api_artifact = api_commands.add_parser("artifact", help="Read authorized artifact metadata")
    add_api_connection(api_artifact)
    api_artifact.add_argument("artifact_id")
    api_evaluation = api_commands.add_parser("evaluation", help="Create or inspect an evaluation")
    add_api_connection(api_evaluation)
    api_evaluation.add_argument("--suite")
    api_evaluation.add_argument("--id", dest="evaluation_id")
    api_antidoom = api_commands.add_parser("antidoom", help="Read AntiDoom dataset/training status")
    add_api_connection(api_antidoom)
    api_config = api_commands.add_parser("config", help="Read or update allowlisted configuration")
    add_api_connection(api_config)
    api_config.add_argument("--set", action="append", default=[])
    api_config.add_argument("--confirmation", default="")
    api_swap = api_commands.add_parser("swap", help="Run an exclusive Qwen3.8-27B escalation")
    add_api_connection(api_swap)
    api_swap.add_argument("--prompt", required=True)
    api_swap.add_argument("--candidate")
    api_swap.add_argument("--expected-contains")
    api_workflow = api_commands.add_parser(
        "workflow", help="Run top-1, diverse top-2, or bounded MoSA"
    )
    add_api_connection(api_workflow)
    api_workflow.add_argument("--mode", choices=["top-1", "top-2", "mosa"], required=True)
    api_workflow.add_argument("--prompt", required=True)
    api_workflow.add_argument("--trigger", default="route_ambiguity")
    api_workflow.add_argument("--lane")
    api_workflow.add_argument("--expected-contains")
    return parser


def _api_token(args: argparse.Namespace) -> str:
    value = args.token or os.environ.get("SPARSE_API_TOKEN")
    if not value:
        raise ValueError("Set --token or SPARSE_API_TOKEN")
    return str(value)


def _api_client(args: argparse.Namespace) -> ControllerClient:
    return ControllerClient(str(args.url), _api_token(args))


def _api_role_tokens(args: argparse.Namespace) -> dict[str, str]:
    values = {
        "viewer": args.viewer_token or os.environ.get("SPARSE_VIEWER_TOKEN"),
        "operator": args.operator_token or os.environ.get("SPARSE_OPERATOR_TOKEN"),
        "evaluator": args.evaluator_token or os.environ.get("SPARSE_EVALUATOR_TOKEN"),
        "administrator": args.administrator_token
        or os.environ.get("SPARSE_ADMINISTRATOR_TOKEN"),
    }
    return {role: str(token) for role, token in values.items() if token}


def _parse_config_changes(values: list[str]) -> dict[str, Any]:
    changes: dict[str, Any] = {}
    for value in values:
        key, separator, raw = value.partition("=")
        if not separator:
            raise ValueError(f"Configuration update must use key=value: {value}")
        try:
            changes[key] = json.loads(raw)
        except json.JSONDecodeError:
            changes[key] = raw
    return changes


def _doctor_human(result: dict[str, Any]) -> None:
    platform_data = result["platform"]
    print(
        f"Platform: {platform_data['system']} {platform_data['release']} "
        f"({platform_data['machine']})"
    )
    memory_gib = result["memory"]["total_bytes"] / 1024**3
    print(f"System RAM: {memory_gib:.1f} GiB")
    for gpu in result["gpus"]:
        print(
            f"GPU {gpu['index']}: {gpu['name']} - "
            f"{gpu['memory_total_mib'] / 1024:.1f} GiB total"
        )
    llama = result["runtimes"]["llama_cpp"]
    print(f"llama.cpp: {llama['resolved'] or 'not found'}")
    print(f"Model cache: {result['paths']['model_cache'] or 'not configured'}")
    for endpoint in result["endpoints"]:
        marker = "ready" if endpoint["artifacts_present"] else "missing"
        print(f"Endpoint {endpoint['id']}: {marker} ({endpoint['adapter']})")


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(local_config=args.config)
        registry = ModelRegistry.load(config)
        if args.command == "api":
            if args.api_command == "serve":
                serve(
                    config,
                    registry,
                    host=args.host,
                    port=args.port,
                    token=_api_token(args),
                    role_tokens=_api_role_tokens(args),
                    load_fleet=args.load_fleet,
                )
                return 0
            client = _api_client(args)
            if args.api_command == "status":
                _json_dump(client.state())
            elif args.api_command == "request":
                _json_dump(client.submit(args.endpoint, args.prompt))
            elif args.api_command == "request-status":
                path = f"/v1/requests/{args.request_id}"
                if args.trace:
                    path += "/trace"
                _json_dump(client.request("GET", path))
            elif args.api_command == "cancel":
                _json_dump(client.cancel(args.request_id, confirmation=args.confirmation))
            elif args.api_command == "events":
                print(client.events(after=args.after, limit=args.limit), end="")
            elif args.api_command == "operation":
                _json_dump(
                    client.operation(
                        args.operation,
                        args.endpoint,
                        confirmation=args.confirmation,
                    )
                )
            elif args.api_command == "profile-plan":
                _json_dump(
                    client.request(
                        "POST",
                        "/v1/profiles/plan",
                        {
                            "profile_id": args.profile_id,
                            "acknowledge_licenses": args.acknowledge_licenses,
                        },
                    )
                )
            elif args.api_command == "profile-apply":
                _json_dump(
                    client.request(
                        "POST",
                        "/v1/profiles/apply",
                        {
                            "plan_id": args.plan_id,
                            "confirmation": args.confirmation,
                            "acknowledge_licenses": args.acknowledge_licenses,
                        },
                    )
                )
            elif args.api_command == "artifact":
                _json_dump(client.request("GET", f"/v1/artifacts/{args.artifact_id}"))
            elif args.api_command == "evaluation":
                if args.evaluation_id:
                    _json_dump(
                        client.request(
                            "GET",
                            f"/v1/evaluations/{args.evaluation_id}",
                            authenticated=True,
                        )
                    )
                elif args.suite:
                    _json_dump(client.request("POST", "/v1/evaluations", {"suite": args.suite}))
                else:
                    raise ValueError("evaluation needs --suite or --id")
            elif args.api_command == "antidoom":
                _json_dump(client.request("GET", "/v1/antidoom", authenticated=False))
            elif args.api_command == "config":
                changes = _parse_config_changes(args.set)
                if changes:
                    _json_dump(
                        client.request(
                            "PATCH",
                            "/v1/config",
                            {
                                "changes": changes,
                                "confirmation": args.confirmation,
                            },
                        )
                    )
                else:
                    _json_dump(client.request("GET", "/v1/config"))
            elif args.api_command == "swap":
                _json_dump(
                    client.request(
                        "POST",
                        "/v1/escalations/qwen38",
                        {
                            "prompt": args.prompt,
                            "candidate": args.candidate,
                            "expected_contains": args.expected_contains,
                        },
                    )
                )
            elif args.api_command == "workflow":
                _json_dump(
                    client.request(
                        "POST",
                        "/v1/workflows",
                        {
                            "mode": args.mode,
                            "prompt": args.prompt,
                            "trigger": args.trigger,
                            "lane": args.lane,
                            "expected_contains": args.expected_contains,
                        },
                    )
                )
            return 0
        if args.command == "doctor":
            doctor_result = run_doctor(config, registry)
            if args.as_json:
                _json_dump(doctor_result)
            else:
                _doctor_human(doctor_result)
            return 0

        if args.command == "models" and args.models_command == "verify":
            endpoints = [registry.get(args.endpoint)] if args.endpoint else list(registry.all())
            results = [
                validate_endpoint_artifacts(endpoint, config.paths["model_cache"]).to_dict()
                for endpoint in endpoints
            ]
            if args.as_json:
                _json_dump(results)
            else:
                for result in results:
                    print(f"{result['endpoint']}: valid")
                    for artifact in result["artifacts"]:
                        gib = artifact["size_bytes"] / 1024**3
                        print(f"  {artifact['name']}: {gib:.2f} GiB - {artifact['resolved_path']}")
            return 0

        if args.command == "models" and args.models_command == "pull-all":
            canonical_registry = ModelRegistry.load(config, apply_overrides=False)
            plan = build_download_plan(
                canonical_registry,
                endpoint_ids=args.endpoints,
            )
            cache_dir = (
                args.cache_dir
                or config.paths["model_cache"]
                or (config.root / "models")
            )
            review_items = [item for item in plan.items if item.license_review_required]
            if not args.dry_run and review_items and not args.acknowledge_licenses:
                names = ", ".join(item.endpoint for item in review_items)
                raise ValueError(
                    "License review acknowledgement is required for: "
                    f"{names}; inspect the upstream terms, then pass "
                    "--acknowledge-licenses"
                )
            if args.dry_run:
                plan_payload = {
                    **plan.to_dict(),
                    "cache_dir": str(cache_dir.resolve(strict=False)),
                }
                if args.as_json:
                    _json_dump(plan_payload)
                else:
                    total_gib = plan.total_size_bytes / 1024**3
                    print(
                        f"Pinned model download plan: {len(plan.items)} items, "
                        f"{total_gib:.2f} GiB"
                    )
                    for sequence, item in enumerate(plan.items, 1):
                        gib = item.size_bytes / 1024**3
                        review = " [license review]" if item.license_review_required else ""
                        snapshots = ", ".join(
                            f"{snapshot.repository}@{snapshot.revision}"
                            for snapshot in item.snapshots
                        )
                        print(
                            f"{sequence:>2}. {item.endpoint}: {gib:.2f} GiB - "
                            f"{snapshots}{review}"
                        )
                    if plan.skipped:
                        print("Skipped non-Hugging-Face endpoints: " + ", ".join(plan.skipped))
                return 0

            def report_progress(index: int, total: int, item: Any) -> None:
                gib = item.size_bytes / 1024**3
                print(
                    f"[{index}/{total}] {item.endpoint} ({gib:.2f} GiB)",
                    file=sys.stderr,
                )

            download_result = pull_download_plan(
                canonical_registry,
                plan,
                cache_dir=cache_dir,
                force=args.force,
                continue_on_error=args.continue_on_error,
                progress=report_progress,
            )
            if args.as_json:
                _json_dump(download_result)
            else:
                for item in download_result["items"]:
                    print(f"{item['endpoint']}: {item['status']}")
                if download_result["skipped_non_huggingface"]:
                    print(
                        "Skipped non-Hugging-Face endpoints: "
                        + ", ".join(download_result["skipped_non_huggingface"])
                    )
            return 0 if download_result["passed"] else 1

        if args.command == "smoke":
            smoke_result = Controller(config, registry).smoke(
                endpoint_id=args.endpoint,
                prompt=args.prompt,
                images=tuple(args.image),
                timeout_seconds=args.timeout,
            )
            if args.as_json:
                _json_dump(smoke_result.to_dict())
            else:
                print(smoke_result.answer or f"Request status: {smoke_result.envelope.status}")
                print(json.dumps(smoke_result.envelope.to_dict(), indent=2, sort_keys=True))
                print(json.dumps(smoke_result.lifecycle, indent=2, sort_keys=True))
            return 0 if smoke_result.envelope.status == "answer" else 1

        if args.command == "admission":
            suite_path = args.suite
            if not suite_path.is_absolute():
                suite_path = config.root / suite_path
            admission_result = run_admission(
                config,
                registry,
                endpoint_id=args.endpoint,
                suite_path=suite_path,
                case_ids=set(args.cases) if args.cases else None,
                timeout_seconds=args.timeout,
            )
            if args.output is not None:
                output_path = args.output
                if not output_path.is_absolute():
                    output_path = config.root / output_path
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(
                    json.dumps(admission_result, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
            if args.as_json:
                _json_dump(admission_result)
            else:
                for case in admission_result["cases"]:
                    print(f"{case['case']}: {'passed' if case['passed'] else 'failed'}")
            return 0 if admission_result["passed"] else 1

        if args.command == "decision":
            questions = json.loads(args.questions.read_text(encoding="utf-8"))
            if not isinstance(questions, dict):
                raise ValueError("Decision question file must contain a JSON object")
            try:
                state: Any = json.loads(args.state)
            except json.JSONDecodeError:
                state = args.state
            decision_backend = create_decision_backend(config, registry, args.endpoint)
            decision_backend.start()
            try:
                decision_result = decision_backend.decide(
                    state,
                    questions,
                    images=tuple(args.image),
                )
            finally:
                decision_backend.stop()
            if args.as_json:
                _json_dump(decision_result.to_dict())
            else:
                print(
                    json.dumps(
                        decision_result.answers,
                        indent=2,
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                )
            return 0

        if args.command == "retrieval":
            retrieval_config = config.data.get("retrieval", {})
            embedding_endpoint = args.embedding_endpoint or str(
                retrieval_config.get("default_endpoint", "harrier-0.6b")
            )
            requested_top_k = getattr(args, "top_k", None)
            top_k = requested_top_k or int(retrieval_config.get("default_top_k", 5))
            backend, index = create_vector_index(config, registry, embedding_endpoint)
            backend.start()
            try:
                if args.retrieval_command == "index":
                    results = [index.ingest(document) for document in args.documents]
                    value: Any = {
                        "schema": "sparse-network-indexing-result.v1",
                        "manifest": index.manifest(),
                        "documents": results,
                    }
                else:
                    search_result = index.search(args.query, top_k=top_k)
                    value = search_result.to_dict()
                    if args.retrieval_command == "ask":
                        smoke_result = Controller(config, registry).smoke(
                            endpoint_id=args.endpoint,
                            prompt=args.query,
                            evidence_references=search_result.evidence_references,
                            timeout_seconds=args.timeout,
                        )
                        value = {
                            "schema": "sparse-network-retrieval-answer.v1",
                            "retrieval": search_result.to_dict(),
                            "answer": smoke_result.to_dict(),
                        }
            finally:
                backend.stop()
            if args.output is not None:
                output_path = args.output
                if not output_path.is_absolute():
                    output_path = config.root / output_path
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(
                    json.dumps(value, indent=2, sort_keys=True), encoding="utf-8"
                )
            if args.as_json:
                _json_dump(value)
            elif args.retrieval_command == "index":
                print(f"{index.index_id}: indexed {len(value['documents'])} document(s)")
            elif args.retrieval_command == "search":
                for hit in value["hits"]:
                    print(
                        f"{hit['artifact_id']} {hit['rerank_score']:.4f} "
                        f"{hit['source_name']}:{hit['line_start']}-{hit['line_end']}"
                    )
                    print(hit["text"].strip())
            else:
                print(value["answer"]["answer"] or "No answer")
            return 0

        if args.command == "route":
            execution_request = _execution_request(args)
            scoring_router = StaticRouter(config, registry)
            execution_request, decision_router = _apply_decision_router(
                config,
                registry,
                scoring_router,
                execution_request,
                args.decision_endpoint,
            )
            if args.route_command == "plan":
                router = scoring_router
                route_decision = router.route(execution_request.route)
                graph = build_top1_graph(
                    route_decision,
                    maximum_nodes=router.maximum_graph_nodes,
                )
                value = {
                    "schema": "sparse-network-route-plan.v1",
                    "decision_router": decision_router,
                    "decision": route_decision.to_dict(),
                    "graph": graph.to_dict(),
                }
                if args.as_json:
                    _json_dump(value)
                else:
                    target = route_decision.endpoint or "rejected"
                    print(f"{route_decision.lane} -> {target}")
                    print(route_decision.reason)
                return 1 if route_decision.rejected else 0

            manager = FleetManager(
                config,
                registry,
                maximum_parallel_generations=args.parallel,
            )
            router = StaticRouter(
                config,
                registry,
                resident_endpoint_ids=set(manager.resident_endpoint_ids),
            )
            executor = Top1Executor(manager, router)
            loaded = None
            try:
                preliminary = router.route(execution_request.route)
                if not preliminary.rejected:
                    loaded = manager.load_all()
                execution_result = executor.run(execution_request)
            finally:
                stopped = manager.shutdown()
            value = {
                "schema": "sparse-network-route-run.v1",
                "decision_router": decision_router,
                "loaded": loaded,
                "execution": execution_result.to_dict(),
                "stopped": stopped,
            }
            if args.output is not None:
                output_path = args.output
                if not output_path.is_absolute():
                    output_path = config.root / output_path
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(
                    json.dumps(value, indent=2, sort_keys=True), encoding="utf-8"
                )
            if args.as_json:
                _json_dump(value)
            else:
                print(
                    execution_result.model_result.answer
                    if execution_result.model_result is not None
                    else execution_result.error
                )
                print(
                    f"{execution_result.decision.lane} -> "
                    f"{execution_result.decision.endpoint}: {execution_result.status}"
                )
            return 0 if execution_result.status in {"accepted", "needs_human_review"} else 1

        if args.command == "fleet" and args.fleet_command == "run":
            manager = FleetManager(
                config,
                registry,
                maximum_parallel_generations=args.parallel,
            )
            try:
                loaded = manager.load_all()
                fleet_result = manager.submit(
                    endpoint_id=args.endpoint,
                    prompt=args.prompt,
                    images=tuple(args.image),
                    priority=args.priority,
                    timeout_seconds=args.timeout,
                )
            finally:
                stopped = manager.shutdown()
            value = {
                "schema": "sparse-network-fleet-run.v1",
                "loaded": loaded,
                "result": fleet_result.to_dict(),
                "stopped": stopped,
            }
            if args.output is not None:
                output_path = args.output
                if not output_path.is_absolute():
                    output_path = config.root / output_path
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(
                    json.dumps(value, indent=2, sort_keys=True), encoding="utf-8"
                )
            if args.as_json:
                _json_dump(value)
            else:
                print(fleet_result.answer or fleet_result.envelope.status)
                print(f"Peak fleet VRAM: {stopped['peak_vram_bytes']} bytes")
            return 0 if fleet_result.envelope.status == "answer" else 1
    except (SparseNetworkError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    parser.error("Unhandled command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
