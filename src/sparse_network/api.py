"""Versioned loopback HTTP API for the sparse-network controller."""

from __future__ import annotations

import base64
import json
import mimetypes
import secrets
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .auth import AuthContext, context_for_role
from .config import AppConfig
from .models import ModelRegistry
from .service import ControllerService


class ControllerHTTPServer(ThreadingHTTPServer):
    service: ControllerService
    tokens: dict[str, str]


class ControllerAPIHandler(BaseHTTPRequestHandler):
    server: ControllerHTTPServer
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: object) -> None:
        return

    def _auth_context(self) -> AuthContext | None:
        supplied = self.headers.get("Authorization", "")
        for token, role in self.server.tokens.items():
            if secrets.compare_digest(supplied, f"Bearer {token}"):
                return context_for_role(role)
        return None

    def _json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length > 1024 * 1024:
            raise ValueError("request body exceeds 1 MiB")
        if length == 0:
            return {}
        value = json.loads(self.rfile.read(length))
        if not isinstance(value, dict):
            raise ValueError("request body must be a JSON object")
        return value

    def _send(self, status: HTTPStatus, value: Any) -> None:
        payload = json.dumps(value, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def _send_dashboard_asset(self, path: str) -> None:
        dashboard_root = (self.server.service.config.root / "dashboard").resolve(strict=True)
        relative = "index.html" if path in {"/", "/dashboard"} else path.removeprefix(
            "/dashboard/"
        )
        candidate = (dashboard_root / relative).resolve(strict=True)
        if not candidate.is_relative_to(dashboard_root) or not candidate.is_file():
            raise FileNotFoundError(path)
        payload = candidate.read_bytes()
        media_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
        if candidate.suffix == ".js":
            media_type = "text/javascript"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", f"{media_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-cache")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self' data:; style-src 'self'; "
            "script-src 'self'; connect-src 'self'",
        )
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.end_headers()
        self.wfile.write(payload)

    def _error(self, status: HTTPStatus, exc: BaseException) -> None:
        self._send(
            status,
            {"schema": "sparse-network-error.v1", "error": str(exc), "type": type(exc).__name__},
        )

    def _require_auth(self) -> bool:
        if self._auth_context() is not None:
            return True
        self._error(HTTPStatus.UNAUTHORIZED, PermissionError("Bearer token required"))
        return False

    def _require_permission(self, permission: str) -> AuthContext | None:
        context = self._auth_context()
        if context is None:
            self._error(HTTPStatus.UNAUTHORIZED, PermissionError("Bearer token required"))
            return None
        if not context.allows(permission):
            self._error(
                HTTPStatus.FORBIDDEN,
                PermissionError(
                    f"Role {context.role!r} does not grant permission {permission!r}"
                ),
            )
            return None
        return context

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        try:
            if path == "/" or path == "/dashboard" or path.startswith("/dashboard/"):
                self._send_dashboard_asset(path)
            elif path == "/v1/health":
                self._send(HTTPStatus.OK, {"schema": "sparse-network-health.v1", "ok": True})
            elif path == "/v1/fleet":
                self._send(HTTPStatus.OK, self.server.service.fleet.status())
            elif path == "/v1/state":
                if self._require_permission("controller:read") is None:
                    return
                self._send(HTTPStatus.OK, self.server.service.state_snapshot())
            elif path == "/v1/events":
                if self._require_permission("dashboard:read") is None:
                    return
                query = parse_qs(parsed.query)
                after = int(query.get("after", ["0"])[0])
                limit = int(query.get("limit", ["1000"])[0])
                events = self.server.service.fleet.event_log.read_after(after, limit=limit)
                payload = b"".join(
                    f"id: {event['event_id']}\nevent: {event['event']}\ndata: "
                    f"{json.dumps(event, sort_keys=True)}\n\n".encode()
                    for event in events
                )
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(payload)
            elif path.startswith("/v1/requests/"):
                if self._require_permission("controller:read") is None:
                    return
                parts = path.split("/")
                request_id = parts[3]
                value = (
                    self.server.service.trace(request_id)
                    if len(parts) == 5 and parts[4] == "trace"
                    else self.server.service.request(request_id)
                )
                self._send(HTTPStatus.OK, value)
            elif path.startswith("/v1/artifacts/"):
                permission = (
                    "artifact:content" if path.endswith("/content") else "dashboard:read"
                )
                if self._require_permission(permission) is None:
                    return
                parts = path.split("/")
                artifact_id = parts[3]
                store = self.server.service.fleet.controller.artifacts
                record = store.get(artifact_id)
                if len(parts) == 5 and parts[4] == "content":
                    if record.media_type.startswith("text/"):
                        content: Any = {
                            "encoding": "utf-8",
                            "content": store.read_text(artifact_id),
                        }
                    else:
                        content = {
                            "encoding": "base64",
                            "content": base64.b64encode(Path(record.path).read_bytes()).decode(),
                        }
                    self._send(
                        HTTPStatus.OK,
                        {"schema": "sparse-network-artifact-content.v1", **content},
                    )
                else:
                    self._send(
                        HTTPStatus.OK,
                        self.server.service.dashboard_artifact(artifact_id),
                    )
            elif path == "/v1/antidoom":
                self._send(HTTPStatus.OK, self.server.service.antidoom_status())
            elif path.startswith("/v1/evaluations/"):
                if self._require_permission("dashboard:read") is None:
                    return
                evaluation_id = path.rsplit("/", 1)[1]
                self._send(HTTPStatus.OK, self.server.service.evaluations[evaluation_id])
            elif path == "/v1/config":
                if self._require_permission("controller:read") is None:
                    return
                self._send(HTTPStatus.OK, self.server.service.config_view())
            elif path == "/v1/session":
                context = self._auth_context()
                if context is None:
                    self._error(
                        HTTPStatus.UNAUTHORIZED,
                        PermissionError("Bearer token required"),
                    )
                    return
                self._send(
                    HTTPStatus.OK,
                    {"schema": "sparse-network-session.v1", **context.to_dict()},
                )
            elif path == "/v1/dashboard":
                context = self._require_permission("dashboard:read")
                if context is None:
                    return
                self._send(
                    HTTPStatus.OK,
                    self.server.service.dashboard_snapshot(session=context.to_dict()),
                )
            elif path == "/v1/profiles":
                if self._require_permission("dashboard:read") is None:
                    return
                self._send(
                    HTTPStatus.OK,
                    self.server.service.dashboard_catalog()["profiles"],
                )
            elif path == "/v1/use-cases":
                if self._require_permission("dashboard:read") is None:
                    return
                self._send(HTTPStatus.OK, self.server.service.use_cases.catalog())
            elif path.startswith("/v1/use-cases/runs/"):
                if self._require_permission("dashboard:read") is None:
                    return
                run_id = path.split("/")[4]
                self._send(HTTPStatus.OK, self.server.service.use_case_run(run_id))
            elif path.startswith("/v1/runs/"):
                if self._require_permission("dashboard:read") is None:
                    return
                run_id = path.rsplit("/", 1)[1]
                self._send(
                    HTTPStatus.OK,
                    self.server.service.dashboard_run(run_id),
                )
            elif path == "/v1/openapi.json":
                schema_path = self.server.service.config.root / "schemas" / "openapi-v1.json"
                self._send(HTTPStatus.OK, json.loads(schema_path.read_text(encoding="utf-8")))
            else:
                self._error(HTTPStatus.NOT_FOUND, KeyError(path))
        except KeyError as exc:
            self._error(HTTPStatus.NOT_FOUND, exc)
        except FileNotFoundError as exc:
            self._error(HTTPStatus.NOT_FOUND, exc)
        except ValueError as exc:
            self._error(HTTPStatus.BAD_REQUEST, exc)
        except BaseException as exc:
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, exc)

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path.rstrip("/")
        try:
            body = self._json_body()
            if path == "/v1/requests":
                if self._require_permission("request:submit") is None:
                    return
                self._send(HTTPStatus.ACCEPTED, self.server.service.submit(body))
            elif path.endswith("/cancel") and path.startswith("/v1/requests/"):
                if self._require_permission("request:cancel") is None:
                    return
                context = self._auth_context()
                if context is None:
                    return
                request_id = path.split("/")[3]
                self._send(
                    HTTPStatus.ACCEPTED,
                    self.server.service.cancel(
                        request_id,
                        confirmation=str(body.get("confirmation", "")),
                        actor=context.role,
                    ),
                )
            elif path.startswith("/v1/fleet/operations/"):
                context = self._require_permission("fleet:operate")
                if context is None:
                    return
                operation = path.rsplit("/", 1)[1]
                endpoint = str(body["endpoint"]) if body.get("endpoint") else None
                self._send(
                    HTTPStatus.OK,
                    self.server.service.fleet_operation(
                        operation,
                        endpoint,
                        confirmation=str(body.get("confirmation", "")),
                        actor=context.role,
                    ),
                )
            elif path == "/v1/profiles/plan":
                context = self._require_permission("profile:apply")
                if context is None:
                    return
                self._send(
                    HTTPStatus.OK,
                    self.server.service.plan_profile_switch(body, actor=context.role),
                )
            elif path == "/v1/profiles/apply":
                context = self._require_permission("profile:apply")
                if context is None:
                    return
                self._send(
                    HTTPStatus.OK,
                    self.server.service.apply_profile_switch(body, actor=context.role),
                )
            elif path == "/v1/evaluations":
                context = self._require_permission("evaluation:run")
                if context is None:
                    return
                self._send(
                    HTTPStatus.CREATED,
                    self.server.service.create_evaluation(body, actor=context.role),
                )
            elif path == "/v1/escalations/qwen38":
                if self._require_permission("request:submit") is None:
                    return
                self._send(HTTPStatus.OK, self.server.service.run_swap(body))
            elif path == "/v1/workflows":
                if self._require_permission("request:submit") is None:
                    return
                self._send(HTTPStatus.OK, self.server.service.run_workflow(body))
            elif path == "/v1/use-cases/runs":
                context = self._require_permission("usecase:run")
                if context is None:
                    return
                self._send(
                    HTTPStatus.ACCEPTED,
                    self.server.service.submit_use_case(
                        body,
                        actor=context.role,
                        permissions=set(context.permissions),
                    ),
                )
            elif path == "/v1/use-cases/readiness":
                context = self._require_permission("usecase:run")
                if context is None:
                    return
                self._send(
                    HTTPStatus.OK,
                    self.server.service.use_case_readiness(
                        body,
                        permissions=set(context.permissions),
                    ),
                )
            elif path.startswith("/v1/use-cases/runs/") and path.endswith(
                "/corrections"
            ):
                context = self._require_permission("usecase:run")
                if context is None:
                    return
                run_id = path.split("/")[4]
                self._send(
                    HTTPStatus.CREATED,
                    self.server.service.correct_use_case_result(
                        run_id,
                        body,
                        actor=context.role,
                    ),
                )
            elif path.startswith("/v1/use-cases/runs/") and path.endswith("/exports"):
                context = self._require_permission("usecase:run")
                if context is None:
                    return
                run_id = path.split("/")[4]
                self._send(
                    HTTPStatus.CREATED,
                    self.server.service.export_use_case_result(
                        run_id,
                        body,
                        actor=context.role,
                    ),
                )
            elif path.startswith("/v1/use-cases/runs/") and path.endswith("/cancel"):
                context = self._require_permission("request:cancel")
                if context is None:
                    return
                run_id = path.split("/")[4]
                self._send(
                    HTTPStatus.ACCEPTED,
                    self.server.service.cancel_use_case(
                        run_id,
                        confirmation=str(body.get("confirmation", "")),
                        actor=context.role,
                    ),
                )
            elif path.startswith("/v1/use-cases/runs/") and path.endswith("/replay"):
                context = self._require_permission("usecase:run")
                if context is None:
                    return
                run_id = path.split("/")[4]
                self._send(
                    HTTPStatus.CREATED,
                    self.server.service.replay_use_case(
                        run_id,
                        body,
                        actor=context.role,
                        permissions=set(context.permissions),
                    ),
                )
            elif path.startswith("/v1/use-cases/runs/") and path.endswith("/admission"):
                context = self._require_permission("configuration:write")
                if context is None:
                    return
                run_id = path.split("/")[4]
                self._send(
                    HTTPStatus.CREATED,
                    self.server.service.decide_use_case_admission(
                        run_id, body, actor=context.role
                    ),
                )
            else:
                self._error(HTTPStatus.NOT_FOUND, KeyError(path))
        except KeyError as exc:
            self._error(HTTPStatus.NOT_FOUND, exc)
        except ValueError as exc:
            self._error(HTTPStatus.BAD_REQUEST, exc)
        except PermissionError as exc:
            self._error(HTTPStatus.FORBIDDEN, exc)
        except BaseException as exc:
            self._error(HTTPStatus.CONFLICT, exc)

    def do_PATCH(self) -> None:  # noqa: N802
        path = urlparse(self.path).path.rstrip("/")
        context = self._require_permission("configuration:write")
        if context is None:
            return
        try:
            if path != "/v1/config":
                raise KeyError(path)
            body = self._json_body()
            changes = body.get("changes")
            if not isinstance(changes, dict):
                raise ValueError("changes must be an object")
            confirmation = str(body.get("confirmation", ""))
            self._send(
                HTTPStatus.OK,
                self.server.service.update_config(
                    changes,
                    confirmation=confirmation,
                    actor=context.role,
                ),
            )
        except KeyError as exc:
            self._error(HTTPStatus.NOT_FOUND, exc)
        except ValueError as exc:
            self._error(HTTPStatus.BAD_REQUEST, exc)


def build_server(
    config: AppConfig,
    registry: ModelRegistry,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    token: str,
    role_tokens: dict[str, str] | None = None,
) -> ControllerHTTPServer:
    if host not in {"127.0.0.1", "::1", "localhost"}:
        raise ValueError("Controller API may only bind to a loopback address")
    if len(token) < 16:
        raise ValueError("API bearer token must contain at least 16 characters")
    configured_tokens = {token: "administrator"}
    for role, role_token in (role_tokens or {}).items():
        context_for_role(role)
        if len(role_token) < 16:
            raise ValueError(f"API bearer token for {role} must contain at least 16 characters")
        if role_token in configured_tokens and configured_tokens[role_token] != role:
            raise ValueError("Controller role tokens must be distinct")
        configured_tokens[role_token] = role
    server = ControllerHTTPServer((host, port), ControllerAPIHandler)
    server.service = ControllerService(config, registry)
    server.tokens = configured_tokens
    return server


def serve(
    config: AppConfig,
    registry: ModelRegistry,
    *,
    host: str,
    port: int,
    token: str,
    role_tokens: dict[str, str] | None = None,
    load_fleet: bool = False,
) -> None:
    server = build_server(
        config,
        registry,
        host=host,
        port=port,
        token=token,
        role_tokens=role_tokens,
    )
    if load_fleet:
        server.service.start()
    try:
        server.serve_forever()
    finally:
        server.service.fleet.shutdown()
        server.server_close()
