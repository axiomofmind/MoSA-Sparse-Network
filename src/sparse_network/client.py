"""Small typed client for the versioned local controller API."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class ControllerClient:
    base_url: str
    token: str
    timeout_seconds: float = 30

    def request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        *,
        authenticated: bool = True,
    ) -> dict[str, Any]:
        headers = {"Accept": "application/json"}
        if authenticated:
            headers["Authorization"] = f"Bearer {self.token}"
        data = None
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = Request(
            self.base_url.rstrip("/") + path,
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:  # noqa: S310
                value = json.loads(response.read())
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Controller API returned {exc.code}: {detail}") from exc
        if not isinstance(value, dict):
            raise RuntimeError("Controller API response was not an object")
        return value

    def fleet(self) -> dict[str, Any]:
        return self.request("GET", "/v1/fleet", authenticated=False)

    def state(self) -> dict[str, Any]:
        return self.request("GET", "/v1/state")

    def submit(self, endpoint: str, prompt: str) -> dict[str, Any]:
        return self.request("POST", "/v1/requests", {"endpoint": endpoint, "prompt": prompt})

    def request_status(self, request_id: str) -> dict[str, Any]:
        return self.request("GET", f"/v1/requests/{request_id}")

    def cancel(self, request_id: str, *, confirmation: str) -> dict[str, Any]:
        return self.request(
            "POST",
            f"/v1/requests/{request_id}/cancel",
            {"confirmation": confirmation},
        )

    def operation(
        self,
        name: str,
        endpoint: str | None = None,
        *,
        confirmation: str = "",
    ) -> dict[str, Any]:
        payload = {"endpoint": endpoint} if endpoint else {}
        if confirmation:
            payload["confirmation"] = confirmation
        return self.request("POST", f"/v1/fleet/operations/{name}", payload)

    def events(self, *, after: int = 0, limit: int = 1000) -> str:
        request = Request(
            self.base_url.rstrip("/") + f"/v1/events?after={after}&limit={limit}",
            headers={
                "Accept": "text/event-stream",
                "Authorization": f"Bearer {self.token}",
            },
        )
        with urlopen(request, timeout=self.timeout_seconds) as response:  # noqa: S310
            content: str = response.read().decode("utf-8")
            return content
