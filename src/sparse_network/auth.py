"""Local role and permission policy for the controller API."""

from __future__ import annotations

from dataclasses import dataclass

ROLE_PERMISSIONS: dict[str, frozenset[str]] = {
    "viewer": frozenset({"dashboard:read"}),
    "operator": frozenset(
        {
            "dashboard:read",
            "controller:read",
            "artifact:content",
            "request:submit",
            "request:cancel",
            "usecase:run",
            "tool:diagnose",
            "fleet:operate",
            "profile:apply",
        }
    ),
    "evaluator": frozenset(
        {
            "dashboard:read",
            "controller:read",
            "artifact:content",
            "request:submit",
            "evaluation:run",
            "usecase:run",
        }
    ),
    "administrator": frozenset(
        {
            "dashboard:read",
            "controller:read",
            "artifact:content",
            "request:submit",
            "request:cancel",
            "usecase:run",
            "tool:diagnose",
            "tool:remediate",
            "fleet:operate",
            "profile:apply",
            "evaluation:run",
            "configuration:write",
        }
    ),
}


@dataclass(frozen=True)
class AuthContext:
    role: str
    permissions: frozenset[str]

    def allows(self, permission: str) -> bool:
        return permission in self.permissions

    def to_dict(self) -> dict[str, object]:
        return {
            "role": self.role,
            "permissions": sorted(self.permissions),
            "read_only": not bool(
                self.permissions
                & {
                    "request:submit",
                    "request:cancel",
                    "usecase:run",
                    "tool:diagnose",
                    "tool:remediate",
                    "fleet:operate",
                    "profile:apply",
                    "evaluation:run",
                    "configuration:write",
                }
            ),
        }


def context_for_role(role: str) -> AuthContext:
    try:
        permissions = ROLE_PERMISSIONS[role]
    except KeyError as exc:
        raise ValueError(f"Unknown controller role: {role}") from exc
    return AuthContext(role=role, permissions=permissions)
