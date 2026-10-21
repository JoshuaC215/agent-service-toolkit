"""Pure ASGI identity binding, including streamed tool execution."""

import secrets

from fastapi import APIRouter, HTTPException
from starlette.datastructures import Headers
from starlette.responses import JSONResponse

from core.settings import settings
from db.access import AccessDenied, Principal, current_principal, principal, require_staff


def resolve_identity(user_key: str | None, staff_key: str | None) -> Principal | None:
    groups = [("staff", settings.APPROVAL_REVIEWERS), ("customer", settings.CUSTOMER_ACCESS_KEYS)]
    all_keys = [value.get_secret_value() for _, mapping in groups for value in mapping.values()]
    if (
        len(set(all_keys)) != len(all_keys)
        or settings.APPROVAL_REVIEWERS.keys() & settings.CUSTOMER_ACCESS_KEYS.keys()
    ):
        raise AccessDenied("DUPLICATE_IDENTITY_KEYS")
    for role, mapping in groups:
        supplied = staff_key if role == "staff" else user_key
        if supplied is None:
            continue
        for user_id, secret in mapping.items():
            value = secret.get_secret_value()
            if not user_id.strip() or len(user_id) > 64 or len(value) < 32:
                continue
            if secrets.compare_digest(supplied.encode(), value.encode()):
                return Principal(user_id, role)
    return None


class IdentityMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        path = scope["path"]
        public = path in (
            "/info",
            "/health",
            "/docs",
            "/openapi.json",
            "/redoc",
            "/docs/oauth2-redirect",
        )
        headers = Headers(scope=scope)
        try:
            identity = resolve_identity(headers.get("X-User-Key"), headers.get("X-Approval-Key"))
        except AccessDenied:
            return await JSONResponse({"detail": "IDENTITY_CONFIGURATION_INVALID"}, 503)(
                scope, receive, send
            )
        if not public and identity is None:
            return await JSONResponse({"detail": "LOGIN_REQUIRED"}, 401)(scope, receive, send)
        if identity and identity.role == "customer":
            # Alternate protocols and agents are not yet isolated for end users.
            if (
                path.startswith("/agui")
                or path == "/feedback"
                or (path.endswith(("/invoke", "/stream")) and not path.startswith("/order-agent/"))
            ):
                return await JSONResponse({"detail": "CUSTOMER_AGENT_NOT_ALLOWED"}, 403)(
                    scope, receive, send
                )
        token = current_principal.set(identity)
        try:
            await self.app(scope, receive, send)
        finally:
            current_principal.reset(token)


router = APIRouter(prefix="/auth", tags=["identity"])


@router.get("/me")
async def me():
    try:
        identity = principal()
        return {"user_id": identity.user_id, "role": identity.role}
    except AccessDenied as exc:
        raise HTTPException(401, str(exc)) from exc


@router.get("/customers")
async def customers():
    require_staff()
    return {"user_ids": sorted(settings.CUSTOMER_ACCESS_KEYS)}
