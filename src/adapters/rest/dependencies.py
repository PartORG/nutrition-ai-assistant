"""
Shared FastAPI dependencies.

- get_factory(): returns the initialized ServiceFactory (set at startup).
- get_current_user(): JWT bearer token extraction and validation.
"""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from factory import ServiceFactory
from application.context import SessionContext

# Module-level reference set by app lifespan
_factory: ServiceFactory | None = None


def set_factory(factory: ServiceFactory) -> None:
    global _factory
    _factory = factory


def get_factory() -> ServiceFactory:
    if _factory is None:
        raise RuntimeError("ServiceFactory not initialized.")
    return _factory


# --- JWT Bearer ---

_bearer_scheme = HTTPBearer()


@dataclass
class CurrentUser:
    """Extracted from JWT payload. Passed to route handlers."""
    user_id: int
    role: str


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer_scheme),
    factory: ServiceFactory = Depends(get_factory),
) -> CurrentUser:
    """Validate JWT and return CurrentUser. Raises 401 on failure."""
    auth_service = factory.create_authentication_service()
    try:
        payload = auth_service.verify_token(credentials.credentials)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return CurrentUser(
        user_id=payload["user_id"],
        role=payload.get("role", "user"),
    )


async def build_session_ctx(
    user_id: int,
    conversation_id: str,
    factory: ServiceFactory,
) -> SessionContext:
    """Create a SessionContext pre-populated with the user's latest profile data.

    Loads the most recent profile snapshot from DB so the recommendation
    pipeline and agent already know the user's health conditions and
    preferences without them having to re-state everything each request.
    """
    profile_svc = factory.create_profile_service()
    user_data = await profile_svc.load_user_context(user_id)
    return SessionContext(
        user_id=user_id,
        conversation_id=conversation_id,
        user_data=user_data,
    )
