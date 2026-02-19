"""
application.services.authentication - User registration and login.

Handles password hashing (bcrypt), JWT creation/verification,
and coordinating between UserRepository and AuthenticationRepository.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import bcrypt as _bcrypt
from jose import jwt, JWTError

from domain.entities import User, Authentication
from domain.ports import UserRepository, AuthenticationRepository
from domain.exceptions import AuthenticationError, DuplicateLoginError
from application.dto import RegisterRequest, LoginRequest, AuthToken

logger = logging.getLogger(__name__)


class AuthenticationService:
    """Handles registration, login, and JWT management."""

    def __init__(
        self,
        user_repo: UserRepository,
        auth_repo: AuthenticationRepository,
        jwt_secret: str,
        jwt_expiry_hours: int = 24,
        jwt_algorithm: str = "HS256",
    ):
        self._user_repo = user_repo
        self._auth_repo = auth_repo
        self._jwt_secret = jwt_secret
        self._jwt_expiry_hours = jwt_expiry_hours
        self._jwt_algorithm = jwt_algorithm
        # bcrypt is used directly (passlib is incompatible with bcrypt >= 4.0)

    async def register(self, request: RegisterRequest) -> AuthToken:
        """Create a new user + authentication record, return JWT."""
        existing = await self._auth_repo.get_by_login(request.login)
        if existing is not None:
            raise DuplicateLoginError(
                f"Login '{request.login}' is already taken."
            )

        user = User(
            name=request.name,
            surname=request.surname,
            age=request.age,
            gender=request.gender,
            caretaker=request.caretaker,
            user_name=request.login,
        )
        user_id = await self._user_repo.save(user)

        auth = Authentication(
            login=request.login,
            password=_bcrypt.hashpw(
                request.password.encode(), _bcrypt.gensalt(),
            ).decode(),
            role="user",
            user_id=user_id,
        )
        await self._auth_repo.save(auth)

        logger.info("Registered user %d with login '%s'", user_id, request.login)
        return self._create_token(user_id, "user")

    async def login(self, request: LoginRequest) -> AuthToken:
        """Verify credentials and return JWT."""
        auth = await self._auth_repo.get_by_login(request.login)
        if auth is None:
            raise AuthenticationError("Invalid login or password.")

        if not _bcrypt.checkpw(
            request.password.encode(), auth.password.encode(),
        ):
            raise AuthenticationError("Invalid login or password.")

        logger.info("User %d logged in", auth.user_id)
        return self._create_token(auth.user_id, auth.role)

    def verify_token(self, token: str) -> dict[str, Any]:
        """Decode and validate a JWT. Returns the payload dict."""
        try:
            payload = jwt.decode(
                token, self._jwt_secret, algorithms=[self._jwt_algorithm],
            )
            if payload.get("user_id") is None:
                raise AuthenticationError("Invalid token payload.")
            return payload
        except JWTError as exc:
            raise AuthenticationError(f"Token verification failed: {exc}")

    async def refresh_token(self, token: str) -> AuthToken:
        """Issue a fresh token from an existing one (even if expired).

        Decodes the token without checking expiry, verifies the user still
        exists in DB, then re-issues a new token with a fresh expiry window.
        Raises AuthenticationError if the token is structurally invalid or
        the user has been deleted.
        """
        try:
            payload = jwt.decode(
                token,
                self._jwt_secret,
                algorithms=[self._jwt_algorithm],
                options={"verify_exp": False},
            )
        except JWTError as exc:
            raise AuthenticationError(f"Token refresh failed: {exc}")

        user_id = payload.get("user_id")
        if not user_id:
            raise AuthenticationError("Invalid token payload.")

        user = await self._user_repo.get_by_id(user_id)
        if user is None:
            raise AuthenticationError("User no longer exists.")

        role = payload.get("role", "user")
        logger.info("Token refreshed for user %d", user_id)
        return self._create_token(user_id, role)

    def _create_token(self, user_id: int, role: str) -> AuthToken:
        expire = datetime.now(timezone.utc) + timedelta(hours=self._jwt_expiry_hours)
        payload = {
            "user_id": user_id,
            "role": role,
            "exp": expire,
        }
        token = jwt.encode(payload, self._jwt_secret, algorithm=self._jwt_algorithm)
        return AuthToken(access_token=token, user_id=user_id, role=role)
