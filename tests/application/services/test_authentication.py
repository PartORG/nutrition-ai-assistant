"""
Unit tests for application/services/authentication.py — AuthenticationService.

Uses AsyncMock for repositories, real bcrypt for hashing, real python-jose for JWT.
No database is touched.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from jose import jwt

from application.dto import RegisterRequest, LoginRequest, AuthToken
from application.services.authentication import AuthenticationService
from domain.entities import User, Authentication
from domain.exceptions import AuthenticationError, DuplicateLoginError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

JWT_SECRET = "test-secret-key"
JWT_ALGORITHM = "HS256"


def make_service(
    auth_repo=None,
    user_repo=None,
) -> AuthenticationService:
    if user_repo is None:
        user_repo = AsyncMock()
        user_repo.save = AsyncMock(return_value=1)
        user_repo.get_by_id = AsyncMock(return_value=User(id=1, name="Alice"))
    if auth_repo is None:
        auth_repo = AsyncMock()
        auth_repo.get_by_login = AsyncMock(return_value=None)
        auth_repo.save = AsyncMock(return_value=1)

    return AuthenticationService(
        user_repo=user_repo,
        auth_repo=auth_repo,
        jwt_secret=JWT_SECRET,
        jwt_expiry_hours=1,
        jwt_algorithm=JWT_ALGORITHM,
    )


# ---------------------------------------------------------------------------
# register()
# ---------------------------------------------------------------------------

class TestRegister:
    async def test_returns_auth_token(self):
        svc = make_service()
        req = RegisterRequest(login="alice", password="password123")
        token = await svc.register(req)
        assert isinstance(token, AuthToken)
        assert token.access_token

    async def test_token_contains_user_id(self):
        svc = make_service()
        req = RegisterRequest(login="alice", password="password123")
        token = await svc.register(req)
        payload = jwt.decode(token.access_token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        assert payload["user_id"] == 1  # user_repo.save returns 1

    async def test_duplicate_login_raises(self):
        auth_repo = AsyncMock()
        auth_repo.get_by_login = AsyncMock(
            return_value=Authentication(login="alice", password="hashed", role="user", user_id=1)
        )
        svc = make_service(auth_repo=auth_repo)
        with pytest.raises(DuplicateLoginError, match="alice"):
            await svc.register(RegisterRequest(login="alice", password="pass"))

    async def test_user_repo_save_called(self):
        user_repo = AsyncMock()
        user_repo.save = AsyncMock(return_value=5)
        auth_repo = AsyncMock()
        auth_repo.get_by_login = AsyncMock(return_value=None)
        auth_repo.save = AsyncMock(return_value=5)

        svc = make_service(auth_repo=auth_repo, user_repo=user_repo)
        await svc.register(RegisterRequest(login="bob", password="pass"))
        user_repo.save.assert_awaited_once()

    async def test_password_is_hashed_not_stored_plaintext(self):
        import bcrypt
        captured = {}

        auth_repo = AsyncMock()
        auth_repo.get_by_login = AsyncMock(return_value=None)

        async def capture_save(auth: Authentication):
            captured["password"] = auth.password
            return 1

        auth_repo.save = capture_save
        user_repo = AsyncMock()
        user_repo.save = AsyncMock(return_value=1)

        svc = make_service(auth_repo=auth_repo, user_repo=user_repo)
        await svc.register(RegisterRequest(login="alice", password="plaintext"))

        assert captured["password"] != "plaintext"
        assert bcrypt.checkpw(b"plaintext", captured["password"].encode())


# ---------------------------------------------------------------------------
# login()
# ---------------------------------------------------------------------------

class TestLogin:
    def _make_hashed_auth(self, login: str, password: str) -> Authentication:
        import bcrypt
        hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
        return Authentication(
            id=1, login=login, password=hashed, role="user", user_id=10
        )

    async def test_valid_login_returns_token(self):
        stored_auth = self._make_hashed_auth("alice", "correct")
        auth_repo = AsyncMock()
        auth_repo.get_by_login = AsyncMock(return_value=stored_auth)
        svc = make_service(auth_repo=auth_repo)

        token = await svc.login(LoginRequest(login="alice", password="correct"))
        assert isinstance(token, AuthToken)
        assert token.user_id == 10

    async def test_wrong_password_raises(self):
        stored_auth = self._make_hashed_auth("alice", "correct")
        auth_repo = AsyncMock()
        auth_repo.get_by_login = AsyncMock(return_value=stored_auth)
        svc = make_service(auth_repo=auth_repo)

        with pytest.raises(AuthenticationError):
            await svc.login(LoginRequest(login="alice", password="wrong"))

    async def test_unknown_login_raises(self):
        auth_repo = AsyncMock()
        auth_repo.get_by_login = AsyncMock(return_value=None)
        svc = make_service(auth_repo=auth_repo)

        with pytest.raises(AuthenticationError):
            await svc.login(LoginRequest(login="nobody", password="pass"))


# ---------------------------------------------------------------------------
# verify_token()
# ---------------------------------------------------------------------------

class TestVerifyToken:
    def _get_token(self) -> tuple[AuthenticationService, str]:
        svc = make_service()
        token_obj = svc._create_token(user_id=7, role="user")
        return svc, token_obj.access_token

    def test_valid_token_returns_payload(self):
        svc, token = self._get_token()
        payload = svc.verify_token(token)
        assert payload["user_id"] == 7
        assert payload["role"] == "user"

    def test_invalid_token_raises(self):
        svc = make_service()
        with pytest.raises(AuthenticationError):
            svc.verify_token("not.a.valid.token")

    def test_tampered_token_raises(self):
        svc = make_service()
        _, token = self._get_token()
        tampered = token[:-5] + "XXXXX"
        with pytest.raises(AuthenticationError):
            svc.verify_token(tampered)

    def test_token_signed_with_wrong_secret_raises(self):
        svc_other = make_service()
        bad_token = svc_other._create_token(1, "user")
        svc = AuthenticationService(
            user_repo=AsyncMock(), auth_repo=AsyncMock(),
            jwt_secret="different-secret", jwt_algorithm=JWT_ALGORITHM,
        )
        with pytest.raises(AuthenticationError):
            svc.verify_token(bad_token.access_token)


# ---------------------------------------------------------------------------
# refresh_token()
# ---------------------------------------------------------------------------

class TestRefreshToken:
    async def test_refresh_issues_new_token(self):
        user_repo = AsyncMock()
        user_repo.get_by_id = AsyncMock(return_value=User(id=3, name="Charlie"))
        user_repo.save = AsyncMock(return_value=3)
        svc = make_service(user_repo=user_repo)

        original = svc._create_token(user_id=3, role="user")
        refreshed = await svc.refresh_token(original.access_token)
        assert isinstance(refreshed, AuthToken)
        assert refreshed.user_id == 3

    async def test_refresh_fails_if_user_deleted(self):
        user_repo = AsyncMock()
        user_repo.get_by_id = AsyncMock(return_value=None)
        svc = make_service(user_repo=user_repo)

        token = svc._create_token(user_id=99, role="user")
        with pytest.raises(AuthenticationError, match="no longer exists"):
            await svc.refresh_token(token.access_token)

    async def test_refresh_with_malformed_token_raises(self):
        svc = make_service()
        with pytest.raises(AuthenticationError):
            await svc.refresh_token("garbage.token.here")
