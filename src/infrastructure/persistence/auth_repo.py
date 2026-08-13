"""
infrastructure.persistence.auth_repo - SQLite authentication repository.

Implements AuthenticationRepository port.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime

from domain.entities import Authentication
from domain.exceptions import DuplicateLoginError
from infrastructure.persistence.connection import AsyncSQLiteConnection

logger = logging.getLogger(__name__)


class SQLiteAuthenticationRepository:
    """Async SQLite implementation of AuthenticationRepository."""

    def __init__(self, connection: AsyncSQLiteConnection):
        self._conn = connection

    async def save(self, auth: Authentication) -> int:
        now = datetime.now().isoformat()
        try:
            async with self._conn.acquire() as conn:
                cursor = await conn.execute(
                    """INSERT INTO authentication
                       (login, password, role, user_id, created_at, updated_at, deleted_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (auth.login, auth.password, auth.role or "user",
                     auth.user_id, now, now, ""),
                )
                return cursor.lastrowid
        except sqlite3.IntegrityError as exc:
            # `login` is UNIQUE. Two simultaneous registrations both pass the
            # service's get_by_login pre-check (separate transactions) and the
            # loser lands here. Translate at the boundary so the application
            # layer never has to know about sqlite — but only for the
            # constraint we mean: a NOT NULL or FK failure is not a duplicate
            # login, and reporting it as one makes the caller soft-delete a
            # user over an unrelated error.
            if "UNIQUE" not in str(exc):
                raise
            raise DuplicateLoginError(
                f"Login '{auth.login}' is already taken."
            ) from exc

    async def get_by_login(self, login: str) -> Authentication | None:
        async with self._conn.acquire() as conn:
            rows = await conn.execute_fetchall(
                """SELECT * FROM authentication
                   WHERE login = ? AND (deleted_at = '' OR deleted_at IS NULL)""",
                (login,),
            )
            return self._row_to_entity(rows[0]) if rows else None

    async def get_by_user_id(self, user_id: int) -> Authentication | None:
        async with self._conn.acquire() as conn:
            rows = await conn.execute_fetchall(
                """SELECT * FROM authentication
                   WHERE user_id = ? AND (deleted_at = '' OR deleted_at IS NULL)""",
                (user_id,),
            )
            return self._row_to_entity(rows[0]) if rows else None

    async def soft_delete(self, auth_id: int) -> None:
        async with self._conn.acquire() as conn:
            await conn.execute(
                "UPDATE authentication SET deleted_at = ? WHERE id = ?",
                (datetime.now().isoformat(), auth_id),
            )

    @staticmethod
    def _row_to_entity(row) -> Authentication:
        # Column order from DDL: id, login, password, role, created_at, deleted_at, updated_at, user_id
        return Authentication(
            id=row[0],
            login=row[1],
            password=row[2],
            role=row[3] or "user",
            created_at=row[4] or "",
            deleted_at=row[5] or "",
            updated_at=row[6] or "",
            user_id=row[7],
        )
