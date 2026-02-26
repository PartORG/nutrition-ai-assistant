"""
infrastructure.persistence.auth_repo - SQLite authentication repository.

Implements AuthenticationRepository port.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from domain.entities import Authentication
from infrastructure.persistence.connection import AsyncSQLiteConnection

logger = logging.getLogger(__name__)


class SQLiteAuthenticationRepository:
    """Async SQLite implementation of AuthenticationRepository."""

    def __init__(self, connection: AsyncSQLiteConnection):
        self._conn = connection

    async def save(self, auth: Authentication) -> int:
        now = datetime.now().isoformat()
        async with self._conn.acquire() as conn:
            cursor = await conn.execute(
                """INSERT INTO authentication
                   (login, password, role, user_id, created_at, updated_at, deleted_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (auth.login, auth.password, auth.role or "user",
                 auth.user_id, now, now, ""),
            )
            return cursor.lastrowid

    async def get_by_login(self, login: str) -> Optional[Authentication]:
        async with self._conn.acquire() as conn:
            rows = await conn.execute_fetchall(
                """SELECT * FROM authentication
                   WHERE login = ? AND (deleted_at = '' OR deleted_at IS NULL)""",
                (login,),
            )
            return self._row_to_entity(rows[0]) if rows else None

    async def get_by_user_id(self, user_id: int) -> Optional[Authentication]:
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
