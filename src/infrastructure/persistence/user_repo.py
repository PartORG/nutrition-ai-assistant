"""
infrastructure.persistence.user_repo - SQLite user repository.

Implements UserRepository port. Extracted from db.py (lines 226-306).
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from domain.entities import User
from infrastructure.persistence.connection import AsyncSQLiteConnection

logger = logging.getLogger(__name__)


class SQLiteUserRepository:
    """Async SQLite implementation of UserRepository."""

    def __init__(self, connection: AsyncSQLiteConnection):
        self._conn = connection

    async def get_by_id(self, user_id: int) -> Optional[User]:
        async with self._conn.acquire() as conn:
            row = await conn.execute_fetchall(
                "SELECT * FROM users WHERE id = ? AND deleted_at = ''",
                (user_id,),
            )
            if not row:
                return None
            return self._row_to_user(row[0])

    async def get_by_name(self, name: str, surname: str) -> Optional[User]:
        async with self._conn.acquire() as conn:
            rows = await conn.execute_fetchall(
                "SELECT * FROM users WHERE name = ? AND surname = ? ORDER BY updated_at DESC LIMIT 1",
                (name, surname),
            )
            if not rows:
                return None
            return self._row_to_user(rows[0])

    async def save(self, user: User) -> int:
        now = datetime.now().isoformat()
        async with self._conn.acquire() as conn:
            cursor = await conn.execute(
                """INSERT INTO users (name, surname, user_name, caretaker, age, gender, created_at, updated_at, deleted_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (user.name, user.surname, user.user_name, user.caretaker,
                 user.age, user.gender, now, now, ""),
            )
            return cursor.lastrowid

    async def update(self, user_id: int, field: str, value: object) -> None:
        allowed_fields = {"name", "surname", "user_name", "caretaker", "age", "gender"}
        if field not in allowed_fields:
            raise ValueError(f"Invalid field '{field}'. Allowed: {allowed_fields}")
        if isinstance(value, list):
            value = ", ".join(map(str, value))

        async with self._conn.acquire() as conn:
            await conn.execute(
                f"UPDATE users SET {field} = ?, updated_at = ? WHERE id = ?",
                (value, datetime.now().isoformat(), user_id),
            )

    async def soft_delete(self, user_id: int) -> None:
        async with self._conn.acquire() as conn:
            await conn.execute(
                "UPDATE users SET deleted_at = ? WHERE id = ?",
                (datetime.now().isoformat(), user_id),
            )

    @staticmethod
    def _row_to_user(row) -> User:
        return User(
            id=row[0], name=row[1], surname=row[2], user_name=row[3],
            caretaker=row[4], created_at=row[5] or "", updated_at=row[6] or "",
            deleted_at=row[7] or "", age=row[8] or 0, gender=row[9] or "",
        )
