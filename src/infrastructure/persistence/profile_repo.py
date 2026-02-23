"""
infrastructure.persistence.profile_repo - SQLite user profile history repository.

Implements ProfileRepository port. Extracted from db.py (lines 502-568).
"""

from __future__ import annotations

import logging
from datetime import datetime

from domain.entities import UserProfileHistory
from infrastructure.persistence.connection import AsyncSQLiteConnection

logger = logging.getLogger(__name__)


class SQLiteProfileRepository:
    """Async SQLite implementation of ProfileRepository."""

    def __init__(self, connection: AsyncSQLiteConnection):
        self._conn = connection

    async def save(self, profile: UserProfileHistory) -> int:
        now = datetime.now().isoformat()
        async with self._conn.acquire() as conn:
            cursor = await conn.execute(
                """INSERT INTO user_profile_history
                   (preferences, user_id, health_condition, restrictions,
                    created_at, updated_at, deleted_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (profile.preferences, profile.user_id, profile.health_condition,
                 profile.restrictions, now, now, ""),
            )
            return cursor.lastrowid

    async def get_by_user(self, user_id: int) -> list[UserProfileHistory]:
        async with self._conn.acquire() as conn:
            rows = await conn.execute_fetchall(
                "SELECT * FROM user_profile_history WHERE user_id = ? AND (deleted_at = '' OR deleted_at IS NULL) ORDER BY created_at DESC",
                (user_id,),
            )
            return [self._row_to_profile(r) for r in rows]

    async def soft_delete(self, history_id: int) -> None:
        async with self._conn.acquire() as conn:
            await conn.execute(
                "UPDATE user_profile_history SET deleted_at = ? WHERE id = ?",
                (datetime.now().isoformat(), history_id),
            )

    @staticmethod
    def _row_to_profile(row) -> UserProfileHistory:
        return UserProfileHistory(
            id=row[0], preferences=row[1] or "", user_id=row[2],
            health_condition=row[3] or "", restrictions=row[4] or "",
            created_at=row[5] or "", updated_at=row[6] or "",
            deleted_at=row[7] or "",
        )
