"""
infrastructure.persistence.nutrition_repo - SQLite nutrition history repository.

Implements NutritionRepository port. Extracted from db.py (lines 646-713).
"""

from __future__ import annotations

import logging
from datetime import datetime

from domain.entities import NutritionHistory
from infrastructure.persistence.connection import AsyncSQLiteConnection

logger = logging.getLogger(__name__)


class SQLiteNutritionRepository:
    """Async SQLite implementation of NutritionRepository."""

    def __init__(self, connection: AsyncSQLiteConnection):
        self._conn = connection

    async def save(self, history: NutritionHistory) -> int:
        now = datetime.now().isoformat()
        async with self._conn.acquire() as conn:
            cursor = await conn.execute(
                """INSERT INTO nutrition_history
                   (user_id, recipe_id, calories, protein, fat, carbohydrates,
                    fiber, sugar, sodium, created_at, updated_at, deleted_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (history.user_id, history.recipe_id, history.calories,
                 history.protein, history.fat, history.carbohydrates,
                 history.fiber, history.sugar, history.sodium,
                 now, now, ""),
            )
            return cursor.lastrowid

    async def get_by_user(self, user_id: int) -> list[NutritionHistory]:
        async with self._conn.acquire() as conn:
            rows = await conn.execute_fetchall(
                "SELECT * FROM nutrition_history WHERE user_id = ? AND (deleted_at = '' OR deleted_at IS NULL) ORDER BY created_at DESC",
                (user_id,),
            )
            return [self._row_to_history(r) for r in rows]

    async def get_today_by_user(self, user_id: int) -> list[NutritionHistory]:
        """Return all nutrition records saved today (local calendar date)."""
        today = datetime.now().strftime("%Y-%m-%d")
        async with self._conn.acquire() as conn:
            rows = await conn.execute_fetchall(
                """SELECT * FROM nutrition_history
                   WHERE user_id = ?
                     AND (deleted_at = '' OR deleted_at IS NULL)
                     AND created_at LIKE ?
                   ORDER BY created_at DESC""",
                (user_id, f"{today}%"),
            )
            return [self._row_to_history(r) for r in rows]

    async def soft_delete(self, history_id: int) -> None:
        async with self._conn.acquire() as conn:
            await conn.execute(
                "UPDATE nutrition_history SET deleted_at = ? WHERE id = ?",
                (datetime.now().isoformat(), history_id),
            )

    @staticmethod
    def _row_to_history(row) -> NutritionHistory:
        return NutritionHistory(
            id=row[0], user_id=row[1], recipe_id=row[2],
            calories=row[3] or 0.0, protein=row[4] or 0.0,
            fat=row[5] or 0.0, carbohydrates=row[6] or 0.0,
            fiber=row[7] or 0.0, sugar=row[8] or 0.0,
            sodium=row[9] or 0.0, created_at=row[10] or "",
            updated_at=row[11] or "", deleted_at=row[12] or "",
        )
