"""
infrastructure.persistence.recipe_repo - SQLite recipe history repository.

Implements RecipeRepository port. Extracted from db.py (lines 574-640).
"""

from __future__ import annotations

import logging
from datetime import datetime

from domain.entities import RecipeHistory
from infrastructure.persistence.connection import AsyncSQLiteConnection

logger = logging.getLogger(__name__)


class SQLiteRecipeRepository:
    """Async SQLite implementation of RecipeRepository."""

    def __init__(self, connection: AsyncSQLiteConnection):
        self._conn = connection

    async def save(self, history: RecipeHistory) -> int:
        now = datetime.now().isoformat()
        async with self._conn.acquire() as conn:
            cursor = await conn.execute(
                """INSERT INTO recipe_history
                   (user_id, recipe_id, recipe_name, cook_instructions,
                    servings, ingredients, prep_time, created_at, updated_at, deleted_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (history.user_id, history.recipe_id, history.recipe_name,
                 history.cook_instructions, history.servings, history.ingredients,
                 history.prep_time, now, now, ""),
            )
            return cursor.lastrowid

    async def get_by_user(self, user_id: int) -> list[RecipeHistory]:
        async with self._conn.acquire() as conn:
            rows = await conn.execute_fetchall(
                "SELECT * FROM recipe_history WHERE user_id = ? AND (deleted_at = '' OR deleted_at IS NULL) ORDER BY created_at DESC",
                (user_id,),
            )
            return [self._row_to_history(r) for r in rows]

    async def soft_delete(self, history_id: int) -> None:
        async with self._conn.acquire() as conn:
            await conn.execute(
                "UPDATE recipe_history SET deleted_at = ? WHERE id = ?",
                (datetime.now().isoformat(), history_id),
            )

    @staticmethod
    def _row_to_history(row) -> RecipeHistory:
        return RecipeHistory(
            id=row[0], user_id=row[1], recipe_id=row[2],
            recipe_name=row[3] or "", cook_instructions=row[4] or "",
            servings=row[5] or 0, ingredients=row[6] or "",
            prep_time=row[7] or "", created_at=row[8] or "",
            updated_at=row[9] or "", deleted_at=row[10] or "",
        )
