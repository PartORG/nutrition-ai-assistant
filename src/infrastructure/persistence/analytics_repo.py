"""
infrastructure.persistence.analytics_repo - Read-only analytics queries.

Not mapped to a domain port (analytics are infrastructure-only read-only
aggregations, not domain operations). Called directly from the factory.
"""

from __future__ import annotations

import logging

from infrastructure.persistence.connection import AsyncSQLiteConnection

logger = logging.getLogger(__name__)


class SQLiteAnalyticsRepository:
    """Runs aggregate queries for the /analytics endpoint."""

    def __init__(self, connection: AsyncSQLiteConnection):
        self._conn = connection

    async def get_overview(self) -> dict:
        """Return high-level counts: users, conversations, messages."""
        async with self._conn.acquire() as conn:
            rows = await conn.execute_fetchall(
                """SELECT
                    (SELECT COUNT(*) FROM users WHERE deleted_at = '' OR deleted_at IS NULL),
                    (SELECT COUNT(*) FROM conversations WHERE deleted_at = '' OR deleted_at IS NULL),
                    (SELECT COUNT(*) FROM chat_messages WHERE deleted_at = '' OR deleted_at IS NULL),
                    (SELECT COUNT(*) FROM recipe_history WHERE deleted_at = '' OR deleted_at IS NULL)
                """,
            )
            row = rows[0]
            return {
                "total_users": row[0] or 0,
                "total_conversations": row[1] or 0,
                "total_messages": row[2] or 0,
                "total_saved_recipes": row[3] or 0,
            }

    async def get_top_recipes(self, limit: int = 5) -> list[dict]:
        """Most saved recipe names across all users."""
        async with self._conn.acquire() as conn:
            rows = await conn.execute_fetchall(
                """SELECT recipe_name, COUNT(*) AS cnt
                   FROM recipe_history
                   WHERE deleted_at = '' OR deleted_at IS NULL
                   GROUP BY recipe_name
                   ORDER BY cnt DESC
                   LIMIT ?""",
                (limit,),
            )
            return [{"recipe": r[0], "saves": r[1]} for r in rows]

    async def get_common_conditions(self, limit: int = 5) -> list[dict]:
        """Most frequently reported health conditions from profile snapshots."""
        async with self._conn.acquire() as conn:
            rows = await conn.execute_fetchall(
                """SELECT health_condition
                   FROM user_profile_history
                   WHERE (deleted_at = '' OR deleted_at IS NULL)
                     AND health_condition != ''""",
            )
        # Health conditions are comma-separated strings — split and count
        counts: dict[str, int] = {}
        for row in rows:
            for cond in row[0].split(","):
                cond = cond.strip().lower()
                if cond:
                    counts[cond] = counts.get(cond, 0) + 1
        top = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:limit]
        return [{"condition": k, "count": v} for k, v in top]

    async def get_recent_conversations(self, limit: int = 10) -> list[dict]:
        """Most recently active conversations across all users."""
        async with self._conn.acquire() as conn:
            rows = await conn.execute_fetchall(
                """SELECT conversation_id, title, last_message_at, user_id
                   FROM conversations
                   WHERE deleted_at = '' OR deleted_at IS NULL
                   ORDER BY last_message_at DESC
                   LIMIT ?""",
                (limit,),
            )
            return [
                {
                    "conversation_id": r[0],
                    "title": r[1] or "(untitled)",
                    "last_message_at": r[2],
                    "user_id": r[3],
                }
                for r in rows
            ]

    async def get_user_dashboard(self, user_id: int) -> dict:
        """Per-user dashboard: overview counts, avg nutrition, recent saved recipes."""
        async with self._conn.acquire() as conn:
            # User-scoped activity counts
            overview_rows = await conn.execute_fetchall(
                """SELECT
                    (SELECT COUNT(*) FROM conversations
                     WHERE user_id = ? AND (deleted_at = '' OR deleted_at IS NULL)),
                    (SELECT COUNT(*) FROM chat_messages
                     WHERE user_id = ? AND (deleted_at = '' OR deleted_at IS NULL)),
                    (SELECT COUNT(*) FROM recipe_history
                     WHERE user_id = ? AND (deleted_at = '' OR deleted_at IS NULL))
                """,
                (user_id, user_id, user_id),
            )
            row = overview_rows[0]
            overview = {
                "total_conversations": row[0] or 0,
                "total_messages": row[1] or 0,
                "saved_recipes": row[2] or 0,
            }

            # Average nutrition across all saved meals for this user
            avg_rows = await conn.execute_fetchall(
                """SELECT
                    AVG(calories), AVG(protein), AVG(fat),
                    AVG(carbohydrates), AVG(fiber), AVG(sugar), AVG(sodium),
                    COUNT(*)
                   FROM nutrition_history
                   WHERE user_id = ? AND (deleted_at = '' OR deleted_at IS NULL)
                """,
                (user_id,),
            )
            nutrition_avg = None
            if avg_rows and avg_rows[0][0] is not None:
                ar = avg_rows[0]
                nutrition_avg = {
                    "calories": round(ar[0] or 0, 1),
                    "protein_g": round(ar[1] or 0, 1),
                    "fat_g": round(ar[2] or 0, 1),
                    "carbs_g": round(ar[3] or 0, 1),
                    "fiber_g": round(ar[4] or 0, 1),
                    "sugar_g": round(ar[5] or 0, 1),
                    "sodium_mg": round(ar[6] or 0, 1),
                    "meal_count": ar[7] or 0,
                }

            # Recent saved recipes joined with their nutrition (last 10)
            recipe_rows = await conn.execute_fetchall(
                """SELECT rh.id, rh.recipe_name, rh.created_at,
                          nh.calories, nh.protein, nh.fat, nh.carbohydrates, nh.fiber
                   FROM recipe_history rh
                   LEFT JOIN nutrition_history nh
                       ON nh.recipe_id = rh.id
                      AND nh.user_id = rh.user_id
                      AND (nh.deleted_at = '' OR nh.deleted_at IS NULL)
                   WHERE rh.user_id = ? AND (rh.deleted_at = '' OR rh.deleted_at IS NULL)
                   ORDER BY rh.created_at DESC
                   LIMIT 10
                """,
                (user_id,),
            )
            recent_recipes = [
                {
                    "id": r[0],
                    "recipe_name": r[1] or "",
                    "saved_at": r[2] or "",
                    "calories": r[3],
                    "protein_g": r[4],
                    "fat_g": r[5],
                    "carbs_g": r[6],
                    "fiber_g": r[7],
                }
                for r in recipe_rows
            ]

        return {
            "overview": overview,
            "nutrition_avg": nutrition_avg,
            "recent_recipes": recent_recipes,
        }
