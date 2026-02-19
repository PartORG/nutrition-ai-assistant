"""
infrastructure.persistence.medical_repo - SQLite medical advice repository.

Implements MedicalRepository port. Extracted from db.py (lines 312-394).
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from domain.entities import MedicalAdvice
from infrastructure.persistence.connection import AsyncSQLiteConnection

logger = logging.getLogger(__name__)


class SQLiteMedicalRepository:
    """Async SQLite implementation of MedicalRepository."""

    def __init__(self, connection: AsyncSQLiteConnection):
        self._conn = connection

    async def save(self, advice: MedicalAdvice) -> int:
        now = datetime.now().isoformat()
        medical_text = advice.medical_advice
        if isinstance(medical_text, list):
            medical_text = "\n".join(medical_text)

        async with self._conn.acquire() as conn:
            cursor = await conn.execute(
                """INSERT INTO medical_advice
                   (health_condition, medical_advice, dietary_limit, avoid,
                    dietary_constraints, created_at, updated_at, deleted_at, user_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (advice.health_condition, medical_text, advice.dietary_limit,
                 advice.avoid, advice.dietary_constraints, now, now, "", advice.user_id),
            )
            return cursor.lastrowid

    async def get_by_user(self, user_id: int) -> list[MedicalAdvice]:
        async with self._conn.acquire() as conn:
            rows = await conn.execute_fetchall(
                "SELECT * FROM medical_advice WHERE user_id = ? AND (deleted_at = '' OR deleted_at IS NULL) ORDER BY created_at DESC",
                (user_id,),
            )
            return [self._row_to_advice(r) for r in rows]

    async def soft_delete(self, advice_id: int) -> None:
        async with self._conn.acquire() as conn:
            await conn.execute(
                "UPDATE medical_advice SET deleted_at = ? WHERE id = ?",
                (datetime.now().isoformat(), advice_id),
            )

    @staticmethod
    def _row_to_advice(row) -> MedicalAdvice:
        return MedicalAdvice(
            id=row[0], health_condition=row[1] or "", medical_advice=row[2] or "",
            dietary_limit=row[3] or "", avoid=row[4] or "",
            dietary_constraints=row[5] or "", created_at=row[6] or "",
            updated_at=row[7] or "", deleted_at=row[8] or "",
            user_id=row[9],
        )
