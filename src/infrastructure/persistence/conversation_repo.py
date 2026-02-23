"""
infrastructure.persistence.conversation_repo - SQLite conversation repository.

Stores conversation metadata (session ID, title, timestamps).
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from domain.entities import Conversation
from infrastructure.persistence.connection import AsyncSQLiteConnection

logger = logging.getLogger(__name__)


class SQLiteConversationRepository:
    """Async SQLite implementation of ConversationRepository."""

    def __init__(self, connection: AsyncSQLiteConnection):
        self._conn = connection

    async def save(self, conversation: Conversation) -> int:
        now = datetime.now().isoformat()
        async with self._conn.acquire() as conn:
            cursor = await conn.execute(
                """INSERT INTO conversations
                   (user_id, conversation_id, title, last_message_at,
                    created_at, updated_at, deleted_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (conversation.user_id, conversation.conversation_id,
                 conversation.title, now, now, now, ""),
            )
            return cursor.lastrowid

    async def get_by_user(self, user_id: int) -> list[Conversation]:
        async with self._conn.acquire() as conn:
            rows = await conn.execute_fetchall(
                """SELECT * FROM conversations
                   WHERE user_id = ? AND (deleted_at = '' OR deleted_at IS NULL)
                   ORDER BY last_message_at DESC""",
                (user_id,),
            )
            return [self._row_to_entity(r) for r in rows]

    async def get_by_conversation_id(
        self, conversation_id: str,
    ) -> Optional[Conversation]:
        async with self._conn.acquire() as conn:
            rows = await conn.execute_fetchall(
                """SELECT * FROM conversations
                   WHERE conversation_id = ?
                   AND (deleted_at = '' OR deleted_at IS NULL)""",
                (conversation_id,),
            )
            return self._row_to_entity(rows[0]) if rows else None

    async def update_last_message(self, conversation_id: str) -> None:
        now = datetime.now().isoformat()
        async with self._conn.acquire() as conn:
            await conn.execute(
                """UPDATE conversations
                   SET last_message_at = ?, updated_at = ?
                   WHERE conversation_id = ?""",
                (now, now, conversation_id),
            )

    async def update_title(self, conversation_id: str, title: str) -> None:
        async with self._conn.acquire() as conn:
            await conn.execute(
                "UPDATE conversations SET title = ? WHERE conversation_id = ?",
                (title, conversation_id),
            )

    async def soft_delete(self, conversation_id: str) -> None:
        async with self._conn.acquire() as conn:
            await conn.execute(
                "UPDATE conversations SET deleted_at = ? WHERE conversation_id = ?",
                (datetime.now().isoformat(), conversation_id),
            )

    async def delete_old_for_user(self, user_id: int, cutoff_iso: str) -> int:
        """Soft-delete all conversations for a user whose last_message_at is
        before *cutoff_iso*.

        Returns the number of rows marked as deleted.
        """
        now = datetime.now().isoformat()
        async with self._conn.acquire() as conn:
            cursor = await conn.execute(
                """UPDATE conversations
                   SET deleted_at = ?
                   WHERE user_id = ?
                     AND (deleted_at = '' OR deleted_at IS NULL)
                     AND last_message_at < ?""",
                (now, user_id, cutoff_iso),
            )
            return cursor.rowcount

    @staticmethod
    def _row_to_entity(row) -> Conversation:
        return Conversation(
            id=row[0],
            user_id=row[1],
            conversation_id=row[2] or "",
            title=row[3] or "",
            last_message_at=row[4] or "",
            created_at=row[5] or "",
            updated_at=row[6] or "",
            deleted_at=row[7] or "",
        )
