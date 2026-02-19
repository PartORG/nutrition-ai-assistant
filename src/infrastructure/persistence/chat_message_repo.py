"""
infrastructure.persistence.chat_message_repo - SQLite chat message repository.

Stores individual conversation messages (user and assistant turns).
"""

from __future__ import annotations

import logging
from datetime import datetime

from domain.entities import ChatMessage
from infrastructure.persistence.connection import AsyncSQLiteConnection

logger = logging.getLogger(__name__)


class SQLiteChatMessageRepository:
    """Async SQLite implementation of ChatMessageRepository."""

    def __init__(self, connection: AsyncSQLiteConnection):
        self._conn = connection

    async def save(self, message: ChatMessage) -> int:
        now = datetime.now().isoformat()
        async with self._conn.acquire() as conn:
            cursor = await conn.execute(
                """INSERT INTO chat_messages
                   (user_id, conversation_id, role, content,
                    created_at, updated_at, deleted_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (message.user_id, message.conversation_id,
                 message.role, message.content, now, now, ""),
            )
            return cursor.lastrowid

    async def get_by_conversation(
        self, conversation_id: str,
    ) -> list[ChatMessage]:
        async with self._conn.acquire() as conn:
            rows = await conn.execute_fetchall(
                """SELECT * FROM chat_messages
                   WHERE conversation_id = ?
                   AND (deleted_at = '' OR deleted_at IS NULL)
                   ORDER BY created_at ASC""",
                (conversation_id,),
            )
            return [self._row_to_entity(r) for r in rows]

    async def get_by_user(self, user_id: int) -> list[ChatMessage]:
        async with self._conn.acquire() as conn:
            rows = await conn.execute_fetchall(
                """SELECT * FROM chat_messages
                   WHERE user_id = ?
                   AND (deleted_at = '' OR deleted_at IS NULL)
                   ORDER BY created_at ASC""",
                (user_id,),
            )
            return [self._row_to_entity(r) for r in rows]

    async def soft_delete(self, message_id: int) -> None:
        async with self._conn.acquire() as conn:
            await conn.execute(
                "UPDATE chat_messages SET deleted_at = ? WHERE id = ?",
                (datetime.now().isoformat(), message_id),
            )

    async def delete_old_for_user(self, user_id: int, cutoff_iso: str) -> int:
        """Soft-delete all messages for a user created before *cutoff_iso*.

        Returns the number of rows marked as deleted.
        """
        now = datetime.now().isoformat()
        async with self._conn.acquire() as conn:
            cursor = await conn.execute(
                """UPDATE chat_messages
                   SET deleted_at = ?
                   WHERE user_id = ?
                     AND (deleted_at = '' OR deleted_at IS NULL)
                     AND created_at < ?""",
                (now, user_id, cutoff_iso),
            )
            return cursor.rowcount

    @staticmethod
    def _row_to_entity(row) -> ChatMessage:
        return ChatMessage(
            id=row[0],
            user_id=row[1],
            conversation_id=row[2] or "",
            role=row[3] or "",
            content=row[4] or "",
            created_at=row[5] or "",
            updated_at=row[6] or "",
            deleted_at=row[7] or "",
        )
