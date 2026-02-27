"""
application.services.chat_history - Conversation persistence service.

Manages saving/loading chat messages and conversation metadata.
Used by AgentExecutor to persist conversations to DB.
"""

from __future__ import annotations

import logging
from typing import Optional

from domain.entities import Conversation, ChatMessage
from domain.ports import ConversationRepository, ChatMessageRepository
from application.context import SessionContext

logger = logging.getLogger(__name__)


class ChatHistoryService:
    """Persists and retrieves conversation messages."""

    def __init__(
        self,
        conversation_repo: ConversationRepository,
        message_repo: ChatMessageRepository,
    ):
        self._conversation_repo = conversation_repo
        self._message_repo = message_repo

    async def ensure_conversation(self, ctx: SessionContext) -> None:
        """Create the conversation record if it doesn't exist yet."""
        existing = await self._conversation_repo.get_by_conversation_id(
            ctx.conversation_id,
        )
        if existing is None:
            conversation = Conversation(
                user_id=ctx.user_id,
                conversation_id=ctx.conversation_id,
                title="",
            )
            await self._conversation_repo.save(conversation)
            logger.info(
                "Created conversation %s for user %d",
                ctx.conversation_id, ctx.user_id,
            )

    async def save_user_message(
        self, ctx: SessionContext, content: str,
    ) -> int:
        """Save a user message and return its ID.

        On the very first message in a conversation, auto-sets the title
        to the first 60 chars of the message so the conversation list is
        meaningful in the Flutter UI.
        """
        msg = ChatMessage(
            user_id=ctx.user_id,
            conversation_id=ctx.conversation_id,
            role="user",
            content=content,
        )
        msg_id = await self._message_repo.save(msg)
        await self._conversation_repo.update_last_message(ctx.conversation_id)

        # Auto-title on first message
        conv = await self._conversation_repo.get_by_conversation_id(
            ctx.conversation_id,
        )
        if conv and not conv.title:
            title = content[:60].strip()
            if len(content) > 60:
                title += "…"
            await self._conversation_repo.update_title(ctx.conversation_id, title)

        return msg_id

    async def save_assistant_message(
        self, ctx: SessionContext, content: str,
    ) -> int:
        """Save an assistant message and return its ID."""
        msg = ChatMessage(
            user_id=ctx.user_id,
            conversation_id=ctx.conversation_id,
            role="assistant",
            content=content,
        )
        msg_id = await self._message_repo.save(msg)
        await self._conversation_repo.update_last_message(ctx.conversation_id)
        return msg_id

    async def load_history(
        self, conversation_id: str,
    ) -> list[ChatMessage]:
        """Load all messages for a conversation, ordered by created_at ASC."""
        return await self._message_repo.get_by_conversation(conversation_id)

    async def list_conversations(
        self, user_id: int,
    ) -> list[Conversation]:
        """List all conversations for a user, newest first."""
        return await self._conversation_repo.get_by_user(user_id)

    async def save_recipe_cache(self, ctx: SessionContext, cache_json: str) -> None:
        """Persist the last recipe list to the conversation so it survives reconnects."""
        try:
            await self._conversation_repo.update_recipe_cache(ctx.conversation_id, cache_json)
        except Exception:
            logger.exception(
                "Failed to save recipe cache for conversation %s", ctx.conversation_id
            )

    async def load_recipe_cache(self, conversation_id: str) -> Optional[str]:
        """Return the cached recipe list JSON for a conversation, or None."""
        try:
            return await self._conversation_repo.get_recipe_cache(conversation_id)
        except Exception:
            logger.exception(
                "Failed to load recipe cache for conversation %s", conversation_id
            )
            return None

    async def purge_old_data(self, user_id: int, cutoff_iso: str) -> None:
        """Soft-delete messages and conversations older than *cutoff_iso*.

        Called on every WebSocket connect to keep stored data bounded.
        Failures are logged but never re-raised — the session continues.
        """
        try:
            msgs = await self._message_repo.delete_old_for_user(user_id, cutoff_iso)
            convs = await self._conversation_repo.delete_old_for_user(user_id, cutoff_iso)
            if msgs or convs:
                logger.info(
                    "Purged %d old message(s) and %d old conversation(s) for user %d",
                    msgs, convs, user_id,
                )
        except Exception:
            logger.exception("Failed to purge old chat data for user %d", user_id)
