"""
agent.memory - Per-session conversation memory with optional DB persistence.

Stores messages as a plain list[BaseMessage] — avoids the deprecated
ConversationBufferMemory. The list is passed directly to the agent
via the 'chat_history' input key on every invoke() call.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

if TYPE_CHECKING:
    from application.services.chat_history import ChatHistoryService

logger = logging.getLogger(__name__)


class ConversationMemory:
    """Per-session conversation memory.

    NOT global — each session/user gets their own instance.
    Optionally backed by ChatHistoryService for DB persistence.
    """

    def __init__(
        self,
        max_messages: int = 50,
        chat_history_service: ChatHistoryService | None = None,
    ):
        self._max_messages = max_messages
        self._chat_history_service = chat_history_service
        self._messages: list[BaseMessage] = []

    @property
    def messages(self) -> list[BaseMessage]:
        """Current history — pass as 'chat_history' to agent invoke()."""
        return self._messages

    @property
    def has_persistence(self) -> bool:
        return self._chat_history_service is not None

    def add_user_message(self, content: str) -> None:
        self._messages.append(HumanMessage(content=content))
        self._trim()

    def add_ai_message(self, content: str) -> None:
        self._messages.append(AIMessage(content=content))
        self._trim()

    def _trim(self) -> None:
        if len(self._messages) > self._max_messages:
            self._messages = self._messages[-self._max_messages:]

    async def load_from_db(self, conversation_id: str) -> int:
        """Load previous messages from DB into the in-memory list.

        Returns the number of messages loaded.
        """
        if not self._chat_history_service:
            return 0

        messages = await self._chat_history_service.load_history(conversation_id)

        for msg in messages[-self._max_messages:]:
            if msg.role == "user":
                self._messages.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                self._messages.append(AIMessage(content=msg.content))

        loaded = min(len(messages), self._max_messages)
        if loaded > 0:
            logger.info(
                "Loaded %d messages from DB for conversation %s",
                loaded, conversation_id,
            )
        return loaded

    def clear(self) -> None:
        """Clear all conversation history."""
        self._messages.clear()
