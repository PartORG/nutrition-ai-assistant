"""
application.context - Request-scoped session context.

Replaces the global AgentState singleton from host_agent.py.
Every function receives its context explicitly. Two concurrent users
get two different SessionContext instances — no race conditions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4


@dataclass
class SessionContext:
    """Per-request/session context passed through all layers.

    Attributes:
        user_id:          Authenticated user ID (provided by adapter).
        user_data:        User profile info (name, conditions, restrictions, etc.).
        conversation_id:  Unique per conversation session.
        request_id:       Unique per request, for tracing/logging.
        scratch:          Request-scoped scratchpad for inter-tool data sharing.
                          E.g. scratch["last_recommendations"] stores typed
                          RecommendationResult set by search tool, read by save tool.
    """
    user_id: int
    conversation_id: str
    user_data: dict[str, Any] = field(default_factory=dict)
    request_id: str = field(default_factory=lambda: uuid4().hex)
    scratch: dict[str, Any] = field(default_factory=dict)

    def new_request(self) -> None:
        """Reset per-request state for a new request within the same session.

        scratch is intentionally NOT cleared — tools like save_recipe need
        to read last_recommendations that were stored in a previous turn.
        """
        self.request_id = uuid4().hex
