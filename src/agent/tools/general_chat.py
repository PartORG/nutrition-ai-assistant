"""
agent.tools.general_chat - Handler for non-food general conversation.

Routes greetings, small talk, and off-topic messages away from
search_recipes so the LLM can respond naturally without triggering
the recipe pipeline.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from application.context import SessionContext
from agent.tools.base import BaseTool, ToolResult


class GeneralChatInput(BaseModel):
    """Input schema for the general_chat tool."""

    message: str = Field(
        default="",
        description="The user's general question or conversational message",
    )
    # Some LLMs send 'query' instead of 'message' — accepted as alias
    query: str = Field(default="", exclude=True, description="Alias for message")


class GeneralChatTool(BaseTool):
    """Handle greetings, small talk, and off-topic requests."""

    name = "general_chat"
    description = (
        "Use for greetings and small talk ('How are you?', 'Hello!', 'Good morning'), "
        "and for requests the app cannot fulfill "
        "('Can you open Calendar?', 'What time is it?', 'Write me a poem'). "
        "Do NOT use for nutrition knowledge questions — answer those directly. "
        "Do NOT use for recipe requests — use search_recipes for those. "
        "NEVER call search_recipes for greetings or off-topic messages — "
        "call this tool instead."
    )

    def get_schema(self) -> type[BaseModel]:
        return GeneralChatInput

    async def execute(
        self,
        ctx: SessionContext,
        message: str = "",
        **kwargs,
    ) -> ToolResult:
        """Signal the LLM to respond naturally without entering the recipe pipeline."""
        return ToolResult(
            output=(
                "This is general conversation, not a recipe or nutrition search request. "
                "Respond naturally, warmly, and briefly as a helpful nutrition assistant. "
                "You may gently mention what the app can help with, but keep the tone light "
                "and conversational — do not force a nutrition redirect."
            )
        )
