"""
agent.tools.nutrition_advice - General condition-based nutritional advice.

Routes third-person health-condition dietary questions to the Medical RAG
so the user gets evidence-based advice without triggering the recipe
pipeline or writing anything to the database.
"""

from __future__ import annotations

import asyncio
import logging

from pydantic import BaseModel, Field

from application.context import SessionContext
from agent.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class NutritionAdviceInput(BaseModel):
    """Input schema for the nutrition_advice tool."""

    query: str = Field(
        default="",
        description="The user's EXACT original message about a health condition, copied verbatim.",
    )
    # Some LLMs send 'message' instead of 'query' — accepted as alias
    message: str = Field(default="", exclude=True, description="Alias for query")


class NutritionAdviceTool(BaseTool):
    """Provide general dietary advice for specific health conditions via Medical RAG."""

    name = "nutrition_advice"
    description = (
        "Use when the user asks GENERALLY (third person, not about themselves) "
        "how a person with a specific health condition or disease should eat. "
        "Examples: 'How should someone with diabetes eat?', "
        "'What diet is recommended for a person with high cholesterol?', "
        "'What foods should people with celiac disease avoid?'. "
        "Do NOT use when the user says 'I have...' (personal condition → use search_recipes). "
        "Do NOT use for recipe requests → use search_recipes. "
        "Do NOT use for general nutrition knowledge without a specific condition "
        "→ answer directly without a tool."
    )

    def __init__(self, medical_rag):
        self._medical_rag = medical_rag

    def get_schema(self) -> type[BaseModel]:
        return NutritionAdviceInput

    async def execute(
        self,
        ctx: SessionContext,
        query: str = "",
        **kwargs,
    ) -> ToolResult:
        """Query the Medical RAG for condition-based dietary advice.

        Uses the original verbatim user message stored in ctx.scratch
        rather than the LLM-generated query argument — same pattern as
        SearchRecipesTool.

        No database reads or writes occur in this path.
        """
        actual_query = ctx.scratch.get("original_query") or query
        logger.info(
            "NutritionAdviceTool querying Medical RAG for user %d: %s",
            ctx.user_id, actual_query[:80],
        )

        try:
            loop = asyncio.get_event_loop()
            answer = await loop.run_in_executor(
                None, self._medical_rag.ask, actual_query,
            )
        except Exception:
            logger.exception("Medical RAG query failed in NutritionAdviceTool")
            answer = (
                "I'm sorry, I couldn't retrieve nutritional advice right now. "
                "Please try again later."
            )

        return ToolResult(output=answer)
