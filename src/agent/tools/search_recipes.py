"""
agent.tools.search_recipes - Recipe search tool.

Calls RecommendationService and stores typed results in ctx.scratch.
No markdown parsing needed — structured Recipe objects come from the service.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from application.context import SessionContext
from application.services.recommendation import RecommendationService
from agent.tools.base import BaseTool, ToolResult

_MEDICAL_DISCLAIMER = (
    "\n\n> ⚠️ **Medical disclaimer:** These are general nutritional recommendations "
    "based on available data. Always consult a healthcare provider or registered "
    "dietitian before making significant dietary changes, especially if you have "
    "a medical condition."
)


class SearchRecipesInput(BaseModel):
    """Input schema for the search_recipes tool."""
    query: str = Field(
        description="The user's EXACT original message, copied verbatim. Do NOT rephrase or summarise."
    )


class SearchRecipesTool(BaseTool):
    """Search for personalized recipe recommendations."""

    name = "search_recipes"
    description = (
        "Search for recipe recommendations based on the user's request. "
        "Use when user asks for recipes, meals, or food suggestions. "
        "Always pass the user's exact original message as 'query'."
    )

    def __init__(self, recommendation_service: RecommendationService):
        self._service = recommendation_service

    def get_schema(self) -> type[BaseModel]:
        return SearchRecipesInput

    async def execute(self, ctx: SessionContext, query: str = "", **kwargs) -> ToolResult:
        """Run the recommendation pipeline and return formatted results.

        Always uses the original verbatim user message stored in ctx.scratch
        rather than the LLM-generated query argument — this prevents the agent
        from silently rewriting the query before it reaches the intent parser
        and Medical RAG.
        """
        # Prefer the original user message stored by AgentExecutor.run()
        actual_query = ctx.scratch.get("original_query") or query
        result = await self._service.get_recommendations(ctx, actual_query)

        safe_recipes = result.safe_recipes
        output = result.safety_result.safe_recipes_markdown

        if safe_recipes:
            footer = (
                f"\n\n---\n"
                f"Found {len(safe_recipes)} recipes above!\n\n"
                f"What would you like to do?\n"
                f"- Cook one? (e.g., 'I'll cook recipe 2' or 'save the salmon')\n"
                f"- See full details? (e.g., 'show me recipe 1')\n"
                f"- Need more options? (e.g., 'Show me more vegetarian recipes')\n"
                f"- Have questions? Just ask!"
                + _MEDICAL_DISCLAIMER
            )
        else:
            footer = (
                "\n\n---\n"
                "No recipes found. Try rephrasing your request "
                "(e.g., 'low-carb dinner with fish')."
            )

        return ToolResult(
            output=output + footer,
            data=result,
            store_as="last_recommendations",
        )
