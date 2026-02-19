"""
agent.tools.analyze_image - CNN ingredient detection tool.

Detects ingredients from a food photo and optionally chains into
the recommendation pipeline.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from application.context import SessionContext
from application.services.image_analysis import ImageAnalysisService
from agent.tools.base import BaseTool, ToolResult


class AnalyzeImageInput(BaseModel):
    """Input schema for the analyze_image tool."""
    image_path: str = Field(
        description="Path to the food photo to analyze"
    )
    find_recipes: bool = Field(
        default=True,
        description="If True, also search for recipes using detected ingredients",
    )


class AnalyzeImageTool(BaseTool):
    """Detect ingredients from a food photo and optionally find recipes."""

    name = "analyze_image"
    description = (
        "Detect ingredients from a food photo and optionally suggest recipes. "
        "ONLY call this tool when the user's message contains an actual file path "
        "(e.g. /home/user/photo.jpg or C:\\photos\\meal.png). "
        "NEVER invent or guess an image path — if no path is present, use search_recipes instead."
    )

    def __init__(self, image_service: ImageAnalysisService):
        self._service = image_service

    def get_schema(self) -> type[BaseModel]:
        return AnalyzeImageInput

    async def execute(
        self,
        ctx: SessionContext,
        image_path: str = "",
        find_recipes: bool = True,
        **kwargs,
    ) -> ToolResult:
        """Detect ingredients and optionally get recipe recommendations."""
        if find_recipes:
            result = await self._service.recommend_from_image(ctx, image_path)

            if result.recommendation:
                safe_recipes = result.recommendation.safe_recipes
                ingredients_str = ", ".join(result.detected.ingredients)
                confidence_parts = [
                    f"{ing}: {result.detected.confidence_scores.get(ing, 0):.0%}"
                    for ing in result.detected.ingredients[:5]
                ]
                header = (
                    f"Detected ingredients: {ingredients_str}\n"
                    f"Confidence: {', '.join(confidence_parts)}\n\n"
                )
                recipes_md = result.recommendation.safety_result.safe_recipes_markdown
                footer = (
                    f"\n\n---\n"
                    f"Found {len(safe_recipes)} recipes using your ingredients!\n"
                    f"Want to cook one? Tell me the recipe number."
                )
                return ToolResult(
                    output=header + recipes_md + footer,
                    data=result.recommendation,
                    store_as="last_recommendations",
                )
            else:
                return ToolResult(
                    output=f"Detected ingredients: {', '.join(result.detected.ingredients)}\n"
                           f"Could not find matching recipes. Try adding more details.",
                    data=result.detected,
                    store_as="detected_ingredients",
                )
        else:
            detected = await self._service.detect_ingredients(ctx, image_path)
            ingredients_str = ", ".join(detected.ingredients)
            return ToolResult(
                output=f"Detected ingredients: {ingredients_str}",
                data=detected,
                store_as="detected_ingredients",
            )
