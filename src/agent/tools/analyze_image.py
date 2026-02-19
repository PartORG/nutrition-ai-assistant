"""
agent.tools.analyze_image - CNN ingredient detection tool.

Detects ingredients from a food photo and optionally chains into
the recommendation pipeline.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from application.context import SessionContext
from application.services.image_analysis import ImageAnalysisService
from agent.tools.base import BaseTool, ToolResult


def _extract_user_text(original_query: str, image_path: str) -> str:
    """Strip image references from the original query to get the user's request.

    Handles both the legacy format ("Please analyze the food in this image: /path")
    and the new combined format ("user text [IMAGE:/path]").
    Returns an empty string when nothing meaningful remains.
    """
    text = original_query

    # Remove [IMAGE:...] marker (new Flutter format)
    text = re.sub(r"\[IMAGE:[^\]]*\]", "", text, flags=re.IGNORECASE)

    # Remove the specific server path (in case LLM rewrote the message)
    if image_path:
        text = text.replace(image_path, "")

    # Remove legacy hardcoded phrases the Flutter app used to send
    text = re.sub(
        r"please analyze the food in this image:?",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"analyze the food in this image:?",
        "",
        text,
        flags=re.IGNORECASE,
    )

    return text.strip()


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
        """Detect ingredients and optionally get recipe recommendations.

        Reads the original user message from ctx.scratch so that any text the
        user typed alongside the photo (e.g. "what can I make for dinner?") is
        forwarded to the recommendation pipeline as additional_query.
        """
        # Extract the user's actual request, stripping the image path reference
        original = ctx.scratch.get("original_query", "")
        additional_query = _extract_user_text(original, image_path)

        if find_recipes:
            result = await self._service.recommend_from_image(
                ctx, image_path, additional_query
            )

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
