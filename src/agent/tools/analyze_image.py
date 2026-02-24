"""
agent.tools.analyze_image - CNN ingredient detection tool.

Detects ingredients from a food photo and optionally chains into
the recommendation pipeline.
"""

from __future__ import annotations

import re
import logging

from pydantic import BaseModel, Field

from application.context import SessionContext
from application.services.image_analysis import ImageAnalysisService
from agent.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)

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
        description=(
            "The file path extracted from the [IMAGE:...] tag in the user message. "
            "Extract ONLY the path inside the brackets — e.g. from '[IMAGE:/tmp/photo.jpg]' "
            "pass '/tmp/photo.jpg'. Do NOT include the [IMAGE:] wrapper."
        )
    )


class AnalyzeImageTool(BaseTool):
    """Detect ingredients from a food photo and optionally find recipes."""

    name = "analyze_image"
    description = (
        "Detect ingredients from a food photo and suggest recipes. "
        "ONLY call this tool when the user's message contains a [IMAGE:...] tag or file path. "
        "Extract the path from [IMAGE:/path/to/file] and pass it as image_path. "
        "NEVER invent or guess an image path."
    )

    def __init__(self, image_service: ImageAnalysisService):
        self._service = image_service

    def get_schema(self) -> type[BaseModel]:
        return AnalyzeImageInput

    async def execute(
        self,
        ctx: SessionContext,
        image_path: str = "",
        **kwargs,
    ) -> ToolResult:
        """Detect ingredients and get recipe recommendations.

        Reads the original user message from ctx.scratch so that any text the
        user typed alongside the photo (e.g. "what can I make for dinner?") is
        forwarded to the recommendation pipeline as additional_query.
        """
        logger.info("AnalyzeImageTool.execute() called, raw image_path=%s", image_path)

        # Always prefer the path from the original user message.
        # The LLM often substitutes a placeholder (e.g. /tmp/photo.jpg) instead
        # of copying the actual uploaded path from the [IMAGE:...] tag.
        original = ctx.scratch.get("original_query", "")
        path_match = re.search(r"\[IMAGE:([^\]]+)\]", original, re.IGNORECASE)
        if path_match:
            image_path = path_match.group(1).strip()
            logger.info("AnalyzeImageTool: path from original_query: %s", image_path)
        else:
            # Fallback: strip [IMAGE:...] wrapper if the LLM passed it whole
            image_tag_match = re.search(r"\[IMAGE:([^\]]+)\]", image_path, re.IGNORECASE)
            if image_tag_match:
                image_path = image_tag_match.group(1).strip()
            logger.info("AnalyzeImageTool: resolved image_path=%s", image_path)

        # Extract the user's actual request, stripping the image path reference
        additional_query = _extract_user_text(original, image_path)
        find_recipes = True

        if find_recipes:
            result = await self._service.recommend_from_image(
                ctx, image_path, additional_query
            )

            detected = result.detected
            ingredients_str = ", ".join(detected.ingredients)
            source_label = detected.source or "unknown"
            if detected.source == "LLaVA":
                source_note = "detected via LLaVA (visual AI)"
            elif detected.source == "YOLO":
                source_note = "detected via YOLO"
            else:
                source_note = f"detected via {source_label}"

            confidence_parts = [
                f"{ing}: {detected.confidence_scores.get(ing, 0):.0%}"
                for ing in detected.ingredients[:5]
            ]
            conf_str = ", ".join(confidence_parts) if confidence_parts else "n/a"
            header = (
                f"Detected {len(detected.ingredients)} ingredient(s) ({source_note}):\n"
                f"{ingredients_str}\n"
                f"Confidence: {conf_str}\n\n"
            )

            rec = result.recommendation
            safe_recipes = rec.safe_recipes if rec else []

            if safe_recipes:
                recipes_md = rec.safety_result.safe_recipes_markdown
                footer = (
                    f"\n\n---\n"
                    f"Found {len(safe_recipes)} recipes using your ingredients!\n"
                    f"Want to cook one? Tell me the recipe number."
                )
                return ToolResult(
                    output=header + recipes_md + footer,
                    data=rec,
                    store_as="last_recommendations",
                )
            elif rec is not None:
                # Pipeline ran but safety filter rejected everything — explain why
                summary = rec.summary
                logger.warning(
                    "All recipes rejected by safety filter for user %d. Summary: %s",
                    ctx.user_id, summary,
                )
                rejected = rec.safety_result.filtered_out
                reasons = "; ".join(
                    f"{r.recipe_name}: {r.issues[0].description}"
                    for r in rejected[:3]
                    if r.issues
                )
                output = (
                    header
                    + "Detected your ingredients but all suggested recipes were filtered "
                    "out by your dietary safety constraints.\n\n"
                )
                if reasons:
                    output += f"Reasons: {reasons}\n\n"
                output += (
                    "Try asking me directly — e.g. "
                    "\"suggest a light dinner with tomatoes and eggs\"."
                )
                return ToolResult(
                    output=output,
                    data=detected,
                    store_as="detected_ingredients",
                )
            else:
                return ToolResult(
                    output=f"{header}Could not find matching recipes. Try adding more details.",
                    data=detected,
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
