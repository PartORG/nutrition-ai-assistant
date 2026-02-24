"""
agent.tools.show_recipe - Display full recipe details from previous search.

Reads typed Recipe objects from ctx.scratch (stored by search_recipes /
analyze_image tools) and returns a detailed markdown view.
No DB writes — display only.
"""

from __future__ import annotations

from typing import Union

from pydantic import BaseModel, Field, field_validator

from application.context import SessionContext
from application.dto import RecommendationResult
from domain.models import Recipe, NutritionValues
from agent.tools.base import BaseTool, ToolResult


class ShowRecipeInput(BaseModel):
    """Input schema for the show_recipe tool."""

    recipe_number: Union[int, list[int]] = Field(
        description=(
            "The 1-based number(s) of the recipe(s) to display, matching the "
            "numbered list already shown to the user (e.g. 1, 2, or [1, 2, 3])."
        )
    )

    @field_validator("recipe_number", mode="before")
    @classmethod
    def coerce_to_int_or_list(cls, v):
        if isinstance(v, list):
            coerced = []
            for item in v:
                try:
                    coerced.append(int(item))
                except (TypeError, ValueError):
                    pass
            return coerced if coerced else 1
        try:
            return int(v)
        except (TypeError, ValueError):
            return 1


class ShowRecipeTool(BaseTool):
    """Show full details of a recipe from the previous search results."""

    name = "show_recipe"
    description = (
        "Display the full ingredients, cooking instructions, and nutrition info "
        "for a specific recipe from the previous search results. "
        "Use when the user says 'show me recipe X', 'see recipe X', "
        "'details of recipe X', or 'what's in recipe X'. "
        "Do NOT use this when the user wants to SAVE or COOK — use save_recipe for that."
    )

    def get_schema(self) -> type[BaseModel]:
        return ShowRecipeInput

    async def execute(
        self,
        ctx: SessionContext,
        recipe_number: Union[int, list[int]] = 1,
        **kwargs,
    ) -> ToolResult:
        """Return a detailed markdown view of the requested recipe(s)."""
        rec_result: RecommendationResult | None = ctx.scratch.get(
            "last_recommendations"
        )
        if rec_result is None:
            return ToolResult(
                output=(
                    "No recipes available to show. "
                    "Please search for recipes first."
                )
            )

        recipes = rec_result.safe_recipes
        if not recipes:
            return ToolResult(output="No recipes were found in the last search.")

        # Normalise to a list of numbers
        numbers: list[int] = (
            recipe_number if isinstance(recipe_number, list) else [recipe_number]
        )

        parts: list[str] = []
        for num in numbers:
            if num < 1 or num > len(recipes):
                parts.append(
                    f"Recipe {num} doesn't exist. "
                    f"Please choose a number between 1 and {len(recipes)}."
                )
            else:
                parts.append(_format_recipe_detail(recipes[num - 1], num))

        return ToolResult(output="\n\n---\n\n".join(parts))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_recipe_detail(recipe: Recipe, number: int) -> str:
    """Render a Recipe as detailed markdown."""
    lines: list[str] = [f"## Recipe {number}: {recipe.name}"]

    if recipe.why_recommended:
        lines.append(f"*{recipe.why_recommended}*")

    # Meta row
    meta: list[str] = []
    if recipe.servings:
        meta.append(f"Servings: {recipe.servings}")
    if recipe.prep_time:
        meta.append(f"Prep time: {recipe.prep_time}")
    if meta:
        lines.append(" · ".join(meta))

    # Ingredients
    if recipe.ingredients:
        lines.append("\n### Ingredients")
        for ing in recipe.ingredients:
            lines.append(f"- {ing}")

    # Instructions
    if recipe.cook_instructions:
        lines.append("\n### Instructions")
        lines.append(recipe.cook_instructions)

    # Nutrition
    n: NutritionValues = recipe.nutrition
    nutrition_parts: list[str] = []
    if n.calories is not None:
        nutrition_parts.append(f"**{n.calories:.0f} kcal**")
    if n.protein_g is not None:
        nutrition_parts.append(f"Protein: {n.protein_g:.1f} g")
    if n.carbs_g is not None:
        nutrition_parts.append(f"Carbs: {n.carbs_g:.1f} g")
    if n.fat_g is not None:
        nutrition_parts.append(f"Fat: {n.fat_g:.1f} g")
    if n.fiber_g is not None:
        nutrition_parts.append(f"Fiber: {n.fiber_g:.1f} g")
    if n.sodium_mg is not None:
        nutrition_parts.append(f"Sodium: {n.sodium_mg:.0f} mg")
    if n.sugar_g is not None:
        nutrition_parts.append(f"Sugar: {n.sugar_g:.1f} g")

    if nutrition_parts:
        lines.append("\n### Nutrition (per serving)")
        lines.append(" | ".join(nutrition_parts))

    lines.append(
        f"\n---\nLike this recipe? Say **'save recipe {number}'** to add it to your history."
    )
    return "\n".join(lines)
