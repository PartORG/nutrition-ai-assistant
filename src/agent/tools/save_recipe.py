"""
agent.tools.save_recipe - Recipe selection and saving tool.

Reads typed Recipe objects from ctx.scratch (no markdown parsing).
Calls RecipeManagerService to persist the selection.
"""

from __future__ import annotations

from typing import List, Optional, Union

from pydantic import BaseModel, Field, field_validator

from application.context import SessionContext
from application.services.recipe_manager import RecipeManagerService
from application.dto import RecommendationResult
from agent.tools.base import BaseTool, ToolResult


class SaveRecipeInput(BaseModel):
    """Input schema for the save_recipe tool."""
    recipe_numbers: List[int] = Field(
        description=(
            "List of recipe numbers to save from the previous recommendations. "
            "Always use a JSON array, even for a single recipe — e.g. [1] or [1, 2, 3]. "
            "Numbers must match the numbered list shown to the user (1-based)."
        )
    )
    rating: Optional[int] = Field(
        default=None,
        description=(
            "Optional star rating 1-5. "
            "Omit this field entirely (or pass null) when the user did not provide a rating."
        ),
    )

    @field_validator("recipe_numbers", mode="before")
    @classmethod
    def coerce_to_int_list(cls, v: object) -> List[int]:
        """Accept int, str, or list — normalise everything to List[int].

        Handles LLM quirks like passing a bare int, a string 'recipe 1',
        or a list of strings ['recipe 1', 'recipe 2'].
        """
        def _parse_one(item: object) -> int:
            if isinstance(item, int):
                return item
            s = str(item).lower().strip()
            # Strip a leading 'recipe ' / 'recipe_' prefix the LLM sometimes adds
            for prefix in ("recipe ", "recipe_", "recipe"):
                if s.startswith(prefix):
                    s = s[len(prefix):].strip()
                    break
            return int(s)

        if isinstance(v, (int, str)):
            return [_parse_one(v)]
        return [_parse_one(item) for item in v]

    @field_validator("rating", mode="before")
    @classmethod
    def coerce_rating(cls, v: object) -> Optional[int]:
        """Treat nil/none/null strings as None so Pydantic doesn't reject them."""
        if v is None:
            return None
        if isinstance(v, str) and v.lower() in ("nil", "none", "null", ""):
            return None
        return v


class SaveRecipeTool(BaseTool):
    """Save one or more selected recipes to user's cooking history."""

    name = "save_recipe"
    description = (
        "Save one or more selected recipes to user's cooking history. "
        "Use when user chooses recipe(s) by number. "
        "Pass recipe_numbers as a JSON array, e.g. [1] or [1, 2, 3]."
    )

    def __init__(self, recipe_manager: RecipeManagerService):
        self._manager = recipe_manager

    def get_schema(self) -> type[BaseModel]:
        return SaveRecipeInput

    async def execute(
        self,
        ctx: SessionContext,
        recipe_numbers: Union[List[int], int] = 0,
        rating: Optional[int] = None,
        **kwargs,
    ) -> ToolResult:
        """Save the user's selected recipe(s)."""
        rec_result: Optional[RecommendationResult] = ctx.scratch.get("last_recommendations")

        if rec_result is None:
            return ToolResult(
                output="No recipes available. Please search for recipes first."
            )

        recipes = rec_result.safe_recipes

        # Normalise to list (execute may receive a bare int if called directly)
        numbers: List[int] = (
            [recipe_numbers] if isinstance(recipe_numbers, int) else list(recipe_numbers)
        )

        saved_lines: List[str] = []
        errors: List[str] = []

        for num in numbers:
            if num < 1 or num > len(recipes):
                errors.append(f"Recipe {num} — invalid number (choose 1–{len(recipes)})")
                continue
            recipe = recipes[num - 1]
            history_id = await self._manager.save_selection(ctx, recipe, rating)
            line = f"- **{recipe.name}** (#{num}, ID {history_id})"
            if recipe.prep_time:
                line += f" — prep {recipe.prep_time}"
            saved_lines.append(line)

        if not saved_lines and errors:
            return ToolResult(output="\n".join(errors))

        response_parts = ["Recipes saved successfully!\n"]
        response_parts.extend(saved_lines)
        if errors:
            response_parts.append("\nCould not save:")
            response_parts.extend(errors)
        if rating:
            response_parts.append(f"\nYour rating: {rating}/5 ⭐")
        response_parts.append("\nEnjoy cooking! Need more recipes? Just ask!")

        return ToolResult(output="\n".join(response_parts))
