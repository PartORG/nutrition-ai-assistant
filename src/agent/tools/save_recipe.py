"""
agent.tools.save_recipe - Recipe selection and saving tool.

Reads typed Recipe objects from ctx.scratch (no markdown parsing).
Calls RecipeManagerService to persist the selection.

Supports selection by number (recipe_numbers=[2]) AND by name
(recipe_name="salmon") with three-strategy fuzzy matching.
"""

from __future__ import annotations

from typing import List, Optional, Union

from pydantic import BaseModel, Field, field_validator

from domain.models import Recipe
from application.context import SessionContext
from application.services.recipe_manager import RecipeManagerService
from application.dto import RecommendationResult
from agent.tools.base import BaseTool, ToolResult


class SaveRecipeInput(BaseModel):
    """Input schema for the save_recipe tool."""

    recipe_numbers: Optional[List[int]] = Field(
        default=None,
        description=(
            "List of 1-based recipe numbers to save, e.g. [2] or [1, 3]. "
            "Use this when the user says 'recipe 2' or 'the second one'. "
            "Omit (or pass null) when using recipe_name instead."
        ),
    )
    recipe_name: Optional[str] = Field(
        default=None,
        description=(
            "Partial or full recipe name to save, e.g. 'salmon' or 'Grilled Salmon'. "
            "Use this when the user refers to a dish by name rather than number. "
            "Fuzzy matching finds the closest recipe automatically."
        ),
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
    def coerce_to_int_list(cls, v: object) -> Optional[List[int]]:
        """Accept int, str, list, or None — normalise to List[int] or None.

        Handles LLM quirks like a bare int, 'recipe 1', or ['recipe 1', 'recipe 2'].
        Returns None for empty / null so name-based fallback activates.
        """
        if v is None:
            return None

        def _parse_one(item: object) -> int:
            if isinstance(item, int):
                return item
            s = str(item).lower().strip()
            for prefix in ("recipe ", "recipe_", "recipe"):
                if s.startswith(prefix):
                    s = s[len(prefix):].strip()
                    break
            return int(s)

        if isinstance(v, (int, str)):
            try:
                return [_parse_one(v)]
            except (ValueError, TypeError):
                return None
        if isinstance(v, list):
            if not v:
                return None
            try:
                return [_parse_one(item) for item in v]
            except (ValueError, TypeError):
                return None
        return None

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
        "Use when user chooses a recipe by NUMBER (recipe_numbers=[2]) "
        "OR by NAME (recipe_name='salmon' — fuzzy matching finds it). "
        "Prefer recipe_numbers when the user says 'recipe 2'. "
        "Use recipe_name when the user mentions the dish name."
    )

    def __init__(self, recipe_manager: RecipeManagerService):
        self._manager = recipe_manager

    def get_schema(self) -> type[BaseModel]:
        return SaveRecipeInput

    async def execute(
        self,
        ctx: SessionContext,
        recipe_numbers: Union[List[int], None] = None,
        recipe_name: Optional[str] = None,
        rating: Optional[int] = None,
        **kwargs,
    ) -> ToolResult:
        """Save the user's selected recipe(s), resolving by number or name."""
        rec_result: Optional[RecommendationResult] = ctx.scratch.get("last_recommendations")

        if rec_result is not None:
            recipes = rec_result.safe_recipes
        else:
            # Fallback: use the recipe list cached from a previous session
            # (restored from DB when the WebSocket reconnects).
            recipes = ctx.scratch.get("_cached_safe_recipes", [])

        if not recipes:
            return ToolResult(
                output=(
                    "No recipes available. Please search for recipes first "
                    "(e.g. 'show me low-carb dinner ideas')."
                )
            )

        numbers: List[int] = list(recipe_numbers) if recipe_numbers else []

        # Name-based fallback: resolve fuzzy name → recipe number
        if not numbers and recipe_name:
            found = _find_by_name(recipe_name, recipes)
            if found is not None:
                numbers = [found]
            else:
                available = "\n".join(
                    f"  {i + 1}. {r.name}" for i, r in enumerate(recipes)
                )
                return ToolResult(
                    output=(
                        f"No recipe matching '{recipe_name}' found.\n\n"
                        f"Available recipes:\n{available}\n\n"
                        "Please use the recipe number (e.g. 'I'll cook recipe 2')."
                    )
                )

        if not numbers:
            return ToolResult(
                output=(
                    "Please specify which recipe you want:\n"
                    "- By number: 'I'll cook recipe 2'\n"
                    "- By name: 'save the salmon recipe'"
                )
            )

        # Turn-scoped dedup: prevent the LLM from calling save_recipe twice
        # for the same recipe in one turn (e.g. first without rating, then with).
        # Key includes request_id so the guard resets automatically each turn.
        turn_key = f"_saved_{ctx.request_id}"
        saved_this_turn: dict = ctx.scratch.setdefault(turn_key, {})

        saved_lines: List[str] = []
        errors: List[str] = []

        for num in numbers:
            if num < 1 or num > len(recipes):
                errors.append(f"Recipe {num} — invalid number (choose 1–{len(recipes)})")
                continue
            recipe = recipes[num - 1]

            if recipe.name in saved_this_turn:
                # Duplicate call within this turn — skip the DB insert to avoid
                # a duplicate row.  Reuse the history_id from the first call.
                history_id = saved_this_turn[recipe.name]
                logger.warning(
                    "Duplicate save_recipe call for '%s' in turn %s — skipping DB insert",
                    recipe.name, ctx.request_id,
                )
            else:
                history_id = await self._manager.save_selection(ctx, recipe, rating)
                saved_this_turn[recipe.name] = history_id

            saved_lines.append(_format_saved_recipe(recipe, num, history_id))

        if not saved_lines and errors:
            return ToolResult(output="\n".join(errors))

        parts = ["✅ **Recipe saved successfully!**\n"]
        parts.extend(saved_lines)
        if errors:
            parts.append("\nCould not save:")
            parts.extend(errors)
        if rating:
            parts.append(f"\nYour rating: {rating}/5 ⭐")
        parts.append(
            "\n**What's next?**\n"
            "- Ready to cook? Your recipe is saved in your history!\n"
            "- Want more ideas? Just ask for new recipes.\n"
            "- Have cooking questions? I'm here to help! 🍽️"
        )

        return ToolResult(output="\n".join(parts))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_by_name(name: str, recipes: List[Recipe]) -> Optional[int]:
    """Find a recipe by fuzzy name matching. Returns 1-based index or None.

    Three strategies (in priority order):
    1. Exact match (case-insensitive)
    2. Partial match (one contains the other)
    3. Word-overlap match (any shared word)
    """
    needle = name.lower().strip()

    # Strategy 1: exact
    for i, r in enumerate(recipes):
        if needle == r.name.lower():
            return i + 1

    # Strategy 2: partial
    for i, r in enumerate(recipes):
        rname = r.name.lower()
        if needle in rname or rname in needle:
            return i + 1

    # Strategy 3: any shared word
    needle_words = set(needle.split())
    for i, r in enumerate(recipes):
        if needle_words & set(r.name.lower().split()):
            return i + 1

    return None


def _format_saved_recipe(recipe: Recipe, number: int, history_id: int) -> str:
    """Build a rich one-line confirmation for a saved recipe."""
    details: List[str] = []
    if recipe.servings:
        details.append(f"serves {recipe.servings}")
    if recipe.prep_time:
        details.append(f"prep {recipe.prep_time}")
    if recipe.nutrition.calories is not None:
        details.append(f"{recipe.nutrition.calories:.0f} kcal")

    detail_str = f" ({', '.join(details)})" if details else ""
    return f"- **{recipe.name}**{detail_str}"
