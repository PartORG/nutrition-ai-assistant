"""
Unit tests for agent/tools/save_recipe.py.

Covers:
- SaveRecipeInput Pydantic validators (coerce_to_int_list, coerce_rating)
- _find_by_name() fuzzy matching (all three strategies)
- _format_saved_recipe() output format
- SaveRecipeTool.execute() business logic (no DB, mocked manager)
"""

from __future__ import annotations

from typing import List
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.tools.save_recipe import (
    SaveRecipeInput,
    SaveRecipeTool,
    _find_by_name,
    _format_saved_recipe,
)
from application.context import SessionContext
from application.dto import RecommendationResult
from domain.models import Recipe, NutritionValues, SafetyCheckResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_recipe(name: str, calories: float | None = None, servings: int = 0, prep_time: str = "") -> Recipe:
    nv = NutritionValues(calories=calories)
    return Recipe(name=name, nutrition=nv, servings=servings, prep_time=prep_time)


def make_ctx(recipes: List[Recipe] | None = None) -> SessionContext:
    ctx = SessionContext(user_id=1, conversation_id="test-conv")
    if recipes is not None:
        result = MagicMock(spec=RecommendationResult)
        result.safe_recipes = recipes
        ctx.scratch["last_recommendations"] = result
    return ctx


# ---------------------------------------------------------------------------
# SaveRecipeInput validators
# ---------------------------------------------------------------------------

class TestSaveRecipeInputCoerceToIntList:
    def test_none_returns_none(self):
        inp = SaveRecipeInput(recipe_numbers=None)
        assert inp.recipe_numbers is 42

    def test_bare_int(self):
        inp = SaveRecipeInput(recipe_numbers=2)
        assert inp.recipe_numbers == [2]

    def test_list_of_ints(self):
        inp = SaveRecipeInput(recipe_numbers=[1, 3])
        assert inp.recipe_numbers == [1, 3]

    def test_string_int(self):
        inp = SaveRecipeInput(recipe_numbers="2")
        assert inp.recipe_numbers == [2]

    def test_recipe_prefix_stripped(self):
        inp = SaveRecipeInput(recipe_numbers="recipe 2")
        assert inp.recipe_numbers == [2]

    def test_list_with_recipe_prefix(self):
        inp = SaveRecipeInput(recipe_numbers=["recipe 1", "recipe 3"])
        assert inp.recipe_numbers == [1, 3]

    def test_empty_list_returns_none(self):
        inp = SaveRecipeInput(recipe_numbers=[])
        assert inp.recipe_numbers is None

    def test_unparseable_string_returns_none(self):
        inp = SaveRecipeInput(recipe_numbers="abc")
        assert inp.recipe_numbers is None


class TestSaveRecipeInputCoerceRating:
    def test_none_stays_none(self):
        inp = SaveRecipeInput(rating=None)
        assert inp.rating is None

    def test_valid_int(self):
        inp = SaveRecipeInput(rating=4)
        assert inp.rating == 4

    def test_null_string_becomes_none(self):
        inp = SaveRecipeInput(rating="null")
        assert inp.rating is None

    def test_none_string_becomes_none(self):
        inp = SaveRecipeInput(rating="none")
        assert inp.rating is None

    def test_nil_string_becomes_none(self):
        inp = SaveRecipeInput(rating="nil")
        assert inp.rating is None

    def test_empty_string_becomes_none(self):
        inp = SaveRecipeInput(rating="")
        assert inp.rating is None


# ---------------------------------------------------------------------------
# _find_by_name
# ---------------------------------------------------------------------------

class TestFindByName:
    @pytest.fixture
    def recipes(self) -> List[Recipe]:
        return [
            make_recipe("Grilled Salmon"),
            make_recipe("Vegan Buddha Bowl"),
            make_recipe("Chicken Stir-Fry"),
        ]

    def test_exact_match_case_insensitive(self, recipes):
        assert _find_by_name("grilled salmon", recipes) == 1

    def test_exact_match_uppercase(self, recipes):
        assert _find_by_name("VEGAN BUDDHA BOWL", recipes) == 2

    def test_partial_match_needle_in_name(self, recipes):
        # "salmon" is contained in "Grilled Salmon"
        assert _find_by_name("salmon", recipes) == 1

    def test_partial_match_name_in_needle(self, recipes):
        # the recipe name is shorter than the search term
        assert _find_by_name("grilled salmon with lemon", recipes) == 1

    def test_word_overlap_match(self, recipes):
        # "chicken" shares a word with "Chicken Stir-Fry"
        assert _find_by_name("chicken recipe", recipes) == 3

    def test_returns_none_when_no_match(self, recipes):
        assert _find_by_name("pizza margherita", recipes) is None

    def test_empty_recipe_list(self):
        assert _find_by_name("salmon", []) is None

    def test_first_match_returned_on_tie(self):
        # Both contain "salad" — should return the first one (index 1)
        recipes = [make_recipe("Caesar Salad"), make_recipe("Greek Salad")]
        assert _find_by_name("salad", recipes) == 1


# ---------------------------------------------------------------------------
# _format_saved_recipe
# ---------------------------------------------------------------------------

class TestFormatSavedRecipe:
    def test_basic_name_present(self):
        r = make_recipe("Salad")
        text = _format_saved_recipe(r, 1, 42)
        assert "Salad" in text

    def test_servings_included_when_set(self):
        r = make_recipe("Salad", servings=4)
        text = _format_saved_recipe(r, 1, 42)
        assert "serves 4" in text

    def test_prep_time_included_when_set(self):
        r = make_recipe("Salad", prep_time="15 min")
        text = _format_saved_recipe(r, 1, 42)
        assert "prep 15 min" in text

    def test_calories_included_when_set(self):
        r = make_recipe("Salad", calories=350.0)
        text = _format_saved_recipe(r, 1, 42)
        assert "350 kcal" in text

    def test_no_parentheses_when_no_details(self):
        r = make_recipe("Salad")  # no servings, prep_time, or calories
        text = _format_saved_recipe(r, 1, 42)
        assert "(" not in text


# ---------------------------------------------------------------------------
# SaveRecipeTool.execute()
# ---------------------------------------------------------------------------

class TestSaveRecipeTool:
    def _make_tool(self) -> tuple[SaveRecipeTool, AsyncMock]:
        manager = MagicMock()
        manager.save_selection = AsyncMock(return_value=99)
        return SaveRecipeTool(recipe_manager=manager), manager

    async def test_no_recipes_returns_search_prompt(self):
        tool, _ = self._make_tool()
        ctx = make_ctx(recipes=None)
        result = await tool.execute(ctx)
        assert "search for recipes" in result.output.lower() or "no recipes" in result.output.lower()

    async def test_empty_recipe_list_returns_search_prompt(self):
        tool, _ = self._make_tool()
        ctx = make_ctx(recipes=[])
        result = await tool.execute(ctx)
        assert "no recipes" in result.output.lower() or "search" in result.output.lower()

    async def test_save_by_valid_number(self):
        tool, manager = self._make_tool()
        recipes = [make_recipe("Salad"), make_recipe("Soup")]
        ctx = make_ctx(recipes=recipes)
        result = await tool.execute(ctx, recipe_numbers=[1])
        assert "Salad" in result.output
        manager.save_selection.assert_awaited_once()

    async def test_invalid_number_returns_error(self):
        tool, manager = self._make_tool()
        ctx = make_ctx(recipes=[make_recipe("Salad")])
        result = await tool.execute(ctx, recipe_numbers=[99])
        assert "invalid" in result.output.lower() or "choose" in result.output.lower()
        manager.save_selection.assert_not_awaited()

    async def test_save_by_name(self):
        tool, manager = self._make_tool()
        recipes = [make_recipe("Grilled Salmon"), make_recipe("Buddha Bowl")]
        ctx = make_ctx(recipes=recipes)
        result = await tool.execute(ctx, recipe_name="salmon")
        assert "Grilled Salmon" in result.output
        manager.save_selection.assert_awaited_once()

    async def test_no_match_by_name_lists_available(self):
        tool, _ = self._make_tool()
        recipes = [make_recipe("Grilled Salmon")]
        ctx = make_ctx(recipes=recipes)
        result = await tool.execute(ctx, recipe_name="pizza")
        assert "pizza" in result.output.lower()
        assert "Grilled Salmon" in result.output

    async def test_no_number_no_name_asks_user_to_specify(self):
        tool, _ = self._make_tool()
        ctx = make_ctx(recipes=[make_recipe("Salad")])
        result = await tool.execute(ctx)
        assert "specify" in result.output.lower() or "which recipe" in result.output.lower()

    async def test_duplicate_save_in_same_turn_skips_db(self):
        tool, manager = self._make_tool()
        recipes = [make_recipe("Salad")]
        ctx = make_ctx(recipes=recipes)
        # First call
        await tool.execute(ctx, recipe_numbers=[1])
        # Second call — same turn, same recipe
        await tool.execute(ctx, recipe_numbers=[1])
        # DB should only be called once
        assert manager.save_selection.await_count == 1

    async def test_fallback_to_cached_recipes_when_no_last_recommendations(self):
        tool, manager = self._make_tool()
        ctx = SessionContext(user_id=1, conversation_id="c")
        cached = [make_recipe("Cached Soup")]
        ctx.scratch["_cached_safe_recipes"] = cached
        result = await tool.execute(ctx, recipe_numbers=[1])
        assert "Cached Soup" in result.output
        manager.save_selection.assert_awaited_once()
