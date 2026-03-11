"""
Unit tests for agent/tools/show_recipe.py.

Covers:
- ShowRecipeInput.coerce_to_int_or_list validator
- _format_recipe_detail() output content
- ShowRecipeTool.execute() routing (no recipes, valid number, invalid number, multiple numbers)
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agent.tools.show_recipe import ShowRecipeTool, ShowRecipeInput, _format_recipe_detail
from application.context import SessionContext
from application.dto import RecommendationResult
from domain.models import Recipe, NutritionValues, SafetyCheckResult, RecipeSafetyResult, SafetyVerdict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_recipe(
    name: str,
    ingredients=None,
    cook_instructions: str = "",
    prep_time: str = "",
    servings: int = 0,
    why_recommended: str = "",
    calories: float | None = None,
    protein_g: float | None = None,
) -> Recipe:
    nv = NutritionValues(calories=calories, protein_g=protein_g)
    return Recipe(
        name=name,
        ingredients=ingredients or [],
        cook_instructions=cook_instructions,
        prep_time=prep_time,
        servings=servings,
        why_recommended=why_recommended,
        nutrition=nv,
    )


def make_ctx(recipes: list[Recipe] | None = None) -> SessionContext:
    ctx = SessionContext(user_id=1, conversation_id="test-conv")
    if recipes is not None:
        verdicts = [
            RecipeSafetyResult(recipe_name=r.name, verdict=SafetyVerdict.SAFE, recipe=r)
            for r in recipes
        ]
        safety = SafetyCheckResult(recipe_verdicts=verdicts, safe_recipes_markdown="")
        result = MagicMock(spec=RecommendationResult)
        result.safe_recipes = recipes
        ctx.scratch["last_recommendations"] = result
    return ctx


# ---------------------------------------------------------------------------
# ShowRecipeInput validator
# ---------------------------------------------------------------------------

class TestShowRecipeInputValidator:
    def test_int_passthrough(self):
        inp = ShowRecipeInput(recipe_number=2)
        assert inp.recipe_number == 2

    def test_string_int_coerced(self):
        inp = ShowRecipeInput(recipe_number="3")
        assert inp.recipe_number == 3

    def test_list_of_ints(self):
        inp = ShowRecipeInput(recipe_number=[1, 2])
        assert inp.recipe_number == [1, 2]

    def test_list_with_string_ints(self):
        inp = ShowRecipeInput(recipe_number=["1", "3"])
        assert inp.recipe_number == [1, 3]

    def test_unparseable_single_falls_back_to_1(self):
        inp = ShowRecipeInput(recipe_number="abc")
        assert inp.recipe_number == 1

    def test_empty_list_falls_back_to_1(self):
        inp = ShowRecipeInput(recipe_number=[])
        assert inp.recipe_number == 1

    def test_list_with_unparseable_items_skips_them(self):
        inp = ShowRecipeInput(recipe_number=["1", "bad", "3"])
        assert inp.recipe_number == [1, 3]


# ---------------------------------------------------------------------------
# _format_recipe_detail
# ---------------------------------------------------------------------------

class TestFormatRecipeDetail:
    def test_header_contains_number_and_name(self):
        r = make_recipe("Caesar Salad")
        text = _format_recipe_detail(r, 1)
        assert "## Recipe 1: Caesar Salad" in text

    def test_why_recommended_shown(self):
        r = make_recipe("Salad", why_recommended="High in fiber")
        text = _format_recipe_detail(r, 1)
        assert "High in fiber" in text

    def test_servings_shown(self):
        r = make_recipe("Soup", servings=4)
        text = _format_recipe_detail(r, 2)
        assert "Servings: 4" in text

    def test_prep_time_shown(self):
        r = make_recipe("Soup", prep_time="20 min")
        text = _format_recipe_detail(r, 1)
        assert "Prep time: 20 min" in text

    def test_ingredients_listed(self):
        r = make_recipe("Salad", ingredients=["lettuce", "tomato", "olive oil"])
        text = _format_recipe_detail(r, 1)
        assert "- lettuce" in text
        assert "- tomato" in text
        assert "- olive oil" in text

    def test_instructions_shown(self):
        r = make_recipe("Salad", cook_instructions="Mix everything together.")
        text = _format_recipe_detail(r, 1)
        assert "Mix everything together." in text

    def test_calories_shown(self):
        r = make_recipe("Salad", calories=320.0)
        text = _format_recipe_detail(r, 1)
        assert "320 kcal" in text

    def test_protein_shown(self):
        r = make_recipe("Salad", protein_g=28.5)
        text = _format_recipe_detail(r, 1)
        assert "28.5 g" in text

    def test_no_ingredients_section_when_empty(self):
        r = make_recipe("Mystery Dish")
        text = _format_recipe_detail(r, 1)
        assert "### Ingredients" not in text

    def test_no_nutrition_section_when_all_none(self):
        r = make_recipe("Empty Dish")
        text = _format_recipe_detail(r, 1)
        assert "### Nutrition" not in text

    def test_save_hint_at_end(self):
        r = make_recipe("Salad")
        text = _format_recipe_detail(r, 2)
        assert "save recipe 2" in text


# ---------------------------------------------------------------------------
# ShowRecipeTool.execute()
# ---------------------------------------------------------------------------

class TestShowRecipeTool:
    @pytest.fixture
    def tool(self):
        return ShowRecipeTool()

    async def test_no_recommendations_returns_prompt(self, tool):
        ctx = SessionContext(user_id=1, conversation_id="c")
        result = await tool.execute(ctx)
        assert "search for recipes" in result.output.lower() or "no recipes" in result.output.lower()

    async def test_valid_number_returns_detail(self, tool):
        ctx = make_ctx(recipes=[make_recipe("Salmon", calories=350.0)])
        result = await tool.execute(ctx, recipe_number=1)
        assert "Salmon" in result.output
        assert "350 kcal" in result.output

    async def test_invalid_number_returns_error_message(self, tool):
        ctx = make_ctx(recipes=[make_recipe("Salad")])
        result = await tool.execute(ctx, recipe_number=99)
        assert "99" in result.output
        assert "doesn't exist" in result.output or "choose" in result.output.lower()

    async def test_multiple_numbers_all_rendered(self, tool):
        recipes = [make_recipe("Salad"), make_recipe("Soup"), make_recipe("Stew")]
        ctx = make_ctx(recipes=recipes)
        result = await tool.execute(ctx, recipe_number=[1, 3])
        assert "Salad" in result.output
        assert "Stew" in result.output
        assert "Soup" not in result.output

    async def test_separator_between_multiple_recipes(self, tool):
        ctx = make_ctx(recipes=[make_recipe("A"), make_recipe("B")])
        result = await tool.execute(ctx, recipe_number=[1, 2])
        assert "---" in result.output

    async def test_empty_recipe_list_returns_not_found(self, tool):
        ctx = make_ctx(recipes=[])
        # Empty list: safe_recipes returns [] but scratch has a result object
        rec_result = MagicMock(spec=RecommendationResult)
        rec_result.safe_recipes = []
        ctx.scratch["last_recommendations"] = rec_result
        result = await tool.execute(ctx, recipe_number=1)
        assert "no recipes" in result.output.lower()
