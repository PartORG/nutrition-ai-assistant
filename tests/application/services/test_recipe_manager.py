"""
Unit tests for application/services/recipe_manager.py — RecipeManagerService.

Uses AsyncMock for RecipeRepository and NutritionRepository.
No database is touched.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, call

import pytest

from application.context import SessionContext
from application.services.recipe_manager import RecipeManagerService
from domain.entities import RecipeHistory, NutritionHistory
from domain.models import Recipe, NutritionValues


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_service(recipe_repo=None, nutrition_repo=None):
    if recipe_repo is None:
        recipe_repo = AsyncMock()
        recipe_repo.save = AsyncMock(return_value=42)
        recipe_repo.get_by_user = AsyncMock(return_value=[])
    if nutrition_repo is None:
        nutrition_repo = AsyncMock()
        nutrition_repo.save = AsyncMock(return_value=99)
    return RecipeManagerService(recipe_repo=recipe_repo, nutrition_repo=nutrition_repo)


def make_ctx(user_id: int = 1) -> SessionContext:
    return SessionContext(user_id=user_id, conversation_id="test-conv")


def make_recipe(
    name: str = "Salad",
    ingredients=None,
    calories: float | None = 350.0,
    protein_g: float | None = 25.0,
    servings: int = 2,
    prep_time: str = "15 min",
    cook_instructions: str = "Mix and serve.",
) -> Recipe:
    nv = NutritionValues(
        calories=calories,
        protein_g=protein_g,
        carbs_g=40.0,
        fat_g=10.0,
        fiber_g=5.0,
        sugar_g=8.0,
        sodium_mg=300.0,
    )
    return Recipe(
        name=name,
        ingredients=ingredients or ["lettuce", "tomato"],
        nutrition=nv,
        servings=servings,
        prep_time=prep_time,
        cook_instructions=cook_instructions,
    )


# ---------------------------------------------------------------------------
# save_selection()
# ---------------------------------------------------------------------------

class TestSaveSelection:
    async def test_returns_history_id(self):
        recipe_repo = AsyncMock()
        recipe_repo.save = AsyncMock(return_value=55)
        svc = make_service(recipe_repo=recipe_repo)

        history_id = await svc.save_selection(make_ctx(), make_recipe())
        assert history_id == 55

    async def test_recipe_repo_save_called_once(self):
        recipe_repo = AsyncMock()
        recipe_repo.save = AsyncMock(return_value=1)
        svc = make_service(recipe_repo=recipe_repo)

        await svc.save_selection(make_ctx(), make_recipe())
        recipe_repo.save.assert_awaited_once()

    async def test_nutrition_repo_save_called_when_calories_present(self):
        nutrition_repo = AsyncMock()
        nutrition_repo.save = AsyncMock(return_value=10)
        svc = make_service(nutrition_repo=nutrition_repo)

        await svc.save_selection(make_ctx(), make_recipe(calories=350.0))
        nutrition_repo.save.assert_awaited_once()

    async def test_nutrition_repo_not_called_when_calories_none(self):
        nutrition_repo = AsyncMock()
        nutrition_repo.save = AsyncMock(return_value=10)
        svc = make_service(nutrition_repo=nutrition_repo)

        await svc.save_selection(make_ctx(), make_recipe(calories=None))
        nutrition_repo.save.assert_not_awaited()

    async def test_recipe_history_contains_correct_user_id(self):
        captured = {}

        async def capture(history: RecipeHistory):
            captured["user_id"] = history.user_id
            return 1

        recipe_repo = AsyncMock()
        recipe_repo.save = capture
        svc = make_service(recipe_repo=recipe_repo)

        await svc.save_selection(make_ctx(user_id=99), make_recipe())
        assert captured["user_id"] == 99

    async def test_recipe_history_contains_correct_name(self):
        captured = {}

        async def capture(history: RecipeHistory):
            captured["recipe_name"] = history.recipe_name
            return 1

        recipe_repo = AsyncMock()
        recipe_repo.save = capture
        svc = make_service(recipe_repo=recipe_repo)

        await svc.save_selection(make_ctx(), make_recipe(name="Grilled Salmon"))
        assert captured["recipe_name"] == "Grilled Salmon"

    async def test_rating_stored_in_recipe_history(self):
        captured = {}

        async def capture(history: RecipeHistory):
            captured["rating"] = history.rating
            return 1

        recipe_repo = AsyncMock()
        recipe_repo.save = capture
        svc = make_service(recipe_repo=recipe_repo)

        await svc.save_selection(make_ctx(), make_recipe(), rating=4)
        assert captured["rating"] == 4

    async def test_ingredients_joined_as_string(self):
        captured = {}

        async def capture(history: RecipeHistory):
            captured["ingredients"] = history.ingredients
            return 1

        recipe_repo = AsyncMock()
        recipe_repo.save = capture
        svc = make_service(recipe_repo=recipe_repo)

        recipe = make_recipe(ingredients=["chicken", "broccoli", "garlic"])
        await svc.save_selection(make_ctx(), recipe)
        assert captured["ingredients"] == "chicken, broccoli, garlic"

    async def test_nutrition_history_linked_to_history_id(self):
        captured = {}

        async def capture_nutrition(nh: NutritionHistory):
            captured["recipe_id"] = nh.recipe_id
            return 1

        recipe_repo = AsyncMock()
        recipe_repo.save = AsyncMock(return_value=77)
        nutrition_repo = AsyncMock()
        nutrition_repo.save = capture_nutrition
        svc = make_service(recipe_repo=recipe_repo, nutrition_repo=nutrition_repo)

        await svc.save_selection(make_ctx(), make_recipe(calories=300.0))
        assert captured["recipe_id"] == 77


# ---------------------------------------------------------------------------
# get_history()
# ---------------------------------------------------------------------------

class TestGetHistory:
    async def test_returns_history_list(self):
        mock_history = [
            RecipeHistory(recipe_name="Salad", user_id=1),
            RecipeHistory(recipe_name="Soup", user_id=1),
        ]
        recipe_repo = AsyncMock()
        recipe_repo.save = AsyncMock(return_value=1)
        recipe_repo.get_by_user = AsyncMock(return_value=mock_history)
        svc = make_service(recipe_repo=recipe_repo)

        history = await svc.get_history(make_ctx(user_id=1))
        assert len(history) == 2
        assert history[0].recipe_name == "Salad"

    async def test_get_by_user_called_with_correct_user_id(self):
        recipe_repo = AsyncMock()
        recipe_repo.save = AsyncMock(return_value=1)
        recipe_repo.get_by_user = AsyncMock(return_value=[])
        svc = make_service(recipe_repo=recipe_repo)

        await svc.get_history(make_ctx(user_id=42))
        recipe_repo.get_by_user.assert_awaited_once_with(42)

    async def test_returns_empty_list_when_no_history(self):
        recipe_repo = AsyncMock()
        recipe_repo.save = AsyncMock(return_value=1)
        recipe_repo.get_by_user = AsyncMock(return_value=[])
        svc = make_service(recipe_repo=recipe_repo)

        result = await svc.get_history(make_ctx())
        assert result == []
