"""
application.services.recipe_manager - Recipe selection and history management.

Handles saving user recipe selections and retrieving history.
Replaces the database_handling_tool_func from host_agent.py.

Key improvements:
    - Receives typed Recipe objects (no markdown parsing)
    - No global state access
    - No hash-based recipe_id generation (DB auto-increments)
    - Async throughout
"""

from __future__ import annotations

import logging

from domain.models import Recipe, NutritionValues
from domain.entities import RecipeHistory, NutritionHistory
from domain.ports import RecipeRepository, NutritionRepository
from application.context import SessionContext

logger = logging.getLogger(__name__)


class RecipeManagerService:
    """Manages recipe selection persistence."""

    def __init__(
        self,
        recipe_repo: RecipeRepository,
        nutrition_repo: NutritionRepository,
    ):
        self._recipe_repo = recipe_repo
        self._nutrition_repo = nutrition_repo

    async def save_selection(
        self,
        ctx: SessionContext,
        recipe: Recipe,
        rating: int | None = None,
    ) -> int:
        """Save a user's recipe selection to history.

        Args:
            ctx:    Session context with user_id.
            recipe: The typed Recipe object (from RecommendationResult.safe_recipes).
            rating: Optional user rating (1-5).

        Returns:
            The ID of the saved recipe history record.
        """
        logger.info(
            "Saving recipe '%s' for user %d (request=%s)",
            recipe.name, ctx.user_id, ctx.request_id,
        )

        history = RecipeHistory(
            user_id=ctx.user_id,
            recipe_name=recipe.name,
            servings=recipe.servings,
            ingredients=", ".join(recipe.ingredients),
            cook_instructions=recipe.cook_instructions,
            prep_time=recipe.prep_time,
        )
        history_id = await self._recipe_repo.save(history)

        # Save nutrition if available
        nutrition = recipe.nutrition
        if nutrition and nutrition.calories is not None:
            nh = NutritionHistory(
                user_id=ctx.user_id,
                recipe_id=history_id,
                calories=nutrition.calories or 0.0,
                protein=nutrition.protein_g or 0.0,
                fat=nutrition.fat_g or 0.0,
                carbohydrates=nutrition.carbs_g or 0.0,
                fiber=nutrition.fiber_g or 0.0,
                sugar=nutrition.sugar_g or 0.0,
                sodium=nutrition.sodium_mg or 0.0,
            )
            await self._nutrition_repo.save(nh)

        logger.info("Recipe saved with history_id=%d", history_id)
        return history_id

    async def get_history(
        self,
        ctx: SessionContext,
    ) -> list[RecipeHistory]:
        """Retrieve the user's recipe history (newest first)."""
        return await self._recipe_repo.get_by_user(ctx.user_id)
