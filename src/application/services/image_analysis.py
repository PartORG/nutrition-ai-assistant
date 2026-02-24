"""
application.services.image_analysis - CNN ingredient detection + recommendation chaining.

New service for the image-to-recipe flow:
    1. User uploads a food photo
    2. CNN detects ingredients
    3. (Optionally) Detected ingredients are fed into the recommendation pipeline

The service composes IngredientDetectorPort with RecommendationService.
"""

from __future__ import annotations

import logging
from typing import Optional

from domain.models import DetectedIngredients
from domain.ports import IngredientDetectorPort
from application.context import SessionContext
from application.dto import ImageAnalysisResult, RecommendationResult

logger = logging.getLogger(__name__)


class ImageAnalysisService:
    """Detects ingredients from images and optionally chains into recommendations."""

    def __init__(
        self,
        detector: IngredientDetectorPort,
        recommendation_service: Optional[object] = None,
    ):
        self._detector = detector
        # Avoid circular import: recommendation_service is typed loosely here,
        # but at runtime it's a RecommendationService instance.
        self._recommendation_service = recommendation_service

    async def detect_ingredients(
        self,
        ctx: SessionContext,
        image_path: str,
    ) -> DetectedIngredients:
        """Detect ingredients from a food photo.

        Args:
            ctx:        Session context.
            image_path: Path to the uploaded image file.

        Returns:
            DetectedIngredients with ingredient list and confidence scores.
        """
        logger.info(
            "Detecting ingredients from image for user %d: %s",
            ctx.user_id, image_path,
        )
        detected = await self._detector.detect(image_path)
        logger.info(
            "Detected %d ingredients: %s",
            len(detected.ingredients),
            ", ".join(detected.ingredients[:5]),
        )
        return detected

    async def recommend_from_image(
        self,
        ctx: SessionContext,
        image_path: str,
        additional_query: str = "",
    ) -> ImageAnalysisResult:
        """Detect ingredients from image, then run recommendation pipeline.

        This is the full image-to-recipe flow:
            1. CNN detects ingredients
            2. Build query: "I have these ingredients: X, Y, Z"
            3. Run recommendation pipeline with the built query

        Args:
            ctx:              Session context.
            image_path:       Path to the uploaded image.
            additional_query: Extra user instructions (e.g., "make it low-carb").

        Returns:
            ImageAnalysisResult with detection results and recommendations.
        """
        detected = await self.detect_ingredients(ctx, image_path)

        if not detected.ingredients:
            logger.warning("No ingredients detected from image")
            return ImageAnalysisResult(detected=detected, recommendation=None)

        if self._recommendation_service is None:
            logger.warning("No recommendation service configured — returning detection only")
            return ImageAnalysisResult(detected=detected, recommendation=None)

        # Build a query from detected ingredients
        ingredients_str = ", ".join(detected.ingredients)
        query = f"I have these ingredients at home: {ingredients_str}. Suggest recipes using them."
        if additional_query:
            query += f" {additional_query}"

        logger.info("Chaining to recommendation pipeline with query: %s", query[:100])

        # Pass `additional_query` as `intent_query` so the intent parser sees
        # what the user actually typed (e.g. "low-carb breakfast with fish"),
        # not the constructed ingredient-list string used for recipe retrieval.
        recommendation = await self._recommendation_service.get_recommendations(
            ctx, query, intent_query=additional_query,
        )

        return ImageAnalysisResult(
            detected=detected,
            recommendation=recommendation,
        )
