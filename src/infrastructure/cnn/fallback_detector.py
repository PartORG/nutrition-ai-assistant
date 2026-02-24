"""
infrastructure.cnn.fallback_detector - Two-stage ingredient detector with fallback.

Tries the YOLO microservice first. Falls back to LLaVA if:
  1. The YOLO service is unreachable (connection error / timeout)
  2. YOLO returns an empty ingredient list (image contains no recognizable COCO food
     classes — e.g. raw ingredients, spice jars, packaged food)
  3. YOLO returns only 1 ingredient — too few to build a meaningful recipe query;
     LLaVA is used instead to get a richer ingredient description from the full image.

This gives the best of both detectors:
  - YOLO: fast, structured, high confidence on whole-food items (pizza, banana, etc.)
  - LLaVA: flexible, handles any ingredient via natural language description

Usage (wired in factory.py):
    yolo = YOLOServiceDetector(service_url=config.yolo_service_url)
    llava = LLaVAIngredientDetector(ollama_base_url=config.ollama_base_url)
    detector = FallbackIngredientDetector(primary=yolo, fallback=llava)
"""

from __future__ import annotations

import asyncio
import logging

from domain.models import DetectedIngredients
from domain.ports import IngredientDetectorPort
from domain.exceptions import IngredientDetectionError

logger = logging.getLogger(__name__)


class FallbackIngredientDetector:
    """Try primary detector; fall back to secondary on failure or empty result.

    Implements IngredientDetectorPort (structural typing — no explicit inheritance).

    Designed for YOLO-primary + LLaVA-fallback, but generic enough for any pair.
    """

    def __init__(
        self,
        primary: IngredientDetectorPort,
        fallback: IngredientDetectorPort,
        timeout: float = 10.0,
    ):
        self._primary = primary
        self._fallback = fallback
        self._timeout = timeout

    async def detect(self, image_path: str) -> DetectedIngredients:
        """Detect ingredients, falling back to secondary on failure or empty result.

        Args:
            image_path: Path to the image file.

        Returns:
            DetectedIngredients from whichever detector succeeded.

        Raises:
            IngredientDetectionError: Only if BOTH detectors fail.
        """
        # --- Try primary (YOLO) ---
        try:
            result = await asyncio.wait_for(
                self._primary.detect(image_path),
                timeout=self._timeout,
            )
            if len(result.ingredients) > 1:
                logger.info(
                    "YOLO detected %d ingredient(s) — using YOLO result",
                    len(result.ingredients),
                )
                return result

            # YOLO found 0 or 1 ingredient — not enough for a useful recipe search.
            # Fall back to LLaVA which describes the full image in natural language.
            if result.ingredients:
                logger.info(
                    "YOLO returned only 1 ingredient ('%s') for %s "
                    "— falling back to LLaVA for richer detection",
                    result.ingredients[0], image_path,
                )
            else:
                logger.info(
                    "YOLO returned 0 ingredients for %s — falling back to LLaVA",
                    image_path,
                )

        except asyncio.TimeoutError:
            logger.warning(
                "YOLO service timed out after %.1fs — falling back to LLaVA",
                self._timeout,
            )
        except IngredientDetectionError as e:
            logger.warning("YOLO service unavailable (%s) — falling back to LLaVA", e)
        except Exception as e:
            logger.warning("YOLO unexpected error (%s) — falling back to LLaVA", e)

        # --- Fallback (LLaVA) ---
        logger.info("Running LLaVA ingredient detection for %s", image_path)
        return await self._fallback.detect(image_path)
