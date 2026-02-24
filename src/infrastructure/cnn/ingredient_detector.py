"""
infrastructure.cnn.ingredient_detector - LLaVA-based ingredient detection.

Uses LLaVA (Large Language and Vision Assistant) via Ollama for zero-shot,
context-aware ingredient detection from food photos.

Why LLaVA over CLIP:
  - Already running Ollama — no extra downloads needed (just `ollama pull llava`)
  - Understands food context (e.g. "browned beef" = beef, not just "brown thing")
  - Returns natural language ingredient names, not similarity scores
  - Handles complex scenes (multiple dishes, garnishes, sauces)

Setup:
    ollama pull llava

Usage:
    detector = LLaVAIngredientDetector(ollama_base_url="http://localhost:11434")
    result = await detector.detect("/path/to/food.jpg")
"""

from __future__ import annotations

import asyncio
import base64
import logging
import re
from pathlib import Path

from domain.models import DetectedIngredients
from domain.exceptions import IngredientDetectionError

logger = logging.getLogger(__name__)


class LLaVAIngredientDetector:
    """Detect food ingredients from images using LLaVA via Ollama.

    No training required. Requires Ollama running with llava model pulled:
        ollama pull llava
    """

    _PROMPT = (
        "Look at this food image carefully. "
        "List every ingredient or food item you can see. "
        "Respond with ONLY a comma-separated list of ingredient names, nothing else. "
        "Example: chicken breast, garlic, olive oil, rosemary, lemon"
    )

    def __init__(
        self,
        ollama_base_url: str = "http://localhost:11434",
        model: str = "llava",
    ):
        self._base_url = ollama_base_url.rstrip("/")
        self._model = model

    async def detect(self, image_path: str) -> DetectedIngredients:
        """Detect ingredients from a food image using LLaVA."""
        path = Path(image_path)
        if not path.exists():
            raise IngredientDetectionError(f"Image file not found: {image_path}")

        loop = asyncio.get_event_loop()
        try:
            ingredients = await loop.run_in_executor(
                None, self._run_llava, str(path)
            )
        except IngredientDetectionError:
            raise
        except Exception as e:
            raise IngredientDetectionError(f"LLaVA inference failed: {e}") from e

        return DetectedIngredients(
            ingredients=ingredients,
            # LLaVA doesn't give confidence scores — mark all as high confidence
            confidence_scores={ing: 0.95 for ing in ingredients},
            image_path=image_path,
            source="LLaVA",
        )

    def _run_llava(self, image_path: str) -> list[str]:
        """Call Ollama LLaVA synchronously (runs in thread pool)."""
        try:
            import ollama
        except ImportError as e:
            raise IngredientDetectionError(
                "ollama package is required. Run: pip install ollama"
            ) from e

        # Encode image to base64
        image_bytes = Path(image_path).read_bytes()
        image_b64 = base64.b64encode(image_bytes).decode()

        logger.info("Sending image to LLaVA (%s) at %s", self._model, self._base_url)

        client = ollama.Client(host=self._base_url)
        response = client.chat(
            model=self._model,
            messages=[{
                "role": "user",
                "content": self._PROMPT,
                "images": [image_b64],
            }],
        )

        raw = response["message"]["content"].strip()
        logger.info("LLaVA raw response: %s", raw)

        return self._parse_ingredients(raw)

    @staticmethod
    def _parse_ingredients(text: str) -> list[str]:
        """Parse comma-separated ingredient list from LLaVA response."""
        # Strip markdown, bullet points, numbering if LLaVA added them anyway
        text = re.sub(r"[*_`#]", "", text)
        text = re.sub(r"^\s*[\-\d]+[\.\)]\s*", "", text, flags=re.MULTILINE)

        # Split by comma or newline
        parts = re.split(r"[,\n]+", text)

        ingredients = []
        for part in parts:
            clean = part.strip().lower()
            # Skip empty, very long (sentences), or non-ingredient phrases
            if clean and 2 <= len(clean) <= 40 and not clean.startswith("i "):
                ingredients.append(clean)

        return ingredients


# Alias so existing imports (factory.py) still work
CNNIngredientDetector = LLaVAIngredientDetector
