"""
infrastructure.cnn.ingredient_detector - VLM-based ingredient detection.

Uses any Ollama vision model for zero-shot, open-vocabulary ingredient
detection from food photos. Defaults to qwen2.5vl; `llava` still works and
is selected the same way, via DETECTOR_VLM_MODEL.

Why a VLM at all:
  - Already running Ollama — no extra serving stack
  - Understands food context ("browned beef" = beef, not "brown thing")
  - Open vocabulary: no fine-tuning to know what a shallot is
  - Handles complex scenes (multiple dishes, garnishes, sauces)

Its weakness is the reason for hybrid_detector.py: it returns no calibrated
confidence and can hallucinate a plausible ingredient. YOLO grades it.

Setup:
    ollama pull qwen2.5vl:7b

Usage:
    detector = VLMIngredientDetector(ollama_base_url="http://localhost:11434")
    result = await detector.detect("/path/to/food.jpg")
"""

from __future__ import annotations

import asyncio
import base64
import logging
import re
from pathlib import Path

from domain.exceptions import IngredientDetectionError
from domain.models import DetectedIngredients

logger = logging.getLogger(__name__)


class VLMIngredientDetector:
    """Detect food ingredients from images using an Ollama vision model.

    No training required. Requires Ollama running with a vision model pulled:
        ollama pull qwen2.5vl:7b
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
        model: str = "qwen2.5vl:7b",
        num_gpu: int = 0,
    ):
        self._base_url = ollama_base_url.rstrip("/")
        self._model = model
        # Ollama's memory estimator is conservative for vision models: on a
        # 14 GB box with 6 GB free it refused qwen2.5vl:3b with "requires more
        # system memory (8.6 GiB)" even though the GPU had 7 GB free and
        # nothing was loaded. Forcing the layer count onto the GPU
        # (num_gpu=99 = all) loads it fine. 0 means "send no option and let
        # Ollama decide", which is right on hosts that have the headroom.
        self._num_gpu = num_gpu

    async def detect(self, image_path: str) -> DetectedIngredients:
        """Detect ingredients from a food image using the configured VLM."""
        path = Path(image_path)
        if not path.exists():
            raise IngredientDetectionError(f"Image file not found: {image_path}")

        try:
            ingredients = await asyncio.to_thread(self._run_vlm, str(path))
        except IngredientDetectionError:
            raise
        except Exception as e:
            raise IngredientDetectionError(f"VLM inference failed: {e}") from e

        return DetectedIngredients(
            ingredients=ingredients,
            # A VLM returns no calibrated confidence. 0.95 for everything was a
            # fabricated number that looked like evidence; 0.5 says "detected,
            # unscored". Run the hybrid detector to get a real score.
            confidence_scores={ing: 0.5 for ing in ingredients},
            image_path=image_path,
            source=self._model,
        )

    def _run_vlm(self, image_path: str) -> list[str]:
        """Call the Ollama vision model synchronously (runs in thread pool)."""
        try:
            import ollama
        except ImportError as e:
            raise IngredientDetectionError(
                "ollama package is required. Run: pip install ollama"
            ) from e

        # Encode image to base64
        image_bytes = Path(image_path).read_bytes()
        image_b64 = base64.b64encode(image_bytes).decode()

        logger.info("Sending image to VLM (%s) at %s", self._model, self._base_url)

        client = ollama.Client(host=self._base_url)
        response = client.chat(
            model=self._model,
            messages=[{
                "role": "user",
                "content": self._PROMPT,
                "images": [image_b64],
            }],
            options={"num_gpu": self._num_gpu} if self._num_gpu else None,
        )

        raw = response["message"]["content"].strip()
        logger.info("VLM raw response: %s", raw)

        return self._parse_ingredients(raw)

    @staticmethod
    def _parse_ingredients(text: str) -> list[str]:
        """Parse comma-separated ingredient list from the VLM response."""
        # Strip markdown, bullet points, numbering if the model added them anyway
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

        # A VLM shown a page of text (or stuck in a repetition loop) can emit
        # thousands of items, which then flow into the reconcile loop, the RAG
        # query and the logs. No real plate has 50 ingredients.
        return ingredients[:50]


# Alias: the class was LLaVA-specific before it became model-agnostic.
LLaVAIngredientDetector = VLMIngredientDetector
