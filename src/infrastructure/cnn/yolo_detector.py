"""
infrastructure.cnn.yolo_detector - YOLO-based ingredient detection.

Implements IngredientDetectorPort with an Ultralytics YOLO model, as an
alternative to the LLaVA detector. The trade-off between the two:

    LLaVA   zero-shot, no training, open vocabulary, but no real confidence
            scores and a round-trip to the Ollama container per image.
    YOLO    fixed vocabulary and needs training to know food ingredients, but
            runs locally in ~50 ms on CPU and returns genuine per-box
            confidence, plus box counts (three apples, not "apple").

Out of the box this loads a COCO-pretrained checkpoint, which knows only a
handful of foods (banana, apple, orange, broccoli, carrot, sandwich, pizza,
hot dog, donut, cake). To detect real ingredients, fine-tune on a labelled
food dataset and point YOLO_WEIGHTS at the resulting best.pt — see
training/README.md.

Setup:
    # uses a COCO checkpoint, auto-downloaded on first run
    YOLO_WEIGHTS=yolo11n.pt

    # or your own fine-tuned model
    YOLO_WEIGHTS=models/ingredients/best.pt
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from domain.exceptions import IngredientDetectionError
from domain.models import DetectedIngredients

logger = logging.getLogger(__name__)

# COCO class names that are actually edible, mapped to ingredient wording.
# Only consulted when the loaded model still has stock COCO labels; a
# fine-tuned model's own class names are used verbatim.
_COCO_FOOD_ALIASES: dict[str, str] = {
    "banana": "banana",
    "apple": "apple",
    "sandwich": "sandwich",
    "orange": "orange",
    "broccoli": "broccoli",
    "carrot": "carrot",
    "hot dog": "hot dog",
    "pizza": "pizza",
    "donut": "donut",
    "cake": "cake",
}


class YOLOIngredientDetector:
    """Detect food ingredients from images using an Ultralytics YOLO model.

    The model is loaded lazily on first detection, not in __init__, so that
    constructing the detector (and therefore app startup) stays cheap.
    """

    def __init__(
        self,
        weights: str = "yolo11n.pt",
        confidence_threshold: float = 0.25,
        device: str = "cpu",
        filter_non_food: bool = True,
    ):
        self._weights = weights
        self._conf = confidence_threshold
        self._device = device
        self._filter_non_food = filter_non_food
        self._model = None  # loaded on first use

    # ------------------------------------------------------------------
    # Port implementation
    # ------------------------------------------------------------------

    async def detect(self, image_path: str) -> DetectedIngredients:
        """Detect ingredients in a food image. Implements IngredientDetectorPort."""
        path = Path(image_path)
        if not path.exists():
            raise IngredientDetectionError(f"Image file not found: {image_path}")

        try:
            return await asyncio.to_thread(self._run_yolo, str(path))
        except IngredientDetectionError:
            raise
        except Exception as e:
            raise IngredientDetectionError(f"YOLO inference failed: {e}") from e

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _load_model(self):
        """Load and cache the YOLO model. Called on first detection."""
        if self._model is not None:
            return self._model
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise IngredientDetectionError(
                "ultralytics is required for YOLO detection. Run: uv sync"
            ) from e

        logger.info("Loading YOLO weights: %s (device=%s)", self._weights, self._device)
        self._model = YOLO(self._weights)
        return self._model

    def _is_stock_coco(self, names: dict[int, str]) -> bool:
        """True if the model still carries stock COCO labels rather than food classes."""
        return "person" in names.values() and "traffic light" in names.values()

    def _run_yolo(self, image_path: str) -> DetectedIngredients:
        model = self._load_model()

        results = model.predict(
            source=image_path,
            conf=self._conf,
            device=self._device,
            verbose=False,
        )
        if not results:
            return DetectedIngredients(image_path=image_path, source="YOLO")

        names: dict[int, str] = results[0].names
        stock_coco = self._is_stock_coco(names)

        # Keep the highest confidence per class — five boxes of "apple" is
        # still one ingredient, and the best box is the honest score for it.
        best: dict[str, float] = {}
        for box in results[0].boxes:
            label = names[int(box.cls)]
            confidence = float(box.conf)

            if stock_coco and self._filter_non_food:
                if label not in _COCO_FOOD_ALIASES:
                    continue
                label = _COCO_FOOD_ALIASES[label]

            if confidence > best.get(label, 0.0):
                best[label] = confidence

        if stock_coco and self._filter_non_food and not best:
            logger.warning(
                "YOLO found no food in %s using stock COCO labels. This model "
                "only knows 10 foods — fine-tune it to detect real ingredients.",
                image_path,
            )

        ingredients = sorted(best, key=lambda k: best[k], reverse=True)
        return DetectedIngredients(
            ingredients=ingredients,
            confidence_scores=best,
            image_path=image_path,
            source="YOLO",
        )
