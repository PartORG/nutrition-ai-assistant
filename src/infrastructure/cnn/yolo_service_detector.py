"""
infrastructure.cnn.yolo_service_detector - HTTP client for the YOLO microservice.

Implements IngredientDetectorPort by calling the yolo-detector FastAPI microservice.
Uses requests (already a project dependency) via run_in_executor for async compat.

The microservice accepts base64-encoded images and returns detected food ingredients.
If the service is unreachable, raises IngredientDetectionError — the caller
(FallbackIngredientDetector) catches this and falls back to LLaVA.

Setup:
    Start the yolo-detector service:
        docker-compose up yolo-detector
    Or locally:
        cd services/yolo_detector && uvicorn main:app --port 8001
"""

from __future__ import annotations

import asyncio
import base64
import logging
from pathlib import Path

import requests

from domain.models import DetectedIngredients
from domain.exceptions import IngredientDetectionError

logger = logging.getLogger(__name__)


class YOLOServiceDetector:
    """Call the YOLO microservice to detect food ingredients from an image.

    Implements IngredientDetectorPort (structural typing — no explicit inheritance).

    The microservice runs in its own Python environment with ultralytics/torch,
    avoiding dependency conflicts with the main application.
    """

    def __init__(
        self,
        service_url: str = "http://localhost:8001",
        timeout: float = 15.0,
    ):
        self._detect_url = service_url.rstrip("/") + "/detect"
        self._health_url = service_url.rstrip("/") + "/health"
        self._timeout = timeout

    async def detect(self, image_path: str) -> DetectedIngredients:
        """Detect food ingredients by calling the YOLO microservice.

        Args:
            image_path: Path to the image file on disk.

        Returns:
            DetectedIngredients with ingredient list and Food101 confidence scores.

        Raises:
            IngredientDetectionError: If the service is unreachable or returns an error.
        """
        path = Path(image_path)
        if not path.exists():
            raise IngredientDetectionError(f"Image file not found: {image_path}")

        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(
                None, self._call_service, str(path)
            )
        except IngredientDetectionError:
            raise
        except Exception as e:
            raise IngredientDetectionError(f"YOLO service call failed: {e}") from e

        return DetectedIngredients(
            ingredients=result["ingredients"],
            confidence_scores=result["confidence_scores"],
            image_path=image_path,
        )

    def _call_service(self, image_path: str) -> dict:
        """Synchronous HTTP call to the YOLO microservice (runs in thread pool)."""
        image_b64 = base64.b64encode(Path(image_path).read_bytes()).decode()

        logger.info("Calling YOLO service at %s", self._detect_url)
        try:
            response = requests.post(
                self._detect_url,
                json={"image_base64": image_b64},
                timeout=self._timeout,
            )
        except requests.exceptions.ConnectionError as e:
            raise IngredientDetectionError(
                f"YOLO service unreachable at {self._detect_url}: {e}"
            ) from e
        except requests.exceptions.Timeout:
            raise IngredientDetectionError(
                f"YOLO service timed out after {self._timeout}s"
            )

        if not response.ok:
            raise IngredientDetectionError(
                f"YOLO service returned HTTP {response.status_code}: {response.text[:200]}"
            )

        data = response.json()
        logger.info(
            "YOLO service returned %d ingredient(s): %s",
            len(data.get("ingredients", [])),
            ", ".join(data.get("ingredients", [])[:5]),
        )
        return data
