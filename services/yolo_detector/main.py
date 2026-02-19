"""
yolo_detector.main - FastAPI microservice for YOLO-based ingredient detection.

Exposes two endpoints:
    POST /detect  - accepts base64-encoded image, returns detected ingredients
    GET  /health  - liveness probe

Environment variables:
    YOLO_MODEL_PATH  - path to yolov8n.pt (default: "yolov8n.pt", auto-downloaded)
    FOOD_MODEL_PATH  - path to food101_resnet18_best.pth (required)
    CONF_THRESHOLD   - detection confidence threshold (default: 0.6)

Usage:
    uvicorn main:app --host 0.0.0.0 --port 8001
"""

from __future__ import annotations

import base64
import logging
import os
import tempfile
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from detector import YOLOFoodDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class DetectRequest(BaseModel):
    image_base64: str


class DetectResponse(BaseModel):
    ingredients: list[str]
    confidence_scores: dict[str, float]
    image_path: str = ""


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

_detector: YOLOFoodDetector | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _detector
    yolo_path = os.getenv("YOLO_MODEL_PATH", "yolov8n.pt")
    food_path = os.getenv("FOOD_MODEL_PATH", "/app/models/food101_resnet18_best.pth")
    conf = float(os.getenv("CONF_THRESHOLD", "0.6"))

    logger.info("Loading YOLO food detector...")
    _detector = YOLOFoodDetector(
        yolo_model_path=yolo_path,
        food_model_path=food_path,
        conf_threshold=conf,
    )
    logger.info("YOLO food detector ready")
    yield
    _detector = None


app = FastAPI(
    title="YOLO Ingredient Detector",
    version="1.0.0",
    description="Detects food ingredients from images using YOLOv8 + Food101 ResNet18.",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    """Liveness probe."""
    return {"status": "ok", "model_loaded": _detector is not None}


@app.post("/detect", response_model=DetectResponse)
async def detect(request: DetectRequest):
    """Detect food ingredients from a base64-encoded image.

    Returns deduplicated ingredients with their highest Food101 confidence scores.
    """
    if _detector is None:
        raise HTTPException(status_code=503, detail="Detector not initialized")

    # Decode base64 → temp file
    try:
        image_bytes = base64.b64decode(request.image_base64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image data")

    suffix = ".jpg"
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(image_bytes)
            tmp_path = f.name

        result = _detector.run(tmp_path)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error("Detection failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Detection error: {e}")
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    # Map detections → deduplicated ingredients with best confidence
    best: dict[str, float] = {}
    for obj in result["objects"]:
        name = obj["classified_food"].replace("_", " ")
        conf = obj["food_confidence"]
        if name not in best or conf > best[name]:
            best[name] = conf

    ingredients = list(best.keys())
    confidence_scores = best

    logger.info(
        "Detected %d ingredient(s): %s",
        len(ingredients),
        ", ".join(ingredients[:5]),
    )

    return DetectResponse(
        ingredients=ingredients,
        confidence_scores=confidence_scores,
    )
