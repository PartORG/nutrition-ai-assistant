"""
yolo_detector.detector - YOLO + Food101 ResNet18 ingredient detection pipeline.

Adapted from new_scripts/cnn/yolo/cnn_class.py for service use:
- Accepts image file paths (no webcam capture)
- Configurable model paths via environment variables
- Returns structured detection results

Models required:
    FOOD_MODEL_PATH  - path to food101_resnet18_best.pth (custom trained)
    YOLO_MODEL_PATH  - path to yolov8n.pt (auto-downloaded if not found)

Food101 class names are embedded here so the Docker container does not
need the full Food101 dataset directory at runtime.
"""

from __future__ import annotations

import os
import logging
from typing import Any

import cv2
import torch
from torchvision import models, transforms
from PIL import Image
from ultralytics import YOLO

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Food101 class names (101 categories, alphabetical order as torchvision loads them)
# Embedded to avoid requiring the full dataset in the Docker image.
# ---------------------------------------------------------------------------
FOOD101_CLASSES = [
    "apple_pie", "baby_back_ribs", "baklava", "beef_carpaccio", "beef_tartare",
    "beet_salad", "beignets", "bibimbap", "bread_pudding", "breakfast_burrito",
    "bruschetta", "caesar_salad", "cannoli", "caprese_salad", "carrot_cake",
    "ceviche", "cheese_plate", "cheesecake", "chicken_curry", "chicken_quesadilla",
    "chicken_wings", "chocolate_cake", "chocolate_mousse", "churros", "clam_chowder",
    "club_sandwich", "crab_cakes", "creme_brulee", "croque_madame", "cup_cakes",
    "deviled_eggs", "donuts", "dumplings", "edamame", "eggs_benedict",
    "escargots", "falafel", "filet_mignon", "fish_and_chips", "foie_gras",
    "french_fries", "french_onion_soup", "french_toast", "fried_calamari",
    "fried_rice", "frozen_yogurt", "garlic_bread", "gnocchi", "greek_salad",
    "grilled_cheese_sandwich", "grilled_salmon", "guacamole", "gyoza", "hamburger",
    "hot_and_sour_soup", "hot_dog", "huevos_rancheros", "hummus", "ice_cream",
    "lasagna", "lobster_bisque", "lobster_roll_sandwich", "macaroni_and_cheese",
    "macarons", "miso_soup", "mussels", "nachos", "omelette", "onion_rings",
    "oysters", "pad_thai", "paella", "pancakes", "panna_cotta", "peking_duck",
    "pho", "pizza", "pork_chop", "poutine", "prime_rib", "pulled_pork_sandwich",
    "ramen", "ravioli", "red_velvet_cake", "risotto", "samosa", "sashimi",
    "scallops", "seaweed_salad", "shrimp_and_grits", "spaghetti_bolognese",
    "spaghetti_carbonara", "spring_rolls", "steak", "strawberry_shortcake",
    "sushi", "tacos", "takoyaki", "tiramisu", "tuna_tartare", "waffles",
]

# YOLO COCO classes that are food-related (used as first-stage filter)
_YOLO_FOOD_CLASSES = {
    "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake",
}


class YOLOFoodDetector:
    """Two-stage food detector: YOLO detection → Food101 ResNet18 classification.

    Stage 1: YOLOv8 detects objects in the image, filtered to COCO food classes.
    Stage 2: ResNet18 (fine-tuned on Food101) classifies each detected crop.

    Returns a structured list of detections with confidence scores.
    """

    def __init__(
        self,
        yolo_model_path: str = "yolov8n.pt",
        food_model_path: str = "/app/models/food101_resnet18_best.pth",
        conf_threshold: float = 0.6,
        food_conf_threshold: float = 0.5,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.conf_threshold = conf_threshold
        self.food_conf_threshold = food_conf_threshold
        self.class_names = FOOD101_CLASSES

        logger.info("Loading YOLO model from %s", yolo_model_path)
        self.yolo = YOLO(yolo_model_path)

        logger.info("Loading Food101 classifier from %s (device=%s)", food_model_path, self.device)
        self.food_model = models.resnet18(weights=None)
        self.food_model.fc = torch.nn.Linear(self.food_model.fc.in_features, 101)
        self.food_model.load_state_dict(
            torch.load(food_model_path, map_location=self.device, weights_only=True)
        )
        self.food_model.to(self.device)
        self.food_model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        # Build the set of YOLO class names that are food-related
        self._food_yolo_classes = {
            name for name in self.yolo.names.values()
            if name in _YOLO_FOOD_CLASSES
        }
        logger.info(
            "YOLOFoodDetector ready — food classes: %s",
            sorted(self._food_yolo_classes),
        )

    def _classify_crop(self, crop_bgr) -> tuple[str, float]:
        """Run Food101 ResNet18 on a BGR numpy crop. Returns (class_name, confidence)."""
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(crop_rgb)
        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.food_model(tensor)
            probs = torch.softmax(logits, dim=1)
            idx = torch.argmax(probs, dim=1).item()
            conf = float(probs[0][idx])

        return self.class_names[idx], conf

    def run(self, image_path: str) -> dict[str, Any]:
        """Detect and classify food items in an image file.

        Args:
            image_path: Absolute path to the image file.

        Returns:
            Dict with keys:
                "objects"     - list of detection dicts
                "description" - human-readable summary
        """
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"Could not read image: {image_path}")

        results = self.yolo(frame)
        r = results[0]

        detections = []
        for box in r.boxes:
            label = r.names[int(box.cls[0])]
            conf = float(box.conf[0])

            if conf < self.conf_threshold:
                continue
            if label not in self._food_yolo_classes:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            food_label, food_conf = self._classify_crop(crop)

            # If Food101 confidence is too low, fall back to the YOLO COCO class name.
            # Food101 is trained on plated dishes — raw ingredients (e.g. carrot) have
            # no Food101 equivalent, so the model guesses a visually similar dish name
            # (e.g. "foie gras") with low confidence instead of the correct ingredient.
            if food_conf < self.food_conf_threshold:
                food_label = label          # use the YOLO name (e.g. "carrot", "cake")
                food_conf = conf            # reuse the YOLO detection confidence
                logger.info(
                    "Food101 low confidence (%.2f) for '%s' crop — using YOLO label '%s'",
                    food_conf, label, label,
                )

            detections.append({
                "detected_object": label,
                "detection_confidence": round(conf, 3),
                "classified_food": food_label,
                "food_confidence": round(food_conf, 3),
            })

        return {
            "objects": detections,
            "description": f"Detected {len(detections)} food item(s) in the image.",
        }
