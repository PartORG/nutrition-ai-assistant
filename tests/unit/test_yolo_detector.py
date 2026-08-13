"""
Tests for the YOLO ingredient detector.

Inference itself is stubbed — a real forward pass would need weights and add
seconds per test. What matters here is the logic wrapped around it: the port
contract, per-class deduplication, honest confidence, and COCO filtering.
"""

from __future__ import annotations

import pytest

from domain.exceptions import IngredientDetectionError
from infrastructure.cnn.yolo_detector import YOLOIngredientDetector

# A real stock COCO checkpoint carries exactly 80 classes, and the detector
# now keys on that count as well as the labels — an open-vocabulary model
# (YOLOE, 4585 classes) also contains "person" and "traffic light" but must
# NOT have its vocabulary filtered down to COCO's 10 foods.
COCO_NAMES = {0: "person", 1: "traffic light", 2: "apple", 3: "broccoli"}
COCO_NAMES.update({i: f"coco_class_{i}" for i in range(4, 80)})
assert len(COCO_NAMES) == 80

FOOD_NAMES = {0: "chicken", 1: "garlic", 2: "tomato"}
OPEN_VOCAB_NAMES = {0: "person", 1: "traffic light", 2: "rice", 3: "chicken breast"}
OPEN_VOCAB_NAMES.update({i: f"open_class_{i}" for i in range(4, 4585)})


class _Box:
    def __init__(self, cls: int, conf: float):
        self.cls = cls
        self.conf = conf


class _Result:
    def __init__(self, names, boxes):
        self.names = names
        self.boxes = boxes


class _FakeModel:
    """Stands in for ultralytics.YOLO."""

    def __init__(self, names, boxes):
        self._result = _Result(names, boxes)

    def predict(self, **_kwargs):
        return [self._result]


def _detector(names, boxes, **kwargs) -> YOLOIngredientDetector:
    d = YOLOIngredientDetector(**kwargs)
    d._model = _FakeModel(names, boxes)
    return d


async def test_missing_image_raises(tmp_path):
    d = YOLOIngredientDetector()
    with pytest.raises(IngredientDetectionError, match="not found"):
        await d.detect(str(tmp_path / "nope.jpg"))


async def test_returns_real_confidence_not_a_constant(tmp_path):
    """The VLM has no calibrated score; YOLO must report the model's real one."""
    img = tmp_path / "food.jpg"
    img.write_bytes(b"fake")

    d = _detector(FOOD_NAMES, [_Box(0, 0.91), _Box(1, 0.42)])
    result = await d.detect(str(img))

    assert result.confidence_scores == {"chicken": 0.91, "garlic": 0.42}
    assert result.source == "YOLO"
    assert 0.95 not in result.confidence_scores.values()


async def test_duplicate_boxes_collapse_to_best_score(tmp_path):
    """Three tomatoes is one ingredient, scored by its most confident box."""
    img = tmp_path / "food.jpg"
    img.write_bytes(b"fake")

    d = _detector(FOOD_NAMES, [_Box(2, 0.55), _Box(2, 0.88), _Box(2, 0.31)])
    result = await d.detect(str(img))

    assert result.ingredients == ["tomato"]
    assert result.confidence_scores["tomato"] == 0.88


async def test_ingredients_ordered_by_confidence(tmp_path):
    img = tmp_path / "food.jpg"
    img.write_bytes(b"fake")

    d = _detector(FOOD_NAMES, [_Box(1, 0.30), _Box(0, 0.95), _Box(2, 0.60)])
    result = await d.detect(str(img))

    assert result.ingredients == ["chicken", "tomato", "garlic"]


async def test_stock_coco_drops_non_food(tmp_path):
    """A COCO checkpoint sees people and traffic lights — those aren't ingredients."""
    img = tmp_path / "food.jpg"
    img.write_bytes(b"fake")

    d = _detector(COCO_NAMES, [_Box(0, 0.99), _Box(1, 0.88), _Box(2, 0.75)])
    result = await d.detect(str(img))

    assert result.ingredients == ["apple"]
    assert "person" not in result.confidence_scores


async def test_open_vocab_model_is_not_treated_as_stock_coco(tmp_path):
    """YOLOE shares COCO's labels but has 4585 classes — do not filter it.

    Without the class-count check this returned nothing at all: "rice" and
    "chicken breast" are not in the 10-item COCO food alias map.
    """
    img = tmp_path / "food.jpg"
    img.write_bytes(b"fake")

    d = _detector(OPEN_VOCAB_NAMES, [_Box(2, 0.71), _Box(3, 0.66)])
    result = await d.detect(str(img))

    assert set(result.ingredients) == {"rice", "chicken breast"}


async def test_finetuned_model_labels_pass_through_unfiltered(tmp_path):
    """A custom model's classes are all food — never filter them against COCO."""
    img = tmp_path / "food.jpg"
    img.write_bytes(b"fake")

    d = _detector(FOOD_NAMES, [_Box(0, 0.8), _Box(1, 0.7), _Box(2, 0.6)])
    result = await d.detect(str(img))

    assert set(result.ingredients) == {"chicken", "garlic", "tomato"}


async def test_no_detections_returns_empty_not_error(tmp_path):
    img = tmp_path / "food.jpg"
    img.write_bytes(b"fake")

    d = _detector(FOOD_NAMES, [])
    result = await d.detect(str(img))

    assert result.ingredients == []
    assert result.image_path == str(img)
