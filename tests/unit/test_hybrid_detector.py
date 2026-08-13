"""
Tests for the VLM+YOLO cross-validating detector.

Both backends are stubbed — what matters is the reconciliation logic: that
agreement raises confidence, that a VLM-only guess is marked unverified, and
above all that one backend dying does not take the request down with it.
"""

from __future__ import annotations

import pytest

from domain.exceptions import IngredientDetectionError
from domain.models import DetectedIngredients
from infrastructure.cnn.hybrid_detector import HybridIngredientDetector, _agrees

# Stand-in for the words loaded from data/cleaned_nutrition.csv.
VOCAB = frozenset({
    "apple", "lime", "broccoli", "chicken", "breast", "rice", "carrot",
    "soup", "fruit", "dish", "cup", "plate", "saffron", "garlic", "tomato",
})


class _Stub:
    """Stands in for either detector."""

    def __init__(self, result=None, error=None):
        self._result = result
        self._error = error

    async def detect(self, image_path: str) -> DetectedIngredients:
        if self._error is not None:
            raise self._error
        return self._result


def _vlm(*names: str) -> _Stub:
    return _Stub(DetectedIngredients(
        ingredients=list(names),
        confidence_scores={n: 0.5 for n in names},
        source="qwen2.5vl:7b",
    ))


def _yolo(**scores: float) -> _Stub:
    return _Stub(DetectedIngredients(
        ingredients=sorted(scores, key=lambda k: scores[k], reverse=True),
        confidence_scores=dict(scores),
        source="YOLO",
    ))


# ── name matching ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("a,b", [
    ("broccoli", "broccoli"),
    ("broccoli florets", "broccoli"),
    ("chicken breast", "chicken"),
    ("brown rice", "rice"),
    ("chopped tomatoes", "tomato"),
])
def test_names_that_should_agree(a, b):
    assert _agrees(a, b)


@pytest.mark.parametrize("a,b", [
    ("apple", "pineapple"),      # substring matching would wrongly pair these
    ("green beans", "green pepper"),  # colour alone must not match
    ("chicken", "rice"),
])
def test_names_that_should_not_agree(a, b):
    assert not _agrees(a, b)


# ── reconciliation ─────────────────────────────────────────────────────────

async def test_agreement_beats_vlm_only_confidence():
    """A cross-confirmed ingredient must outrank one only the VLM saw."""
    d = HybridIngredientDetector(
        vlm=_vlm("broccoli", "saffron"), yolo=_yolo(broccoli=0.7),
    )
    r = await d.detect("/x.jpg")

    assert r.confirmed == ["broccoli"]
    assert r.confidence_scores["broccoli"] > r.confidence_scores["saffron"]
    assert r.ingredients[0] == "broccoli"


async def test_vlm_only_ingredient_is_not_given_a_fabricated_score():
    d = HybridIngredientDetector(vlm=_vlm("saffron"), yolo=_yolo())
    r = await d.detect("/x.jpg")

    assert r.confirmed == []
    assert r.confidence_scores["saffron"] == 0.5


async def test_confidence_tracks_the_real_yolo_score():
    """Two confirmed ingredients with different YOLO scores must not tie."""
    d = HybridIngredientDetector(
        vlm=_vlm("broccoli", "carrot"), yolo=_yolo(broccoli=0.9, carrot=0.3),
    )
    r = await d.detect("/x.jpg")

    assert r.confidence_scores["broccoli"] > r.confidence_scores["carrot"]
    assert set(r.confirmed) == {"broccoli", "carrot"}


async def test_yolo_only_scene_nouns_are_discarded():
    """Open-vocab YOLO emits scenery; it must not become an ingredient."""
    d = HybridIngredientDetector(
        vlm=_vlm("broccoli"),
        yolo=_yolo(broccoli=0.8, chair=0.9, **{"still life": 0.7}),
        food_vocabulary=VOCAB,
    )
    r = await d.detect("/x.jpg")

    assert r.ingredients == ["broccoli"]
    assert "chair" not in r.confidence_scores


async def test_yolo_only_food_is_kept_when_the_vlm_missed_it():
    """Measured case: the VLM saw apples in a fruit bowl but not the limes."""
    d = HybridIngredientDetector(
        vlm=_vlm("green apples"),
        yolo=_yolo(apple=0.68, lime=0.88, **{"fruit dish": 0.46}),
        food_vocabulary=VOCAB,
    )
    r = await d.detect("/x.jpg")

    assert "lime" in r.ingredients
    assert r.confidence_scores["lime"] == 0.88   # YOLO's real score, unboosted
    assert "lime" not in r.confirmed             # only YOLO saw it
    assert "fruit dish" not in r.ingredients     # "dish" is not an ingredient


async def test_tableware_is_rejected_even_though_the_csv_contains_the_word():
    """"cup" and "plate" occur inside real food names ("custard cup")."""
    d = HybridIngredientDetector(
        vlm=_vlm("soup"), yolo=_yolo(cup=0.9, plate=0.8), food_vocabulary=VOCAB,
    )
    r = await d.detect("/x.jpg")

    assert r.ingredients == ["soup"]


async def test_without_a_vocabulary_yolo_only_items_are_dropped():
    """Fail safe: no vocabulary must not mean 'emit everything YOLO saw'."""
    d = HybridIngredientDetector(vlm=_vlm("broccoli"), yolo=_yolo(broccoli=0.8, chair=0.9))
    r = await d.detect("/x.jpg")

    assert r.ingredients == ["broccoli"]


# ── fail-safe ──────────────────────────────────────────────────────────────

async def test_yolo_failure_falls_back_to_vlm():
    d = HybridIngredientDetector(
        vlm=_vlm("chicken", "garlic"),
        yolo=_Stub(error=IngredientDetectionError("weights missing")),
    )
    r = await d.detect("/x.jpg")

    assert r.ingredients == ["chicken", "garlic"]


async def test_vlm_failure_falls_back_to_yolo():
    """Ollama being down must not kill image analysis when YOLO is local."""
    d = HybridIngredientDetector(
        vlm=_Stub(error=IngredientDetectionError("ollama unreachable")),
        yolo=_yolo(broccoli=0.8),
        food_vocabulary=VOCAB,
    )
    r = await d.detect("/x.jpg")

    assert r.ingredients == ["broccoli"]
    assert r.confidence_scores["broccoli"] == 0.8


async def test_vlm_failure_still_filters_scenery():
    """This is the branch that runs when Ollama is down — it must not ship
    "still life, dining table, person" into the recipe query."""
    d = HybridIngredientDetector(
        vlm=_Stub(error=IngredientDetectionError("ollama unreachable")),
        yolo=_yolo(broccoli=0.8, chair=0.9, **{"still life": 0.7}),
        food_vocabulary=VOCAB,
    )
    r = await d.detect("/x.jpg")

    assert r.ingredients == ["broccoli"]
    assert "chair" not in r.confidence_scores
    assert "still life" not in r.confidence_scores


async def test_both_failing_raises():
    d = HybridIngredientDetector(
        vlm=_Stub(error=IngredientDetectionError("down")),
        yolo=_Stub(error=IngredientDetectionError("down")),
    )
    with pytest.raises(IngredientDetectionError, match="Both detectors failed"):
        await d.detect("/x.jpg")


# ── evidence is not counted twice ──────────────────────────────────────────

async def test_one_yolo_box_confirms_only_one_ingredient():
    """A single "chicken" detection cannot confirm breast AND thigh."""
    d = HybridIngredientDetector(
        vlm=_vlm("chicken breast", "chicken thigh"),
        yolo=_yolo(chicken=0.7),
        food_vocabulary=VOCAB,
    )
    r = await d.detect("/x.jpg")

    assert len(r.confirmed) == 1
    assert r.confidence_scores["chicken thigh"] == 0.5   # the unconfirmed one


async def test_a_repeated_vlm_name_is_not_confirmed_twice():
    d = HybridIngredientDetector(
        vlm=_vlm("tomato", "tomato"), yolo=_yolo(tomato=0.6), food_vocabulary=VOCAB,
    )
    r = await d.detect("/x.jpg")

    assert r.ingredients == ["tomato"]
    assert r.confirmed == ["tomato"]
