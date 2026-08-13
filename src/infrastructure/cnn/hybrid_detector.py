"""
infrastructure.cnn.hybrid_detector - VLM + YOLO cross-validated detection.

Runs both detectors on the same image concurrently and reconciles them.
Neither is good enough alone:

    VLM     open vocabulary and food-literate ("browned beef", "olive oil"),
            but reports no calibrated confidence and will happily hallucinate
            a plausible ingredient that is not in the picture.
    YOLO    genuine per-box confidence and it either saw a thing or it did
            not, but its vocabulary is fixed. Even open-vocab YOLOE emits
            scene nouns ("still life", "chair", "embellishment") alongside
            the food.

So the VLM proposes the ingredient list and YOLO grades it. An ingredient
both detectors see is reported with YOLO's real score, boosted for the
agreement; one only the VLM saw is reported as unverified. That turns the
old fabricated flat 0.95 into something that actually varies with evidence.

Fail-safe: if either detector errors, the other's result is returned on its
own rather than failing the request. Only a double failure raises.
"""

from __future__ import annotations

import asyncio
import csv
import logging
import re
from dataclasses import replace
from pathlib import Path

from domain.exceptions import IngredientDetectionError
from domain.models import DetectedIngredients

logger = logging.getLogger(__name__)

# Confidence for an ingredient only the VLM saw. Deliberately mid-range: the
# VLM gives no calibrated score, so anything higher would be invented.
_UNVERIFIED_CONFIDENCE = 0.5

# Boost applied to YOLO's real score when the VLM independently agrees.
_AGREEMENT_BOOST = 0.25

# Words that carry no food identity, dropped before matching so that
# "green beans" and "green pepper" do not match on "green".
_MODIFIERS = frozenset({
    "fresh", "raw", "cooked", "grilled", "roasted", "fried", "baked", "steamed",
    "chopped", "sliced", "diced", "minced", "whole", "ground", "dried", "frozen",
    "green", "red", "yellow", "white", "black", "brown", "purple", "orange",
    "large", "small", "medium", "baby", "young", "hot", "cold", "sweet",
    "and", "with", "of", "the", "a", "some", "few",
})


def _tokens(name: str) -> frozenset[str]:
    """Content words of an ingredient name, crudely singularised.

    "Broccoli florets" -> {broccoli, floret}; "chicken breasts" -> {chicken, breast}.
    Matching on tokens rather than substrings keeps "apple" from matching
    "pineapple".
    """
    words = re.sub(r"[^a-z]+", " ", name.lower()).split()
    return frozenset(
        w[:-1] if w.endswith("s") and len(w) > 3 else w
        for w in words
        if w not in _MODIFIERS and len(w) > 2
    )


def _agrees(a: str, b: str) -> bool:
    """True if two ingredient names plausibly refer to the same food.

    Tokens match on a shared prefix rather than equality, so the crude
    singulariser above ("tomatoes" -> "tomatoe") still meets "tomato". The
    4-character floor keeps "apple" from matching "pineapple", and the
    1-character length budget keeps the prefix rule from pairing genuinely
    different foods: butter/butternut, pepper/peppermint, corn/corned,
    cream/creamer, milk/milkshake all differ by more than one character.
    """
    ta, tb = _tokens(a), _tokens(b)
    return any(
        x == y
        or (
            len(x) >= 4 and len(y) >= 4
            and abs(len(x) - len(y)) <= 1
            and (x.startswith(y) or y.startswith(x))
        )
        for x in ta
        for y in tb
    )


# Tableware and scene nouns that appear in the food CSV as part of longer
# names ("custard cup") and would otherwise let a detected plate through.
_NOT_INGREDIENTS = frozenset({
    "plate", "cup", "bowl", "spoon", "fork", "knife", "table", "chair",
    "glass", "dish", "napkin", "tray", "pan", "pot", "jar", "bottle", "can",
    # Category nouns: real YOLOE labels, and every word of them appears in
    # the food CSV, so only an explicit denial keeps them out.
    "food", "meal", "snack", "dessert", "vegetable", "drink", "beverage",
    "plant", "recipe", "cuisine", "ingredient", "produce", "dinner", "lunch",
    "breakfast",
})


def load_food_vocabulary(csv_path: str | Path) -> frozenset[str]:
    """Every distinct word appearing in a food name in the nutrition CSV.

    Used to decide whether an open-vocabulary YOLO detection is a food at
    all. 8.8k food names yield ~3.4k words, which cleanly separates
    "lime"/"broccoli" from "still life"/"embellishment"/"darkness".
    """
    words: set[str] = set()
    try:
        with open(csv_path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                name = (row.get("name") or "").lower()
                words.update(re.sub(r"[^a-z]+", " ", name).split())
    except OSError:
        logger.warning("Food vocabulary unavailable (%s); YOLO-only detections will be dropped", csv_path)
        return frozenset()
    return frozenset(w for w in words if len(w) > 2)


class HybridIngredientDetector:
    """Cross-validates a VLM's ingredient list against YOLO detections."""

    def __init__(self, vlm, yolo, food_vocabulary: frozenset[str] = frozenset()):
        self._vlm = vlm
        self._yolo = yolo
        self._food_vocabulary = food_vocabulary

    def _is_food(self, label: str) -> bool:
        """True if every content word of `label` names a food.

        Requiring *all* tokens is what rejects "fruit dish" and "western
        food" while keeping "chicken breast". With no vocabulary loaded
        nothing qualifies, so YOLO-only detections are simply dropped.
        """
        tokens = _tokens(label)
        if not tokens or not self._food_vocabulary:
            return False
        if any(t in _NOT_INGREDIENTS for t in tokens):
            return False
        return all(
            t in self._food_vocabulary or t.rstrip("e") in self._food_vocabulary
            for t in tokens
        )

    async def detect(self, image_path: str) -> DetectedIngredients:
        """Detect ingredients with both backends. Implements IngredientDetectorPort."""
        vlm_result, yolo_result = await asyncio.gather(
            self._vlm.detect(image_path),
            self._yolo.detect(image_path),
            return_exceptions=True,
        )

        vlm_ok = isinstance(vlm_result, DetectedIngredients)
        yolo_ok = isinstance(yolo_result, DetectedIngredients)

        if not vlm_ok and not yolo_ok:
            raise IngredientDetectionError(
                f"Both detectors failed. VLM: {vlm_result}. YOLO: {yolo_result}."
            )

        # ── Fail-safe routes: one backend down, use the other verbatim ──
        if not yolo_ok:
            logger.warning("YOLO failed (%s) — falling back to VLM alone", yolo_result)
            return vlm_result
        if not vlm_ok:
            # Still filter: this is the branch that runs when Ollama is down,
            # and raw YOLOE output is mostly scenery. Returning it verbatim
            # sent "still life, dining table, person" into the recipe query.
            logger.warning("VLM failed (%s) — falling back to YOLO alone", vlm_result)
            kept = [y for y in yolo_result.ingredients if self._is_food(y)]
            return replace(
                yolo_result,
                ingredients=kept,
                confidence_scores={
                    k: v for k, v in yolo_result.confidence_scores.items() if k in kept
                },
            )

        return self._reconcile(vlm_result, yolo_result, image_path)

    def _reconcile(
        self,
        vlm: DetectedIngredients,
        yolo: DetectedIngredients,
        image_path: str,
    ) -> DetectedIngredients:
        """Merge two successful detections, scoring by agreement."""
        scores: dict[str, float] = {}
        confirmed: list[str] = []

        # One YOLO box is one piece of evidence: once it has confirmed an
        # ingredient it is spent, so a single "chicken" detection cannot
        # confirm both "chicken breast" and "chicken thigh".
        unclaimed = list(yolo.ingredients)
        for ingredient in vlm.ingredients:
            if ingredient in scores:      # the VLM repeated itself
                continue
            match = next((y for y in unclaimed if _agrees(ingredient, y)), None)
            if match is None:
                scores[ingredient] = _UNVERIFIED_CONFIDENCE
                continue
            unclaimed.remove(match)
            yolo_score = yolo.confidence_scores.get(match, 0.0)
            scores[ingredient] = round(min(0.99, yolo_score + _AGREEMENT_BOOST), 3)
            confirmed.append(ingredient)

        # Whatever is still unclaimed is a YOLO-only detection. It is kept
        # when unambiguously a food: the VLM does miss things (it saw the
        # apples in a fruit bowl but not the limes YOLO scored at 0.88).
        # Everything else YOLO emits is scenery.
        kept, discarded = [], []
        for label in unclaimed:
            if self._is_food(label):
                scores[label] = yolo.confidence_scores.get(label, 0.0)
                kept.append(label)
            else:
                discarded.append(label)

        logger.info(
            "Hybrid detection: %d confirmed %s | %d VLM-only %s | %d YOLO-only kept %s | %d discarded as non-food %s",
            len(confirmed), confirmed,
            len(scores) - len(confirmed) - len(kept),
            [i for i in scores if i not in confirmed and i not in kept],
            len(kept), kept,
            len(discarded), discarded,
        )

        ingredients = sorted(scores, key=lambda k: scores[k], reverse=True)
        return DetectedIngredients(
            ingredients=ingredients,
            confidence_scores=scores,
            image_path=image_path,
            source=f"{vlm.source}+{yolo.source}",
            confirmed=confirmed,
        )
