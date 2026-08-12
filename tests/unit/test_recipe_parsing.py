"""
Characterization tests for the recipe JSON parser.

These pin down what `_parse_json_to_recipes` does *today*, including the parts
that are wrong. Phase 4 replaces the regex extraction with
`with_structured_output`; these tests are the net that makes that swap safe.

Tests marked KNOWN-BAD document defects deliberately. When Phase 4 fixes them
the assertion flips — that is the signal the migration worked, not a
regression.
"""

from __future__ import annotations

import json

from infrastructure.rag.recipe_rag import RecipeNutritionRAG

parse = RecipeNutritionRAG._parse_json_to_recipes


def _payload(**overrides) -> str:
    recipe = {
        "name": "Lemon Chicken",
        "ingredients": ["chicken breast", "lemon", "olive oil"],
        "nutrition": {"calories": 420, "protein_g": 38, "carbs_g": 12, "fat_g": 22},
        "why_recommended": "High protein, low carb.",
        "servings": 2,
        "prep_time": "20 min",
        "cook_instructions": "Roast at 200C for 25 minutes.",
    }
    recipe.update(overrides)
    return json.dumps({"recipes": [recipe]})


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_parses_a_well_formed_recipe():
    [r] = parse(_payload())

    assert r.name == "Lemon Chicken"
    assert r.ingredients == ["chicken breast", "lemon", "olive oil"]
    assert r.nutrition.calories == 420
    assert r.nutrition.protein_g == 38
    assert r.servings == 2
    assert r.cook_instructions.startswith("Roast")


def test_tolerates_prose_around_the_json():
    """Small local models routinely wrap JSON in chatter."""
    noisy = f"Sure! Here are some recipes:\n\n{_payload()}\n\nEnjoy!"
    [r] = parse(noisy)
    assert r.name == "Lemon Chicken"


def test_strips_markdown_code_fences():
    fenced = f"```json\n{_payload()}\n```"
    [r] = parse(fenced)
    assert r.name == "Lemon Chicken"


# ---------------------------------------------------------------------------
# Shape coercions that exist because models return the wrong types
# ---------------------------------------------------------------------------

def test_ingredients_given_as_objects_are_flattened():
    """Models often emit {"name": ..., "amount": ...} instead of a string."""
    [r] = parse(_payload(ingredients=[{"name": "chicken", "amount": "200g"}]))
    assert len(r.ingredients) == 1
    assert "chicken" in r.ingredients[0]
    assert "200g" in r.ingredients[0]


def test_instructions_given_as_a_list_are_joined():
    [r] = parse(_payload(cook_instructions=["Preheat oven.", "Roast 25 min."]))
    assert "Preheat oven." in r.cook_instructions
    assert "Roast 25 min." in r.cook_instructions


def test_missing_nutrition_block_yields_empty_values_not_a_crash():
    [r] = parse(_payload(nutrition=None))
    assert r.nutrition.calories is None
    assert r.name == "Lemon Chicken"


def test_servings_defaults_to_one_when_absent_or_null():
    [r] = parse(_payload(servings=None))
    assert r.servings == 1


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------

def test_no_json_at_all_returns_empty():
    assert parse("I'm sorry, I can't help with that.") == []


def test_malformed_json_returns_empty():
    assert parse('{"recipes": [ {"name": "Broken" ') == []


def test_truncated_output_is_indistinguishable_from_no_recipes():
    """KNOWN-BAD: a cut-off response silently becomes 'No recipes found'.

    A model hitting its token limit mid-object produces unparseable JSON, and
    the caller cannot tell that apart from a legitimately empty result. Phase 4
    should surface this as an error instead. Flip this assertion then.
    """
    truncated = _payload()[: len(_payload()) // 2]
    assert parse(truncated) == []


def test_greedy_regex_spans_two_json_objects():
    """KNOWN-BAD: the extractor takes first '{' to last '}'.

    Given two separate objects it splices them into one malformed blob rather
    than parsing the first, so a chatty model loses everything.
    """
    two_objects = f"{_payload()}\nAnd another:\n{_payload()}"
    assert parse(two_objects) == []
