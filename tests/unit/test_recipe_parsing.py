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


def test_truncated_output_yields_no_recipes():
    """A cut-off response is unrecoverable, so it parses to nothing.

    Was marked KNOWN-BAD in Phase 1 because the caller cannot distinguish this
    from a legitimately empty result. It stays [] — an unbalanced object cannot
    be repaired — but this path is now only a fallback: the chain binds
    _RecipeListSchema, so the provider enforces the shape and a truncated
    response fails validation upstream instead.
    """
    truncated = _payload()[: len(_payload()) // 2]
    assert parse(truncated) == []


def test_two_json_objects_parse_the_first_not_a_splice():
    """FIXED in Phase 4 (was KNOWN-BAD).

    The old extractor took the first '{' to the *last* '}', splicing two
    objects into one malformed blob and losing both. Brace-depth scanning
    returns the first complete object instead.
    """
    two_objects = f"{_payload()}\nAnd another:\n{_payload()}"
    [recipe] = parse(two_objects)
    assert recipe.name == "Lemon Chicken"


def test_braces_inside_strings_do_not_end_the_object():
    """A '}' inside a string value must not terminate extraction early."""
    payload = _payload(why_recommended="Uses a {special} technique")
    [recipe] = parse(payload)
    assert recipe.why_recommended == "Uses a {special} technique"
