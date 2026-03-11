"""
Unit tests for domain/models.py — pure value objects, no mocks needed.
"""

import pytest
from domain.models import (
    UserIntent,
    NutritionConstraints,
    NutritionValues,
    Recipe,
    SafetyVerdict,
    SafetyIssue,
    RecipeSafetyResult,
    SafetyCheckResult,
    DetectedIngredients,
)


# ---------------------------------------------------------------------------
# UserIntent
# ---------------------------------------------------------------------------

class TestUserIntent:
    def test_defaults_are_empty(self):
        intent = UserIntent()
        assert intent.name == ""
        assert intent.preferences == []
        assert intent.restrictions == []
        assert intent.health_conditions == []

    def test_repr_shows_non_empty_fields_only(self):
        intent = UserIntent(name="Alice", health_conditions=["diabetes"])
        text = repr(intent)
        assert "Alice" in text
        assert "diabetes" in text
        assert "surname" not in text  # empty fields omitted

    def test_repr_empty_intent(self):
        intent = UserIntent()
        assert repr(intent) == "UserIntent: (empty)"

    def test_frozen_immutable(self):
        intent = UserIntent(name="Bob")
        with pytest.raises((AttributeError, TypeError)):
            intent.name = "Changed"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# NutritionConstraints
# ---------------------------------------------------------------------------

class TestNutritionConstraints:
    def test_default_returns_healthy_eating_goal(self):
        nc = NutritionConstraints.default()
        assert "General healthy eating guidelines" in nc.dietary_goals
        assert nc.avoid == []
        assert nc.limit == []

    def test_default_has_expected_constraint_keys(self):
        nc = NutritionConstraints.default()
        assert "sugar_g" in nc.constraints
        assert "sodium_mg" in nc.constraints
        assert "fiber_g" in nc.constraints

    def test_to_dict_round_trip(self):
        nc = NutritionConstraints(
            dietary_goals=["low sugar"],
            avoid=["peanuts"],
            constraints={"sugar_g": {"max": 25.0}},
        )
        d = nc.to_dict()
        assert d["dietary_goals"] == ["low sugar"]
        assert d["avoid"] == ["peanuts"]
        assert d["constraints"]["sugar_g"]["max"] == 25.0

    def test_frozen_immutable(self):
        nc = NutritionConstraints.default()
        with pytest.raises((AttributeError, TypeError)):
            nc.notes = "changed"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# NutritionValues
# ---------------------------------------------------------------------------

class TestNutritionValues:
    def test_all_none_by_default(self):
        nv = NutritionValues()
        assert nv.calories is None
        assert nv.protein_g is None
        assert nv.sugar_g is None

    def test_partial_values(self):
        nv = NutritionValues(calories=350.0, protein_g=30.0)
        assert nv.calories == 350.0
        assert nv.protein_g == 30.0
        assert nv.fat_g is None


# ---------------------------------------------------------------------------
# Recipe
# ---------------------------------------------------------------------------

class TestRecipe:
    def test_defaults(self):
        r = Recipe()
        assert r.name == ""
        assert r.ingredients == []
        assert r.servings == 0
        assert r.rating is None

    def test_with_nutrition(self):
        nv = NutritionValues(calories=400.0)
        r = Recipe(name="Salad", nutrition=nv)
        assert r.nutrition.calories == 400.0


# ---------------------------------------------------------------------------
# SafetyVerdict
# ---------------------------------------------------------------------------

class TestSafetyVerdict:
    def test_string_values(self):
        assert SafetyVerdict.SAFE == "safe"
        assert SafetyVerdict.WARNING == "warning"
        assert SafetyVerdict.UNSAFE == "unsafe"


# ---------------------------------------------------------------------------
# RecipeSafetyResult
# ---------------------------------------------------------------------------

class TestRecipeSafetyResult:
    def test_is_safe_true_for_safe_verdict(self):
        result = RecipeSafetyResult(recipe_name="Salad", verdict=SafetyVerdict.SAFE)
        assert result.is_safe is True

    def test_is_safe_false_for_unsafe_verdict(self):
        result = RecipeSafetyResult(recipe_name="Risky Dish", verdict=SafetyVerdict.UNSAFE)
        assert result.is_safe is False

    def test_is_safe_false_for_warning_verdict(self):
        result = RecipeSafetyResult(recipe_name="Borderline", verdict=SafetyVerdict.WARNING)
        assert result.is_safe is False


# ---------------------------------------------------------------------------
# SafetyCheckResult computed properties
# ---------------------------------------------------------------------------

class TestSafetyCheckResult:
    def _make_result(self, verdicts):
        """Build a SafetyCheckResult from a list of (name, SafetyVerdict) tuples."""
        recipe_verdicts = [
            RecipeSafetyResult(recipe_name=name, verdict=v)
            for name, v in verdicts
        ]
        return SafetyCheckResult(recipe_verdicts=recipe_verdicts)

    def test_safe_count_includes_safe_and_warning(self):
        result = self._make_result([
            ("A", SafetyVerdict.SAFE),
            ("B", SafetyVerdict.WARNING),
            ("C", SafetyVerdict.UNSAFE),
        ])
        assert result.safe_count == 2

    def test_total_count(self):
        result = self._make_result([
            ("A", SafetyVerdict.SAFE),
            ("B", SafetyVerdict.UNSAFE),
        ])
        assert result.total_count == 2

    def test_filtered_out_returns_only_unsafe(self):
        result = self._make_result([
            ("Safe Dish", SafetyVerdict.SAFE),
            ("Bad Dish", SafetyVerdict.UNSAFE),
        ])
        assert len(result.filtered_out) == 1
        assert result.filtered_out[0].recipe_name == "Bad Dish"

    def test_warnings_returns_only_warning(self):
        result = self._make_result([
            ("Fine", SafetyVerdict.SAFE),
            ("Borderline", SafetyVerdict.WARNING),
        ])
        assert len(result.warnings) == 1
        assert result.warnings[0].recipe_name == "Borderline"

    def test_safe_recipes_returns_recipe_objects(self):
        recipe = Recipe(name="Good Salad")
        verdicts = [
            RecipeSafetyResult(recipe_name="Good Salad", verdict=SafetyVerdict.SAFE, recipe=recipe),
            RecipeSafetyResult(recipe_name="Bad Dish", verdict=SafetyVerdict.UNSAFE, recipe=Recipe(name="Bad Dish")),
        ]
        result = SafetyCheckResult(recipe_verdicts=verdicts)
        safe = result.safe_recipes
        assert len(safe) == 1
        assert safe[0].name == "Good Salad"

    def test_safe_recipes_empty_when_no_verdicts(self):
        result = SafetyCheckResult()
        assert result.safe_recipes == []
        assert result.safe_count == 0
        assert result.total_count == 0


# ---------------------------------------------------------------------------
# DetectedIngredients
# ---------------------------------------------------------------------------

class TestDetectedIngredients:
    def test_defaults(self):
        di = DetectedIngredients()
        assert di.ingredients == []
        assert di.confidence_scores == {}
        assert di.source == ""

    def test_with_data(self):
        di = DetectedIngredients(
            ingredients=["carrot", "broccoli"],
            confidence_scores={"carrot": 0.95, "broccoli": 0.88},
            source="YOLO",
        )
        assert len(di.ingredients) == 2
        assert di.confidence_scores["carrot"] == pytest.approx(0.95)
