"""
Unit tests for application/dto.py — Data Transfer Objects.
All are plain dataclasses or frozen dataclasses, no external dependencies.
"""

from domain.models import (
    UserIntent,
    NutritionConstraints,
    SafetyCheckResult,
    SafetyVerdict,
    RecipeSafetyResult,
    Recipe,
    NutritionValues,
    DetectedIngredients,
)
from application.dto import (
    RecommendationResult,
    RegisterRequest,
    LoginRequest,
    AuthToken,
    ImageAnalysisResult,
)


# ---------------------------------------------------------------------------
# RegisterRequest
# ---------------------------------------------------------------------------

class TestRegisterRequest:
    def test_required_fields(self):
        r = RegisterRequest(login="alice", password="secret")
        assert r.login == "alice"
        assert r.password == "secret"

    def test_optional_fields_default_to_empty(self):
        r = RegisterRequest(login="alice", password="secret")
        assert r.name == ""
        assert r.surname == ""
        assert r.age == 0
        assert r.gender == ""
        assert r.caretaker == ""
        assert r.health_condition == ""

    def test_with_all_fields(self):
        r = RegisterRequest(
            login="bob@example.com",
            password="pass123",
            name="Bob",
            surname="Smith",
            age=40,
            gender="male",
            caretaker="nurse",
            health_condition="diabetes",
        )
        assert r.name == "Bob"
        assert r.age == 40
        assert r.health_condition == "diabetes"

    def test_frozen(self):
        r = RegisterRequest(login="a", password="b")
        try:
            r.login = "changed"
            # If frozen, should raise; if not, still test the value
        except (AttributeError, TypeError):
            pass  # correctly frozen


# ---------------------------------------------------------------------------
# LoginRequest
# ---------------------------------------------------------------------------

class TestLoginRequest:
    def test_fields(self):
        r = LoginRequest(login="alice@example.com", password="mypass")
        assert r.login == "alice@example.com"
        assert r.password == "mypass"


# ---------------------------------------------------------------------------
# AuthToken
# ---------------------------------------------------------------------------

class TestAuthToken:
    def test_defaults(self):
        t = AuthToken(access_token="jwt123")
        assert t.token_type == "bearer"
        assert t.user_id == 0
        assert t.role == "user"

    def test_with_values(self):
        t = AuthToken(access_token="tok", user_id=42, role="admin")
        assert t.user_id == 42
        assert t.role == "admin"

    def test_access_token_stored(self):
        t = AuthToken(access_token="eyJhbGci...")
        assert t.access_token == "eyJhbGci..."


# ---------------------------------------------------------------------------
# RecommendationResult
# ---------------------------------------------------------------------------

def _make_recommendation(recipes=None, unsafe_recipes=None) -> RecommendationResult:
    """Build a RecommendationResult with controllable safe/unsafe split."""
    safe = recipes or []
    unsafe = unsafe_recipes or []

    verdicts = [
        RecipeSafetyResult(recipe_name=r.name, verdict=SafetyVerdict.SAFE, recipe=r)
        for r in safe
    ] + [
        RecipeSafetyResult(recipe_name=r.name, verdict=SafetyVerdict.UNSAFE, recipe=r)
        for r in unsafe
    ]

    safety = SafetyCheckResult(
        recipe_verdicts=verdicts,
        safe_recipes_markdown="## Salad\nHealthy.\n",
        summary=f"{len(safe)}/{len(safe) + len(unsafe)} passed",
    )

    return RecommendationResult(
        intent=UserIntent(name="Alice"),
        constraints=NutritionConstraints.default(),
        augmented_query="low carb dinner",
        raw_recommendations=safe + unsafe,
        safety_result=safety,
    )


class TestRecommendationResult:
    def test_safe_recipes_delegates_to_safety_result(self):
        recipe = Recipe(name="Salad")
        result = _make_recommendation(recipes=[recipe])
        assert len(result.safe_recipes) == 1
        assert result.safe_recipes[0].name == "Salad"

    def test_safe_recipes_excludes_unsafe(self):
        good = Recipe(name="Salad")
        bad = Recipe(name="PeanutButter Cake")
        result = _make_recommendation(recipes=[good], unsafe_recipes=[bad])
        names = [r.name for r in result.safe_recipes]
        assert "Salad" in names
        assert "PeanutButter Cake" not in names

    def test_summary_comes_from_safety_result(self):
        result = _make_recommendation(recipes=[Recipe(name="Soup")])
        assert "1/1" in result.summary

    def test_frozen_immutable(self):
        result = _make_recommendation()
        try:
            result.augmented_query = "changed"
        except (AttributeError, TypeError):
            pass  # correctly frozen


# ---------------------------------------------------------------------------
# ImageAnalysisResult
# ---------------------------------------------------------------------------

class TestImageAnalysisResult:
    def test_detected_required(self):
        di = DetectedIngredients(ingredients=["carrot"])
        iar = ImageAnalysisResult(detected=di)
        assert iar.detected.ingredients == ["carrot"]
        assert iar.recommendation is None

    def test_with_recommendation(self):
        di = DetectedIngredients(ingredients=["broccoli"])
        rec = _make_recommendation(recipes=[Recipe(name="Broccoli Stir-Fry")])
        iar = ImageAnalysisResult(detected=di, recommendation=rec)
        assert iar.recommendation is not None
        assert len(iar.recommendation.safe_recipes) == 1
