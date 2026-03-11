"""
Unit tests for the RULE-BASED (pure Python) methods in components/safety_filter.py.

No LLM is invoked.  The module-level LangChain imports are stubbed out in
conftest.py, so these tests run without any ML libraries installed.

Covers:
- SafetyFilter._word_match()           — word-boundary regex helper
- SafetyFilter._check_ingredients()   — avoid-list + dietary restriction checks
- SafetyFilter._check_nutrition()     — numeric nutrition limit checks
- SafetyFilter._build_safe_markdown() — markdown filtering by recipe name
"""

from __future__ import annotations

import pytest

# conftest.py (in this directory) has already stubbed heavy imports.
from components.safety_filter import (
    SafetyFilter,
    NutritionValues,
    ParsedRecipe,
    SafetyVerdict,
    SafetyIssue,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_filter() -> SafetyFilter:
    """Instantiate SafetyFilter with all LLM calls stubbed."""
    # OllamaLLM and JsonOutputParser are MagicMocks (see conftest.py),
    # so __init__ succeeds and _build_*_chain() returns a MagicMock chain.
    return SafetyFilter(model_name="stub", debug=False)


def make_nutrition(**kwargs) -> NutritionValues:
    return NutritionValues(**kwargs)


def make_recipe(name: str, ingredients: list[str], **nutrition_kwargs) -> ParsedRecipe:
    return ParsedRecipe(name=name, ingredients=ingredients, nutrition=make_nutrition(**nutrition_kwargs))


# ---------------------------------------------------------------------------
# _word_match
# ---------------------------------------------------------------------------

class TestWordMatch:
    def test_exact_word_matches(self):
        assert SafetyFilter._word_match("salt", "sea salt") is True

    def test_word_in_multi_word_string(self):
        assert SafetyFilter._word_match("peanut", "peanut butter") is True

    def test_substring_does_not_match(self):
        # "salt" must not match "salted" as a substring
        assert SafetyFilter._word_match("salt", "salted butter") is False

    def test_case_insensitive(self):
        assert SafetyFilter._word_match("Sugar", "raw sugar") is True

    def test_no_match_returns_false(self):
        assert SafetyFilter._word_match("wheat", "rice flour") is False

    def test_empty_text_returns_false(self):
        assert SafetyFilter._word_match("sugar", "") is False


# ---------------------------------------------------------------------------
# _check_ingredients
# ---------------------------------------------------------------------------

class TestCheckIngredients:
    def setup_method(self):
        self.sf = make_filter()

    def test_avoid_food_match_is_critical(self):
        issues = self.sf._check_ingredients(
            ingredients=["peanut butter"],
            avoid_foods=["peanut"],
            restrictions=[],
        )
        assert len(issues) == 1
        assert issues[0].severity == "critical"
        assert issues[0].category == "avoid_food"

    def test_multi_word_avoid_phrase_skipped(self):
        # Vague phrases with more than 2 words are skipped
        issues = self.sf._check_ingredients(
            ingredients=["sugar"],
            avoid_foods=["high sugar foods and candy"],
            restrictions=[],
        )
        assert issues == []

    def test_vegetarian_restriction_blocks_chicken(self):
        issues = self.sf._check_ingredients(
            ingredients=["chicken breast"],
            avoid_foods=[],
            restrictions=["vegetarian"],
        )
        assert any(i.category == "restriction_violation" for i in issues)
        assert any(i.severity == "critical" for i in issues)

    def test_vegan_restriction_blocks_dairy(self):
        issues = self.sf._check_ingredients(
            ingredients=["cheddar cheese"],
            avoid_foods=[],
            restrictions=["vegan"],
        )
        assert any(i.category == "restriction_violation" for i in issues)

    def test_gluten_free_blocks_wheat_flour(self):
        issues = self.sf._check_ingredients(
            ingredients=["wheat flour"],
            avoid_foods=[],
            restrictions=["gluten-free"],
        )
        assert any(i.category == "restriction_violation" for i in issues)

    def test_no_issues_for_safe_ingredients(self):
        issues = self.sf._check_ingredients(
            ingredients=["broccoli", "olive oil", "lemon juice"],
            avoid_foods=["peanut"],
            restrictions=["vegetarian"],
        )
        assert issues == []

    def test_unknown_restriction_produces_no_issues(self):
        issues = self.sf._check_ingredients(
            ingredients=["salmon"],
            avoid_foods=[],
            restrictions=["unknown-diet"],
        )
        assert issues == []


# ---------------------------------------------------------------------------
# _check_nutrition
# ---------------------------------------------------------------------------

class TestCheckNutrition:
    def setup_method(self):
        self.sf = make_filter()

    def test_exceeding_max_creates_medium_issue(self):
        nv = make_nutrition(sugar_g=40.0)
        issues = self.sf._check_nutrition(nv, {"sugar_g": {"max": 25.0}})
        assert len(issues) == 1
        assert issues[0].severity == "medium"
        assert issues[0].category == "nutrition_limit"
        assert "exceeds" in issues[0].description

    def test_below_min_creates_medium_issue(self):
        nv = make_nutrition(fiber_g=5.0)
        issues = self.sf._check_nutrition(nv, {"fiber_g": {"min": 10.0}})
        assert len(issues) == 1
        assert "below minimum" in issues[0].description

    def test_within_limits_produces_no_issues(self):
        nv = make_nutrition(sugar_g=20.0, sodium_mg=400.0)
        issues = self.sf._check_nutrition(
            nv, {"sugar_g": {"max": 25.0}, "sodium_mg": {"max": 600.0}}
        )
        assert issues == []

    def test_none_value_skipped(self):
        # sugar_g is None → no constraint check
        nv = make_nutrition(sugar_g=None)
        issues = self.sf._check_nutrition(nv, {"sugar_g": {"max": 25.0}})
        assert issues == []

    def test_none_max_or_min_skipped(self):
        nv = make_nutrition(sodium_mg=1000.0)
        issues = self.sf._check_nutrition(nv, {"sodium_mg": {"max": None}})
        assert issues == []

    def test_multiple_violations_all_reported(self):
        nv = make_nutrition(sugar_g=50.0, sodium_mg=900.0)
        rules = {"sugar_g": {"max": 25.0}, "sodium_mg": {"max": 600.0}}
        issues = self.sf._check_nutrition(nv, rules)
        assert len(issues) == 2

    def test_unknown_nutrient_key_ignored(self):
        nv = make_nutrition()
        issues = self.sf._check_nutrition(nv, {"nonexistent_nutrient": {"max": 10.0}})
        assert issues == []


# ---------------------------------------------------------------------------
# _build_safe_markdown
# ---------------------------------------------------------------------------

class TestBuildSafeMarkdown:
    MARKDOWN_WITH_SEPARATOR = """\
**1. Grilled Salmon**

Rich in omega-3.

---

**2. Peanut Butter Smoothie**

Quick breakfast.

---

**3. Steamed Broccoli**

Light and healthy.
"""

    def test_keeps_only_named_sections(self):
        result = SafetyFilter._build_safe_markdown(
            self.MARKDOWN_WITH_SEPARATOR, ["Grilled Salmon", "Steamed Broccoli"]
        )
        assert "Grilled Salmon" in result
        assert "Steamed Broccoli" in result
        assert "Peanut Butter Smoothie" not in result

    def test_empty_safe_names_returns_empty_string(self):
        result = SafetyFilter._build_safe_markdown(self.MARKDOWN_WITH_SEPARATOR, [])
        assert result == ""

    def test_all_safe_keeps_all_sections(self):
        all_names = ["Grilled Salmon", "Peanut Butter Smoothie", "Steamed Broccoli"]
        result = SafetyFilter._build_safe_markdown(self.MARKDOWN_WITH_SEPARATOR, all_names)
        for name in all_names:
            assert name in result

    def test_fallback_when_no_separator_found(self):
        # If no --- separator, returns original markdown (fallback)
        plain = "Some recipe text without separators\nAnother recipe"
        result = SafetyFilter._build_safe_markdown(plain, ["safe recipe"])
        # Fallback: original text returned since no sections can be matched
        assert result == plain

    def test_case_insensitive_name_matching(self):
        result = SafetyFilter._build_safe_markdown(
            self.MARKDOWN_WITH_SEPARATOR, ["grilled salmon"]
        )
        assert "Grilled Salmon" in result
