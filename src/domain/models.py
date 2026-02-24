"""
domain.models - Value objects for the AI pipeline.

These are immutable data containers with no business logic and no
dependencies on infrastructure (no LangChain, no Ollama, no SQLite).

Migrated from:
    - components/intent_retriever.py  → UserIntent
    - components/safety_filter.py     → SafetyVerdict, NutritionValues, Recipe,
                                        SafetyIssue, RecipeSafetyResult, SafetyCheckResult
    - (new)                           → NutritionConstraints, DetectedIngredients
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# User Intent
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class UserIntent:
    """Structured representation of parsed user intent.

    All collection fields use list[str] instead of comma-separated strings.
    Consumers no longer need to call .split(",") — the domain model stores
    the actual data type.
    """
    name: str = ""
    surname: str = ""
    preferences: list[str] = field(default_factory=list)
    restrictions: list[str] = field(default_factory=list)
    health_conditions: list[str] = field(default_factory=list)
    instructions: list[str] = field(default_factory=list)
    caretaker: str = ""

    def __repr__(self) -> str:
        fields = {
            "name": self.name,
            "surname": self.surname,
            "preferences": self.preferences,
            "restrictions": self.restrictions,
            "health_conditions": self.health_conditions,
            "instructions": self.instructions,
            "caretaker": self.caretaker,
        }
        lines = [f"  {k}: {v}" for k, v in fields.items() if v]
        return "UserIntent:\n" + "\n".join(lines) if lines else "UserIntent: (empty)"


# ---------------------------------------------------------------------------
# Nutrition Constraints (from Medical RAG)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NutritionConstraints:
    """Typed replacement for the raw dict returned by MedicalRAG.get_constraints().

    constraints dict maps nutrient names to {"max": float|None, "min": float|None}.
    Example: {"sugar_g": {"max": 25}, "fiber_g": {"min": 30}}
    """
    dietary_goals: list[str] = field(default_factory=list)
    foods_to_increase: list[str] = field(default_factory=list)
    avoid: list[str] = field(default_factory=list)
    limit: list[str] = field(default_factory=list)
    constraints: dict[str, dict[str, Optional[float]]] = field(default_factory=dict)
    notes: str = ""

    @classmethod
    def default(cls) -> NutritionConstraints:
        """Return default constraints when no medical conditions are provided."""
        return cls(
            dietary_goals=["General healthy eating guidelines"],
            foods_to_increase=["whole grains", "vegetables", "fruits", "lean proteins"],
            avoid=[],
            limit=[],
            constraints={
                "sugar_g": {"max": None},
                "sodium_mg": {"max": None},
                "fiber_g": {"min": None},
                "protein_g": {"max": None},
                "saturated_fat_g": {"max": None},
            },
            notes="No specific medical conditions provided",
        )

    def to_dict(self) -> dict:
        """Convert to plain dict (for backward compatibility during migration)."""
        return {
            "dietary_goals": self.dietary_goals,
            "foods_to_increase": self.foods_to_increase,
            "avoid": self.avoid,
            "limit": self.limit,
            "constraints": self.constraints,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Nutrition Values
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NutritionValues:
    """Parsed nutrition values from a recipe (per serving)."""
    calories: Optional[float] = None
    protein_g: Optional[float] = None
    carbs_g: Optional[float] = None
    fat_g: Optional[float] = None
    fiber_g: Optional[float] = None
    sodium_mg: Optional[float] = None
    sugar_g: Optional[float] = None
    saturated_fat_g: Optional[float] = None


# ---------------------------------------------------------------------------
# Recipe
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Recipe:
    """A single recipe with all structured data.

    Renamed from ParsedRecipe — this is the domain concept.
    """
    name: str = ""
    ingredients: list[str] = field(default_factory=list)
    nutrition: NutritionValues = field(default_factory=NutritionValues)
    why_recommended: str = ""
    servings: int = 0
    prep_time: str = ""
    cook_instructions: str = ""
    rating: Optional[int] = None    # 1-5 user rating, if available


# ---------------------------------------------------------------------------
# Safety
# ---------------------------------------------------------------------------

class SafetyVerdict(str, Enum):
    """Overall safety classification for a recipe."""
    SAFE = "safe"
    WARNING = "warning"
    UNSAFE = "unsafe"


@dataclass(frozen=True)
class SafetyIssue:
    """A single safety concern found during recipe checking.

    Categories: "allergen", "avoid_food", "nutrition_limit",
                "restriction_violation", "hidden_ingredient"
    Severities: "critical", "high", "medium"
    """
    category: str
    severity: str
    description: str
    detail: str = ""


@dataclass(frozen=True)
class RecipeSafetyResult:
    """Safety check result for one recipe."""
    recipe_name: str
    verdict: SafetyVerdict
    issues: list[SafetyIssue] = field(default_factory=list)
    recipe: Optional[Recipe] = None

    @property
    def is_safe(self) -> bool:
        return self.verdict == SafetyVerdict.SAFE


@dataclass
class SafetyCheckResult:
    """Aggregate result for all recipes in the output.

    recipe_verdicts: Per-recipe safety verdicts.
    safe_recipes_markdown: Filtered markdown containing only safe/warning recipes.
    summary: Human-readable summary of the check.
    """
    recipe_verdicts: list[RecipeSafetyResult] = field(default_factory=list)
    safe_recipes_markdown: str = ""
    summary: str = ""

    @property
    def safe_count(self) -> int:
        return sum(
            1 for v in self.recipe_verdicts
            if v.is_safe or v.verdict == SafetyVerdict.WARNING
        )

    @property
    def total_count(self) -> int:
        return len(self.recipe_verdicts)

    @property
    def safe_recipes(self) -> list[Recipe]:
        """Return Recipe objects for all safe/warning recipes."""
        return [
            v.recipe for v in self.recipe_verdicts
            if v.recipe is not None
            and v.verdict != SafetyVerdict.UNSAFE
        ]

    @property
    def filtered_out(self) -> list[RecipeSafetyResult]:
        """Return verdicts for recipes that were filtered out (UNSAFE)."""
        return [
            v for v in self.recipe_verdicts
            if v.verdict == SafetyVerdict.UNSAFE
        ]

    @property
    def warnings(self) -> list[RecipeSafetyResult]:
        """Return verdicts for recipes that have warnings."""
        return [
            v for v in self.recipe_verdicts
            if v.verdict == SafetyVerdict.WARNING
        ]


# ---------------------------------------------------------------------------
# CNN / Image Detection
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DetectedIngredients:
    """Result of CNN-based ingredient detection from an image."""
    ingredients: list[str] = field(default_factory=list)
    confidence_scores: dict[str, float] = field(default_factory=dict)
    image_path: str = ""
    source: str = ""  # Which detector produced this result: "YOLO", "LLaVA", etc.
