"""
application.dto - Data Transfer Objects for service input/output.

These are the structured results that services return to callers
(agent tools, REST endpoints, CLI adapters).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from domain.models import (
    UserIntent,
    NutritionConstraints,
    SafetyCheckResult,
    Recipe,
    DetectedIngredients,
)


@dataclass(frozen=True)
class RecommendationResult:
    """Complete result from the recommendation pipeline."""
    intent: UserIntent
    constraints: NutritionConstraints
    augmented_query: str
    raw_recommendations: list[Recipe]
    safety_result: SafetyCheckResult

    @property
    def safe_recipes(self) -> list[Recipe]:
        """Convenience — safe + warning recipes from safety result."""
        return self.safety_result.safe_recipes

    @property
    def summary(self) -> str:
        return self.safety_result.summary


@dataclass(frozen=True)
class RegisterRequest:
    """Input for user registration."""
    login: str
    password: str
    name: str = ""
    surname: str = ""
    age: int = 0
    gender: str = ""
    caretaker: str = ""
    health_condition: str = ""


@dataclass(frozen=True)
class LoginRequest:
    """Input for user login."""
    login: str
    password: str


@dataclass(frozen=True)
class AuthToken:
    """JWT token response after successful register/login."""
    access_token: str
    token_type: str = "bearer"
    user_id: int = 0
    role: str = "user"


@dataclass(frozen=True)
class ImageAnalysisResult:
    """Result from CNN ingredient detection, optionally with recommendations."""
    detected: DetectedIngredients
    recommendation: Optional[RecommendationResult] = None
