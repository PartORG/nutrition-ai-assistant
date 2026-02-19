"""
Database package for the nutrition-ai-assistant.

Provides dataclass models for all 6 tables and the UserDBHandler
class for SQLite CRUD operations.
"""

from database.models import (
    User,
    MedicalAdvice,
    Authentication,
    UserProfileHistory,
    RecipeHistory,
    NutritionHistory,
)
from database.db import UserDBHandler

__all__ = [
    "User",
    "MedicalAdvice",
    "Authentication",
    "UserProfileHistory",
    "RecipeHistory",
    "NutritionHistory",
    "UserDBHandler",
]
