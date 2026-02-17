"""
Data models for the nutrition-ai-assistant database.

Each dataclass maps 1:1 to a SQLite table. Timestamp fields (created_at,
updated_at, deleted_at) have sensible defaults so callers only need to
supply the domain-specific columns.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


def _now() -> str:
    """Return the current local time as an ISO-8601 string."""
    return datetime.now().isoformat()


@dataclass
class User:
    """Represents a row in the 'users' table.

    Stores basic demographic and identity information for a user.
    The 'user_name' column has a UNIQUE constraint in the DB schema.
    """
    name: str
    surname: str
    user_name: str
    caretaker: str
    age: int
    gender: str
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    deleted_at: str = ""
    id: Optional[int] = None


@dataclass
class MedicalAdvice:
    """Represents a row in the 'medical_advice' table.

    Stores a single piece of medical/dietary guidance tied to a user's
    health condition. Linked to a user via user_id (foreign key).
    """
    health_condition: str
    medical_advice: str
    limit: str
    avoid: str
    constraints: str
    user_id: int
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    deleted_at: str = ""
    id: Optional[int] = None


@dataclass
class Authentication:
    """Represents a row in the 'authentication' table.

    Stores login credentials and role for a user.
    The 'login' column has a UNIQUE constraint in the DB schema.
    """
    login: str
    password: str
    role: str
    user_id: int
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    deleted_at: str = ""
    id: Optional[int] = None


@dataclass
class UserProfileHistory:
    """Represents a row in the 'user_profile_history' table.

    Tracks a snapshot of a user's dietary preferences, health conditions,
    and food restrictions at a point in time. Each update creates a new
    row rather than modifying an existing one.
    """
    preferences: str
    user_id: int
    health_condition: str
    restrictions: str
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    deleted_at: str = ""
    id: Optional[int] = None


@dataclass
class RecipeHistory:
    """Represents a row in the 'recipe_history' table.

    Records that a user was recommended or used a specific recipe,
    along with serving details and ingredient information.
    """
    user_id: int
    recipe_id: int
    servings: int
    ingredients: str
    prep_time: str
    cook_instructions: str
    recipe_name: str
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    deleted_at: str = ""
    id: Optional[int] = None


@dataclass
class NutritionHistory:
    """Represents a row in the 'nutrition_history' table.

    Stores the nutritional breakdown of a specific recipe for a user,
    enabling historical tracking of nutrient intake.
    """
    user_id: int
    recipe_id: int
    calories: float
    protein: float
    fat: float
    carbohydrates: float
    fiber: float
    sugar: float
    sodium: float
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    deleted_at: str = ""
    id: Optional[int] = None
