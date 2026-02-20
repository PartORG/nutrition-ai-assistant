"""
domain.entities - Persistence-aware types (have IDs, timestamps).

Migrated from database/models.py. These are the same dataclasses but
decoupled from any persistence strategy — no SQL concerns, no DB imports.

Timestamps are set by the repository implementations, not by the entities
themselves. The _now() helper from the old models.py is removed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class User:
    """Core user entity."""
    id: Optional[int] = None
    name: str = ""
    surname: str = ""
    user_name: str = ""
    caretaker: str = ""
    age: int = 0
    gender: str = ""
    created_at: str = ""
    updated_at: str = ""
    deleted_at: str = ""


@dataclass
class MedicalAdvice:
    """Medical dietary advice linked to a user."""
    id: Optional[int] = None
    health_condition: str = ""
    medical_advice: str = ""
    dietary_limit: str = ""
    avoid: str = ""
    dietary_constraints: str = ""
    user_id: Optional[int] = None
    created_at: str = ""
    updated_at: str = ""
    deleted_at: str = ""


@dataclass
class Authentication:
    """Login credentials for a user."""
    id: Optional[int] = None
    login: str = ""
    password: str = ""
    role: str = ""
    user_id: Optional[int] = None
    created_at: str = ""
    updated_at: str = ""
    deleted_at: str = ""


@dataclass
class UserProfileHistory:
    """Historical snapshot of user dietary profile."""
    id: Optional[int] = None
    preferences: str = ""
    health_condition: str = ""
    restrictions: str = ""
    user_id: Optional[int] = None
    created_at: str = ""
    updated_at: str = ""
    deleted_at: str = ""


@dataclass
class RecipeHistory:
    """Record of a recipe recommendation/selection by a user."""
    id: Optional[int] = None
    user_id: Optional[int] = None
    recipe_id: Optional[int] = None
    rating: Optional[int] = None
    recipe_name: str = ""
    servings: int = 0
    ingredients: str = ""
    cook_instructions: str = ""
    prep_time: str = ""
    created_at: str = ""
    updated_at: str = ""
    deleted_at: str = ""


@dataclass
class NutritionHistory:
    """Nutritional facts for a saved recipe."""
    id: Optional[int] = None
    user_id: Optional[int] = None
    recipe_id: Optional[int] = None
    calories: float = 0.0
    protein: float = 0.0
    fat: float = 0.0
    carbohydrates: float = 0.0
    fiber: float = 0.0
    sugar: float = 0.0
    sodium: float = 0.0
    created_at: str = ""
    updated_at: str = ""
    deleted_at: str = ""


@dataclass
class Conversation:
    """Metadata for a conversation session."""
    id: Optional[int] = None
    user_id: Optional[int] = None
    conversation_id: str = ""
    title: str = ""
    last_message_at: str = ""
    created_at: str = ""
    updated_at: str = ""
    deleted_at: str = ""


@dataclass
class ChatMessage:
    """A single message in a conversation."""
    id: Optional[int] = None
    user_id: Optional[int] = None
    conversation_id: str = ""
    role: str = ""  # "user" or "assistant"
    content: str = ""
    created_at: str = ""
    updated_at: str = ""
    deleted_at: str = ""
