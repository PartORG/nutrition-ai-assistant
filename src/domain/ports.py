"""
domain.ports - Abstract interfaces (Protocols) for all system boundaries.

These define WHAT the system needs without specifying HOW. Infrastructure
modules provide concrete implementations. Application services depend only
on these protocols, never on concrete classes.

Using typing.Protocol (structural typing) instead of ABC — any class that
implements the methods satisfies the port without explicit inheritance.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from domain.models import (
    UserIntent,
    NutritionConstraints,
    SafetyCheckResult,
    DetectedIngredients,
    Recipe,
)
from domain.entities import (
    User,
    Authentication,
    MedicalAdvice,
    RecipeHistory,
    NutritionHistory,
    UserProfileHistory,
    Conversation,
    ChatMessage,
)


# ---------------------------------------------------------------------------
# AI Component Ports
# ---------------------------------------------------------------------------

@runtime_checkable
class IntentParserPort(Protocol):
    """Parse user natural language into structured intent."""

    async def parse(self, query: str) -> UserIntent: ...


@runtime_checkable
class MedicalRAGPort(Protocol):
    """Retrieve medical dietary constraints for health conditions."""

    async def get_constraints(
        self, conditions: list[str],
    ) -> NutritionConstraints: ...


@runtime_checkable
class RecipeRAGPort(Protocol):
    """Retrieve and generate recipe recommendations as structured Recipe objects."""

    async def async_ask(self, query: str) -> list[Recipe]: ...


@runtime_checkable
class SafetyFilterPort(Protocol):
    """Validate recipe recommendations against user constraints."""

    async def check(
        self,
        recipes: list[Recipe],
        constraints: NutritionConstraints,
        intent: UserIntent,
    ) -> SafetyCheckResult: ...


@runtime_checkable
class IngredientDetectorPort(Protocol):
    """Detect ingredients from a food image using CNN."""

    async def detect(self, image_path: str) -> DetectedIngredients: ...


# ---------------------------------------------------------------------------
# Repository Ports
# ---------------------------------------------------------------------------

@runtime_checkable
class UserRepository(Protocol):
    """CRUD operations for User entities."""

    async def get_by_id(self, user_id: int) -> User | None: ...
    async def get_by_name(self, name: str, surname: str) -> User | None: ...
    async def save(self, user: User) -> int: ...
    async def update(self, user_id: int, field: str, value: object) -> None: ...
    async def soft_delete(self, user_id: int) -> None: ...


@runtime_checkable
class MedicalRepository(Protocol):
    """CRUD operations for MedicalAdvice entities."""

    async def save(self, advice: MedicalAdvice) -> int: ...
    async def get_by_user(self, user_id: int) -> list[MedicalAdvice]: ...
    async def soft_delete(self, advice_id: int) -> None: ...


@runtime_checkable
class RecipeRepository(Protocol):
    """CRUD operations for RecipeHistory entities."""

    async def save(self, history: RecipeHistory) -> int: ...
    async def get_by_user(self, user_id: int) -> list[RecipeHistory]: ...
    async def soft_delete(self, history_id: int) -> None: ...


@runtime_checkable
class NutritionRepository(Protocol):
    """CRUD operations for NutritionHistory entities."""

    async def save(self, history: NutritionHistory) -> int: ...
    async def get_by_user(self, user_id: int) -> list[NutritionHistory]: ...
    async def get_today_by_user(self, user_id: int) -> list[NutritionHistory]: ...
    async def soft_delete(self, history_id: int) -> None: ...


@runtime_checkable
class ProfileRepository(Protocol):
    """CRUD operations for UserProfileHistory entities."""

    async def save(self, profile: UserProfileHistory) -> int: ...
    async def get_by_user(self, user_id: int) -> list[UserProfileHistory]: ...
    async def soft_delete(self, history_id: int) -> None: ...


@runtime_checkable
class ConversationRepository(Protocol):
    """CRUD for Conversation metadata."""

    async def save(self, conversation: Conversation) -> int: ...
    async def get_by_user(self, user_id: int) -> list[Conversation]: ...
    async def get_by_conversation_id(self, conversation_id: str) -> Conversation | None: ...
    async def update_last_message(self, conversation_id: str) -> None: ...
    async def update_title(self, conversation_id: str, title: str) -> None: ...
    async def soft_delete(self, conversation_id: str) -> None: ...
    async def delete_old_for_user(self, user_id: int, cutoff_iso: str) -> int: ...


@runtime_checkable
class ChatMessageRepository(Protocol):
    """CRUD for ChatMessage entities."""

    async def save(self, message: ChatMessage) -> int: ...
    async def get_by_conversation(self, conversation_id: str) -> list[ChatMessage]: ...
    async def get_by_user(self, user_id: int) -> list[ChatMessage]: ...
    async def soft_delete(self, message_id: int) -> None: ...
    async def delete_old_for_user(self, user_id: int, cutoff_iso: str) -> int: ...


@runtime_checkable
class AuthenticationRepository(Protocol):
    """CRUD operations for Authentication entities."""

    async def save(self, auth: Authentication) -> int: ...
    async def get_by_login(self, login: str) -> Authentication | None: ...
    async def get_by_user_id(self, user_id: int) -> Authentication | None: ...
    async def soft_delete(self, auth_id: int) -> None: ...
