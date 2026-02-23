"""
domain.exceptions - Custom exception hierarchy for the nutrition AI assistant.

All domain-level errors inherit from DomainError so callers can catch
broad or specific exceptions as needed.
"""


class DomainError(Exception):
    """Base exception for all domain-level errors."""


class IntentParsingError(DomainError):
    """Raised when user intent cannot be parsed from a query."""


class RAGError(DomainError):
    """Raised when a RAG system fails to retrieve or generate."""


class SafetyCheckError(DomainError):
    """Raised when safety filtering encounters an unrecoverable error."""


class IngredientDetectionError(DomainError):
    """Raised when CNN ingredient detection fails."""


class RepositoryError(DomainError):
    """Raised when a database operation fails."""


class AuthenticationError(DomainError):
    """Raised when authentication fails (bad credentials, expired token)."""


class DuplicateLoginError(DomainError):
    """Raised when attempting to register with a login that already exists."""
