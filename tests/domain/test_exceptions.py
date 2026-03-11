"""
Unit tests for domain/exceptions.py — custom exception hierarchy.
"""

import pytest
from domain.exceptions import (
    DomainError,
    IntentParsingError,
    RAGError,
    SafetyCheckError,
    IngredientDetectionError,
    RepositoryError,
    AuthenticationError,
    DuplicateLoginError,
)


class TestExceptionHierarchy:
    """Every custom exception must inherit from DomainError."""

    def test_domain_error_is_exception(self):
        assert issubclass(DomainError, Exception)

    def test_intent_parsing_error_is_domain_error(self):
        assert issubclass(IntentParsingError, DomainError)

    def test_rag_error_is_domain_error(self):
        assert issubclass(RAGError, DomainError)

    def test_safety_check_error_is_domain_error(self):
        assert issubclass(SafetyCheckError, DomainError)

    def test_ingredient_detection_error_is_domain_error(self):
        assert issubclass(IngredientDetectionError, DomainError)

    def test_repository_error_is_domain_error(self):
        assert issubclass(RepositoryError, DomainError)

    def test_authentication_error_is_domain_error(self):
        assert issubclass(AuthenticationError, DomainError)

    def test_duplicate_login_error_is_domain_error(self):
        assert issubclass(DuplicateLoginError, DomainError)


class TestExceptionsRaisable:
    """Exceptions can be raised and caught correctly."""

    def test_raise_authentication_error(self):
        with pytest.raises(AuthenticationError, match="bad creds"):
            raise AuthenticationError("bad creds")

    def test_raise_duplicate_login_error(self):
        with pytest.raises(DuplicateLoginError):
            raise DuplicateLoginError("Login 'alice' is taken.")

    def test_catch_as_domain_error(self):
        """Any specific error can be caught as the base DomainError."""
        with pytest.raises(DomainError):
            raise RepositoryError("DB failure")

    def test_catch_as_base_exception(self):
        with pytest.raises(Exception):
            raise RAGError("vector search failed")

    def test_error_message_preserved(self):
        msg = "intent could not be parsed"
        exc = IntentParsingError(msg)
        assert str(exc) == msg

    def test_authentication_error_not_caught_as_duplicate(self):
        """Sibling exceptions must not be interchangeable."""
        with pytest.raises(AuthenticationError):
            try:
                raise AuthenticationError("bad")
            except DuplicateLoginError:
                pass  # must NOT match
