"""
Shared pytest fixtures for the nutrition-ai-assistant test suite.

sys.path is patched here (before any project import) so that packages
under src/ are importable as top-level names regardless of how pytest is
invoked.  This is needed because src/__init__.py exists, which can cause
the pytest.ini `pythonpath` option to resolve the path after test modules
are already being imported.
"""

import sys
from pathlib import Path

# Ensure src/ is at the front of sys.path before any project import.
_SRC = str(Path(__file__).resolve().parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import pytest
from application.context import SessionContext


@pytest.fixture
def session_ctx() -> SessionContext:
    """A minimal SessionContext for use in agent/service tests."""
    return SessionContext(user_id=1, conversation_id="test-conv-001")
