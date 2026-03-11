"""
Root conftest.py — loaded by pytest before any other conftest or test module.

Adds src/ to sys.path so that `from application.context import ...` style
imports work without requiring a PYTHONPATH environment variable.

Note: src/__init__.py exists (it is part of the production package), which
prevents the pytest.ini `pythonpath` option from resolving it cleanly.
This explicit sys.path insertion is the reliable cross-platform fix.
"""

import sys
from pathlib import Path

# Insert src/ at the front of sys.path so our packages take priority.
sys.path.insert(0, str(Path(__file__).parent / "src"))
