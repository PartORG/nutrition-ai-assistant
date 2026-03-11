"""
Conftest for components/ tests.

The legacy components (safety_filter, intent_retriever) import LangChain and
OllamaLLM at module load time, which are not available in the lightweight CI
environment.  We stub those out here — before any test in this directory
causes them to be imported — so that only the pure Python rule-based logic
is exercised.
"""

import sys
from unittest.mock import MagicMock

# Stub out heavy ML imports before any test file in this package imports them.
_STUBS = [
    "langchain_core",
    "langchain_core.prompts",
    "langchain_core.output_parsers",
    "langchain_ollama",
]

for _mod in _STUBS:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

# Ensure the stub modules expose the names used in the source files
sys.modules["langchain_core.prompts"].ChatPromptTemplate = MagicMock()
sys.modules["langchain_core.output_parsers"].JsonOutputParser = MagicMock()
sys.modules["langchain_ollama"].OllamaLLM = MagicMock()
