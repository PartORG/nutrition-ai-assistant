"""
Run the Nutrition AI Assistant REST API.

Usage:
    uv run python run_api.py     (or: uv run nutriai-api)

Environment variables (all optional):
    JWT_SECRET          Secret key for signing JWT tokens (change in production!)
    JWT_EXPIRY_HOURS    Token lifetime in hours (default: 24)
    LLM_PROVIDER        "openai", "groq", or "ollama" — controls ALL LLM components
    LLM_MODEL_OPENAI    Model name when LLM_PROVIDER=openai (default: gpt-4.1-mini)
    LLM_MODEL_GROQ      Model name when LLM_PROVIDER=groq (default: llama-3.3-70b-versatile)
    LLM_MODEL_OLLAMA    Model name when LLM_PROVIDER=ollama (default: llama3.2)
    OPENAI_API_KEY      Required when LLM_PROVIDER=openai
    GROQ_API_KEY        Required when LLM_PROVIDER=groq
    DB_PATH             SQLite database file path (default: users.db)
    OLLAMA_BASE_URL     Ollama server URL (default: http://localhost:11434/)
"""

# The package is installed (uv sync), so `adapters.rest.app` resolves
# without any sys.path manipulation.
from adapters.rest.app import main

if __name__ == "__main__":
    main()
