"""
Run the Nutrition AI Assistant REST API.

Usage:
    python run_api.py

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

import sys
from pathlib import Path

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).parent / "src"))

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "adapters.rest.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
