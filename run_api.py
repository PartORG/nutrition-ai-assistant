"""
Run the Nutrition AI Assistant REST API.

Usage:
    python run_api.py

Environment variables (all optional):
    JWT_SECRET          Secret key for signing JWT tokens (change in production!)
    JWT_EXPIRY_HOURS    Token lifetime in hours (default: 24)
    LLM_MODEL           Ollama model for RAG (default: llama3.2)
    AGENT_LLM_MODEL     Model for conversational agent (default: llama-3.3-70b-versatile)
    AGENT_LLM_PROVIDER  "groq" or "ollama" (default: groq)
    GROQ_API_KEY        Required when AGENT_LLM_PROVIDER=groq
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
