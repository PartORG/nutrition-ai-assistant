"""
Run the Nutrition AI Assistant CLI.

Usage:
    python run_cli.py [COMMAND] [OPTIONS]

Commands:
    register   Create a new account
    login      Sign in and save credentials locally (~/.nutrition-ai/session.json)
    logout     Clear stored credentials
    whoami     Show the currently logged-in user
    profile    Display your health profile and medical advice
    ask        One-shot recipe/nutrition query  (requires login, loads full pipeline)
    chat       Interactive chat session          (requires login, loads full pipeline)
    init       Build/rebuild RAG vectorstores   (admin / first-time setup)

Examples:
    python run_cli.py login
    python run_cli.py ask "healthy dinner with chicken"
    python run_cli.py chat

Environment variables (all optional):
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

from adapters.cli.main import app

if __name__ == "__main__":
    app()
