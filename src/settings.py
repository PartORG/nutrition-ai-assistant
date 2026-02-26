"""
Centralized configuration for the nutrition-ai-assistant project.

All paths are anchored to PROJECT_ROOT (auto-detected from this file's location).
Import from here instead of hardcoding paths in other modules.
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────
# This file lives at src/settings.py → .parent.parent = project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "src" / "data"
PDF_DIR = PROJECT_ROOT / "data_test" / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

MEDICAL_VECTORSTORE_PATH = PROJECT_ROOT / "vector_databases" / "vector_db_medi"
RECIPES_NUTRITION_VECTOR_PATH = PROJECT_ROOT / "vector_databases"

# ── LLM / Model ───────────────────────────────────────────────
LLM_MODEL = "llama3.2"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
OLLAMA_BASE_URL = "http://localhost:11434/"
