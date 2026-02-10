"""
Centralized configuration for the nutrition-ai-assistant project.

All paths are anchored to PROJECT_ROOT (auto-detected from this file's location).
Import from here instead of hardcoding paths in pipeline modules.
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────
# This file lives at src/pipeline/config.py → .parent.parent.parent = project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_DIR = PROJECT_ROOT / "src" / "data"
PDF_DIR = DATA_DIR / "medical_pfds"
PROCESSED_DIR = DATA_DIR / "processed"

MEDICAL_VECTORSTORE_PATH = PROCESSED_DIR / "medical_pdfs_vectorstore"
NUTRITION_VECTORSTORE_PATH = PROCESSED_DIR / "nutrition_vectorstore"

# ── LLM / Model ───────────────────────────────────────────────
LLM_MODEL = "llama3.2"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
OLLAMA_BASE_URL = "http://localhost:11434/"
