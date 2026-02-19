"""
infrastructure.config - Typed, injectable configuration.

Replaces the module-level constants in settings.py with a frozen dataclass
that can be constructed from environment or passed explicitly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class Settings:
    """Centralized configuration for the nutrition AI assistant.

    All paths are absolute. No module-level globals — construct via from_env()
    or pass explicitly in tests.
    """
    project_root: Path

    # Data directories
    data_dir: Path
    pdf_dir: Path
    processed_dir: Path

    # Vector databases
    medical_vectorstore_path: Path
    recipes_nutrition_vector_path: Path

    # LLM / Embeddings
    llm_model: str = "llama3.2"
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    ollama_base_url: str = "http://localhost:11434/"

    # Agent LLM (can differ from RAG LLM)
    # Supported Ollama models: "llama3.2", "llama3:8b", "llama3.1", "mistral", "gemma2"
    # Supported Groq models:   "llama-3.1-8b-instant", "llama-3.3-70b-versatile", "mixtral-8x7b-32768"
    # Set via env: AGENT_LLM_MODEL=llama3:8b  AGENT_LLM_PROVIDER=ollama
    agent_llm_model: str = "llama3.2"
    agent_llm_provider: str = "ollama"  # "ollama" or "groq"
    agent_max_iterations: int = 5      # max agentic loop steps (3 is often too tight)

    # Database
    db_path: str = "users.db"

    # CNN / Ingredient Detector
    # cnn_model_path is repurposed as the LLaVA model name (e.g. "llava", "llava:13b")
    cnn_model_path: str = ""
    cnn_class_labels_path: str = ""
    # Detector selection: "yolo_with_fallback" | "yolo_only" | "llava_only"
    cnn_detector_type: str = "yolo_with_fallback"
    # URL of the YOLO microservice (Docker: http://yolo-detector:8001, local: http://localhost:8001)
    yolo_service_url: str = os.getenv("YOLO_SERVICE_URL", "http://localhost:8001")

    # JWT
    jwt_secret: str = "change-me-in-production"
    jwt_expiry_hours: int = 24

    @classmethod
    def from_env(cls, project_root: Optional[Path] = None) -> Settings:
        """Build Settings from standard project layout.

        Autodetects project root from this file's location if not provided.
        """
        root = project_root or Path(__file__).resolve().parent.parent.parent
        return cls(
            project_root=root,
            data_dir=root / "data",
            pdf_dir=root / "data_test" / "raw",
            processed_dir=root / "data" / "processed",
            medical_vectorstore_path=root / "vector_databases" / "vector_db_medi",
            recipes_nutrition_vector_path=root / "vector_databases",
            llm_model=os.getenv("LLM_MODEL", "llama3.2"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2"),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/"),
            agent_llm_model=os.getenv("AGENT_LLM_MODEL", "llama3.2"),
            agent_llm_provider=os.getenv("AGENT_LLM_PROVIDER", "ollama"),
            agent_max_iterations=int(os.getenv("AGENT_MAX_ITERATIONS", "5")),
            db_path=os.getenv("DB_PATH", "users.db"),
            cnn_model_path=os.getenv("CNN_MODEL_PATH", ""),
            cnn_class_labels_path=os.getenv("CNN_CLASS_LABELS_PATH", ""),
            cnn_detector_type=os.getenv("CNN_DETECTOR_TYPE", "yolo_with_fallback"),
            yolo_service_url=os.getenv("YOLO_SERVICE_URL", "http://localhost:8001"),
            jwt_secret=os.getenv("JWT_SECRET", "change-me-in-production"),
            jwt_expiry_hours=int(os.getenv("JWT_EXPIRY_HOURS", "24")),
        )
