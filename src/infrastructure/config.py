"""
infrastructure.config - Typed, injectable configuration.

Replaces the module-level constants in settings.py with a frozen dataclass
that can be constructed from environment or passed explicitly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Single source of truth for the embedding model. Previously three different
# defaults lived in config.py, base_rag.py and medical_rag.py; the factory
# always overrode them, so the class defaults were traps for any code that
# constructed a RAG directly (scripts, tests, the index CLI).
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"


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

    # ── Centralized LLM Provider ────────────────────────────────
    # One setting controls ALL LLM components (agent, RAGs,
    # intent parser, safety filter). Allowed: "openai", "groq", "ollama"
    llm_provider: str = "ollama"

    # Model names — only the one matching llm_provider is used.
    llm_model_ollama: str = "llama3.2"
    llm_model_openai: str = "gpt-4.1-mini"
    llm_model_groq: str = "llama-3.3-70b-versatile"

    # Embeddings (always local HuggingFace — not affected by llm_provider).
    # Shared by both RAGs. Switching this auto-rebuilds the FAISS indices on
    # next startup (dimension-mismatch detection).
    embedding_model: str = DEFAULT_EMBEDDING_MODEL

    # Chunking for the medical corpus (the recipe corpus is one doc per CSV row
    # and is not split). Changing either invalidates the index — the app will
    # refuse to load it and tell you to rebuild.
    #
    # 300 chars is roughly 70 tokens, which splits dosage guidance mid-sentence.
    # Kept as the default so upgrading does not silently force a rebuild;
    # 800/150 is the better starting point when you next reindex.
    rag_chunk_size: int = 300
    rag_chunk_overlap: int = 50

    # RAG retrieval quality
    #   rag_use_hybrid   - combine dense (FAISS) + sparse (BM25) via RRF
    #   rag_use_reranker - cross-encoder rerank pass over the candidates
    rag_use_hybrid: bool = True
    rag_use_reranker: bool = True
    rag_reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rag_fetch_k: int = 30        # candidate pool size before rerank
    rag_rerank_top_n: int = 8    # docs kept after rerank (medical RAG)
    # Recipe docs passed to the LLM. Was 14, which put ~8-12k tokens of context
    # in front of a small local model alongside a prompt full of shape rules.
    # The shape rules now live in the bound schema, so the budget can shrink.
    # Raise it back via RAG_RECIPE_K if recall suffers.
    rag_recipe_k: int = 5

    # Connection details
    ollama_base_url: str = "http://localhost:11434/"
    groq_api_key: str = ""
    openai_api_key: str = ""

    # Agent
    agent_max_iterations: int = 5

    # Database
    db_path: str = "users.db"

    # CNN / Ingredient Detector.
    #   detector_backend = "llava" (zero-shot via Ollama) or "yolo" (local weights)
    # yolo_weights accepts a stock checkpoint name (auto-downloaded) or a path
    # to a fine-tuned best.pt — see training/README.md.
    detector_backend: str = "llava"
    cnn_model_path: str = ""
    cnn_class_labels_path: str = ""
    yolo_weights: str = "yolo11n.pt"
    yolo_confidence: float = 0.25
    yolo_device: str = "cpu"

    # Upload directory for images.
    # In Docker: mount ./uploads → /app/uploads.
    # Locally: defaults to ./uploads/ relative to the project root.
    upload_dir: str = "./uploads"

    # JWT
    jwt_secret: str = "change-me-in-production"
    jwt_expiry_hours: int = 24

    @property
    def active_llm_model(self) -> str:
        """Return the model name for the currently active LLM provider."""
        if self.llm_provider == "openai":
            return self.llm_model_openai
        elif self.llm_provider == "groq":
            return self.llm_model_groq
        return self.llm_model_ollama

    @classmethod
    def from_env(cls, project_root: Path | None = None) -> Settings:
        """Build Settings from standard project layout."""
        from dotenv import load_dotenv
        load_dotenv()

        root = project_root or Path(__file__).resolve().parent.parent.parent
        
        # Get API keys from environment
        openai_key = os.getenv("OPENAI_API_KEY", "")
        groq_key = os.getenv("GROQ_API_KEY", "")
        
        return cls(
            project_root=root,
            data_dir=root / "data",
            pdf_dir=root / "data_test" / "raw",
            processed_dir=root / "data" / "processed",
            medical_vectorstore_path=root / "vector_databases" / "vector_db_medi",
            recipes_nutrition_vector_path=root / "vector_databases",

            # Centralized LLM provider
            llm_provider=os.getenv("LLM_PROVIDER", "ollama"),
            llm_model_ollama=os.getenv("LLM_MODEL_OLLAMA", "llama3.2"),
            llm_model_openai=os.getenv("LLM_MODEL_OPENAI", "gpt-4.1-mini"),
            llm_model_groq=os.getenv("LLM_MODEL_GROQ", "llama-3.3-70b-versatile"),
            embedding_model=os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
            rag_chunk_size=int(os.getenv("RAG_CHUNK_SIZE", "300")),
            rag_chunk_overlap=int(os.getenv("RAG_CHUNK_OVERLAP", "50")),
            rag_use_hybrid=os.getenv("RAG_USE_HYBRID", "true").lower() in ("1", "true", "yes"),
            rag_use_reranker=os.getenv("RAG_USE_RERANKER", "true").lower() in ("1", "true", "yes"),
            rag_reranker_model=os.getenv("RAG_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
            rag_fetch_k=int(os.getenv("RAG_FETCH_K", "30")),
            rag_rerank_top_n=int(os.getenv("RAG_RERANK_TOP_N", "8")),
            rag_recipe_k=int(os.getenv("RAG_RECIPE_K", "5")),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/"),
            groq_api_key=groq_key,
            openai_api_key=openai_key,

            agent_max_iterations=int(os.getenv("AGENT_MAX_ITERATIONS", "5")),
            db_path=os.getenv("DB_PATH", "users.db"),
            detector_backend=os.getenv("DETECTOR_BACKEND", "llava").strip().lower(),
            cnn_model_path=os.getenv("CNN_MODEL_PATH", ""),
            cnn_class_labels_path=os.getenv("CNN_CLASS_LABELS_PATH", ""),
            yolo_weights=os.getenv("YOLO_WEIGHTS", "yolo11n.pt"),
            yolo_confidence=float(os.getenv("YOLO_CONFIDENCE", "0.25")),
            yolo_device=os.getenv("YOLO_DEVICE", "cpu"),
            upload_dir=os.getenv("UPLOAD_DIR", "./uploads"),
            jwt_secret=os.getenv("JWT_SECRET", "change-me-in-production"),
            jwt_expiry_hours=int(os.getenv("JWT_EXPIRY_HOURS", "24")),
        )
