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
    #
    # The Ollama tag must be one that is actually pulled: the HTTP API does
    # NOT auto-pull the way `ollama run` does, so a bare "llama3.2" against a
    # host that only has "llama3.2:3b" 404s on every request. Pin the tag.
    # For better tool-calling with the 9-tool agent, `ollama pull qwen3:4b`
    # and set LLM_MODEL_OLLAMA=qwen3:4b.
    llm_model_ollama: str = "llama3.2:3b"
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

    # data/cleaned_healthy_meals.csv is fabricated filler, not food data: names
    # are 835 random words x 8 dish suffixes ("Suffer Stew"), macros are
    # uniformly random, 65% of its "Keto" rows exceed 50g carbs and 100% of its
    # "Vegan" rows carry dietary cholesterol. Its documents contain no name, no
    # ingredients and no instructions — only macros — so the safety filter has
    # nothing to check them against, yet they are 39% of the retrievable corpus.
    # Excluded by default. Set RAG_INCLUDE_SYNTHETIC_MEALS=true to restore the
    # old behaviour; changing it requires `nutriai-index rebuild`.
    rag_include_synthetic_meals: bool = False

    # Connection details
    ollama_base_url: str = "http://localhost:11434/"
    groq_api_key: str = ""
    openai_api_key: str = ""

    # Agent
    agent_max_iterations: int = 5

    # Database
    db_path: str = "users.db"

    # CNN / Ingredient Detector.
    #   detector_backend = "hybrid" (default: VLM proposes, YOLO cross-checks
    #                      and scores, either one failing falls back to the
    #                      other), "vlm"/"llava" (VLM only), "yolo" (YOLO only)
    #
    # yolo_weights defaults to YOLOE prompt-free: 4585 open-vocabulary classes
    # including ~200 foods ("chicken breast", "boiled egg", "rice"), versus
    # the 10 foods a stock COCO checkpoint knows. Auto-downloaded on first
    # run. A fine-tuned best.pt also works — see training/README.md.
    detector_backend: str = "hybrid"
    vlm_model: str = "qwen2.5vl:7b"
    # Layers to force onto the GPU for the VLM. 0 = let Ollama decide. Set to
    # 99 (all) when Ollama refuses to load a vision model with "requires more
    # system memory" despite the GPU having room — its estimator is
    # conservative for multimodal models.
    vlm_num_gpu: int = 0
    cnn_model_path: str = ""
    cnn_class_labels_path: str = ""
    yolo_weights: str = "yoloe-11s-seg-pf.pt"
    yolo_confidence: float = 0.25
    yolo_device: str = "cpu"

    # Upload directory for images.
    # In Docker: mount ./uploads → /app/uploads.
    # Locally: defaults to ./uploads/ relative to the project root.
    upload_dir: str = "./uploads"

    # JWT. There is deliberately no usable default: from_env() refuses to boot
    # without a real secret, because anyone who has read this repo can forge
    # {"user_id": 1, "role": "admin"} against a known one.
    jwt_secret: str = ""
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

        jwt_secret = os.getenv("JWT_SECRET", "").strip()

        return cls(
            project_root=root,
            data_dir=root / "data",
            pdf_dir=root / "data_test" / "raw",
            processed_dir=root / "data" / "processed",
            medical_vectorstore_path=root / "vector_databases" / "vector_db_medi",
            recipes_nutrition_vector_path=root / "vector_databases",

            # Centralized LLM provider
            llm_provider=os.getenv("LLM_PROVIDER", "ollama"),
            llm_model_ollama=os.getenv("LLM_MODEL_OLLAMA", "llama3.2:3b"),
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
            rag_include_synthetic_meals=os.getenv(
                "RAG_INCLUDE_SYNTHETIC_MEALS", "false"
            ).lower() in ("1", "true", "yes"),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/"),
            groq_api_key=groq_key,
            openai_api_key=openai_key,

            agent_max_iterations=int(os.getenv("AGENT_MAX_ITERATIONS", "5")),
            db_path=os.getenv("DB_PATH", "users.db"),
            detector_backend=os.getenv("DETECTOR_BACKEND", "hybrid").strip().lower(),
            vlm_model=os.getenv("DETECTOR_VLM_MODEL", "qwen2.5vl:7b"),
            vlm_num_gpu=int(os.getenv("DETECTOR_VLM_NUM_GPU", "0")),
            cnn_model_path=os.getenv("CNN_MODEL_PATH", ""),
            cnn_class_labels_path=os.getenv("CNN_CLASS_LABELS_PATH", ""),
            yolo_weights=os.getenv("YOLO_WEIGHTS", "yoloe-11s-seg-pf.pt"),
            yolo_confidence=float(os.getenv("YOLO_CONFIDENCE", "0.25")),
            yolo_device=os.getenv("YOLO_DEVICE", "cpu"),
            upload_dir=os.getenv("UPLOAD_DIR", "./uploads"),
            jwt_secret=jwt_secret,
            jwt_expiry_hours=int(os.getenv("JWT_EXPIRY_HOURS", "24")),
        )
