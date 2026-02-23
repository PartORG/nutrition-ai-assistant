"""
infrastructure.rag.base_rag - Abstract base class for all RAG systems.

Migrated from rags/base_rag.py — kept largely intact as it's the strongest
abstraction in the original codebase. Changes:
    - Imports updated (no sys.path.insert)
    - ask() wrapped for async compatibility in subclasses
"""

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Optional, Set

from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy
from langchain_core.language_models import BaseChatModel
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

from infrastructure.llm.llm_builder import build_llm as _build_llm_central

logger = logging.getLogger(__name__)


class BaseRAG(ABC):
    """Abstract base class for RAG systems.

    Subclasses must implement:
        _ingest_documents()    – load all raw data and return List[Document]
        _ingest_single_file()  – load one file and return List[Document]

    Subclasses must define:
        SYSTEM_PROMPT (str)    – class-level prompt with {input} and {context}
    """

    SYSTEM_PROMPT: str = ""

    def __init__(
        self,
        vectorstore_path: str,
        model_name: str = "llama3.2",
        temperature: float = 0.3,
        embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
        ollama_base_url: str = "http://localhost:11434/",
        llm_format: Optional[str] = None,
        llm_provider: str = "ollama",  # NEW: "ollama", "groq", or "openai"
        openai_api_key: Optional[str] = None,  # NEW: API key for OpenAI
        groq_api_key: Optional[str] = None,  # NEW: API key for Groq
    ):
        self.vectorstore_path = Path(vectorstore_path)
        self.model_name = model_name
        self.temperature = temperature
        self.embedding_model_name = embedding_model
        self.ollama_base_url = ollama_base_url
        self.llm_format = llm_format
        self.llm_provider = llm_provider  # NEW
        self.openai_api_key = openai_api_key  # NEW
        self.groq_api_key = groq_api_key  # NEW

        self.embeddings: Optional[HuggingFaceEmbeddings] = None
        self.llm: Optional[BaseChatModel] = None  # CHANGED: more generic type
        self.vectorstore: Optional[FAISS] = None
        self.retriever = None
        self.rag_chain = None

        self._custom_prompt: Optional[str] = None

        logger.info(
            "%s instance created (provider=%s, model=%s, vectorstore=%s)",
            self.__class__.__name__, self.llm_provider, self.model_name, self.vectorstore_path,
        )

    # ================================================================
    # Public API
    # ================================================================

    def initialize(self, force_rebuild: bool = False) -> None:
        """Orchestrate full system setup (template method)."""
        logger.info("Initializing %s...", self.__class__.__name__)

        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name,
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info("Embedding model loaded")

        # Build LLM based on configured provider
        self.llm = self._build_llm()
        logger.info("LLM initialized (provider=%s, model=%s)", self.llm_provider, self.model_name)

        if not force_rebuild and self._vectorstore_exists():
            logger.info("Loading existing vectorstore from disk")
            self._load_vectorstore()
            if self.vectorstore is not None:
                try:
                    self.vectorstore.similarity_search("dimension check", k=1)
                except AssertionError:
                    logger.warning(
                        "Vectorstore dimension mismatch detected — rebuilding "
                        "index with current embedding model (%s)",
                        self.embedding_model_name,
                    )
                    docs = self._ingest_documents()
                    if not docs:
                        logger.warning(
                            "No documents available to rebuild — RAG non-functional"
                        )
                        return
                    self._build_vectorstore(docs)
        else:
            reason = "force_rebuild=True" if force_rebuild else "no existing index"
            logger.info("Building vectorstore (%s)", reason)
            docs = self._ingest_documents()
            if not docs:
                logger.warning("No documents ingested — RAG will be non-functional")
                return
            self._build_vectorstore(docs)

        self._setup_retriever()
        self._build_chain()
        logger.info("%s ready for queries", self.__class__.__name__)

    def _build_llm(self) -> BaseChatModel:
        """Build the LLM based on the configured provider.
        
        Delegates to the centralized llm_builder so that provider logic
        lives in exactly one place.
        
        Returns:
            LLM instance for the selected provider.
        """
        return _build_llm_central(
            provider=self.llm_provider,
            model=self.model_name,
            temperature=self.temperature,
            ollama_base_url=self.ollama_base_url,
            openai_api_key=self.openai_api_key or "",
            groq_api_key=self.groq_api_key or "",
            json_mode=bool(self.llm_format),
        )

    def ask(self, query: str) -> str:
        """Query the RAG system and return the answer string."""
        if not self.rag_chain:
            logger.error("RAG chain not initialized — call initialize() first")
            return "System not initialized. Call initialize() first."

        logger.debug("Processing query: %s...", query[:80])
        response = self.rag_chain.invoke({"input": query})
        return response.get("answer", "No response generated.")

    def add_documents(self, file_path: str) -> int:
        """Process a single file and merge into the existing vectorstore."""
        if self.embeddings is None:
            raise RuntimeError("Call initialize() before add_documents()")

        file_path_resolved = Path(file_path).resolve()
        indexed = self._get_indexed_files()

        if str(file_path_resolved) in indexed:
            logger.info("File already indexed, skipping: %s", file_path_resolved.name)
            return 0

        logger.info("Ingesting file: %s", file_path_resolved.name)
        docs = self._ingest_single_file(str(file_path_resolved))
        if not docs:
            logger.warning("No documents produced from %s", file_path_resolved.name)
            return 0

        self._merge_into_vectorstore(docs)

        indexed.add(str(file_path_resolved))
        self._save_indexed_files(indexed)

        logger.info("Added %d chunks from %s", len(docs), file_path_resolved.name)
        return len(docs)

    # ================================================================
    # Hooks – override in subclasses
    # ================================================================

    @abstractmethod
    def _ingest_documents(self) -> List[Document]:
        """Load all raw data and return a list of LangChain Documents."""

    @abstractmethod
    def _ingest_single_file(self, file_path: str) -> List[Document]:
        """Load a single file and return a list of LangChain Documents."""

    # ================================================================
    # System prompt
    # ================================================================

    def _get_system_prompt(self) -> str:
        if self._custom_prompt is not None:
            return self._custom_prompt
        return self.SYSTEM_PROMPT

    # ================================================================
    # Vectorstore management
    # ================================================================

    def _vectorstore_exists(self) -> bool:
        return self.vectorstore_path.exists()

    def _build_vectorstore(self, documents: List[Document]) -> None:
        logger.info("Creating FAISS index from %d documents", len(documents))
        self.vectorstore = FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings,
            distance_strategy=DistanceStrategy.COSINE,
        )
        self.vectorstore_path.parent.mkdir(parents=True, exist_ok=True)
        self.vectorstore.save_local(str(self.vectorstore_path))
        logger.info("Vectorstore saved to %s", self.vectorstore_path)

    def _load_vectorstore(self) -> None:
        logger.info("Loading vectorstore from %s", self.vectorstore_path)
        self.vectorstore = FAISS.load_local(
            folder_path=str(self.vectorstore_path),
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True,
        )
        logger.info("Vectorstore loaded (%d vectors)", self.vectorstore.index.ntotal)

    def _merge_into_vectorstore(self, documents: List[Document]) -> None:
        new_store = FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings,
            distance_strategy=DistanceStrategy.COSINE,
        )
        if self.vectorstore is None:
            self.vectorstore = new_store
        else:
            self.vectorstore.merge_from(new_store)

        self.vectorstore_path.parent.mkdir(parents=True, exist_ok=True)
        self.vectorstore.save_local(str(self.vectorstore_path))
        logger.info(
            "Merged %d new chunks — total vectors: %d",
            len(documents), self.vectorstore.index.ntotal,
        )

    def _setup_retriever(self) -> None:
        if self.vectorstore is None:
            logger.warning("No vectorstore available — retriever not created")
            return
        self.retriever = self.vectorstore.as_retriever()

    # ================================================================
    # Chain building
    # ================================================================

    def _build_chain(self) -> None:
        if self.retriever is None:
            logger.warning("No retriever available — chain not built")
            return

        system_prompt = self._get_system_prompt()
        prompt = ChatPromptTemplate.from_template(system_prompt)

        stuff_chain = create_stuff_documents_chain(llm=self.llm, prompt=prompt)
        self.rag_chain = create_retrieval_chain(
            retriever=self.retriever,
            combine_docs_chain=stuff_chain,
        )

    # ================================================================
    # Indexed file tracking
    # ================================================================

    def _index_meta_path(self) -> Path:
        return self.vectorstore_path.parent / f"{self.vectorstore_path.name}_indexed.json"

    def _get_indexed_files(self) -> Set[str]:
        meta_path = self._index_meta_path()
        if not meta_path.exists():
            return set()
        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return set(data.get("files", []))

    def _save_indexed_files(self, files: Set[str]) -> None:
        meta_path = self._index_meta_path()
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({"files": sorted(files)}, f, indent=2)

    # ================================================================
    # Utilities
    # ================================================================

    def update_system_prompt(self, new_prompt: str) -> None:
        if "{input}" not in new_prompt or "{context}" not in new_prompt:
            raise ValueError("Prompt must contain {input} and {context} placeholders")
        self._custom_prompt = new_prompt
        self._build_chain()
        logger.info("System prompt updated and chain rebuilt")

    def get_stats(self) -> Dict[str, Any]:
        stats: Dict[str, Any] = {
            "class": self.__class__.__name__,
            "provider": self.llm_provider,
            "model": self.model_name,
            "temperature": self.temperature,
            "vectorstore_path": str(self.vectorstore_path),
            "status": "initialized" if self.rag_chain else "not_initialized",
        }
        if self.vectorstore:
            stats["vector_count"] = self.vectorstore.index.ntotal
        return stats
