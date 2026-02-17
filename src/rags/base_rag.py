"""
BaseRAG - Abstract base class for all RAG systems in the pipeline.

Provides a unified interface and shared infrastructure (embeddings, LLM,
FAISS vectorstore management, chain building) so that concrete RAG classes
only need to implement data-specific logic.

Template Method pattern:
    initialize()  orchestrates the full setup by calling overridable hooks:
        _ingest_documents()   -> load raw data into LangChain Documents
        _build_vectorstore()  -> create FAISS index from documents
        _load_vectorstore()   -> load existing FAISS index from disk
        _setup_retriever()    -> wrap vectorstore(s) into a retriever
        _get_system_prompt()  -> return the system prompt string
        _build_chain()        -> assemble the RAG chain

Incremental ingestion:
    add_documents(file_path)  -> process a single file and merge into
                                 the existing vectorstore without rebuilding.
    _ingest_single_file()     -> subclass hook for per-file loading.
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
from langchain_ollama import OllamaLLM
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

logger = logging.getLogger(__name__)


class BaseRAG(ABC):
    """Abstract base class for RAG systems.

    Subclasses must implement:
        _ingest_documents()    – load all raw data and return List[Document]
        _ingest_single_file()  – load one file and return List[Document]

    Subclasses must define:
        SYSTEM_PROMPT (str)    – class-level prompt with {input} and {context}

    Subclasses may override:
        _build_vectorstore()  – custom vectorstore creation logic
        _load_vectorstore()   – custom vectorstore loading logic
        _setup_retriever()    – custom retriever (e.g. SmartRetriever)
        _build_chain()        – custom chain assembly
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
    ):
        self.vectorstore_path = Path(vectorstore_path)
        self.model_name = model_name
        self.temperature = temperature
        self.embedding_model_name = embedding_model
        self.ollama_base_url = ollama_base_url
        self.llm_format = llm_format

        # Initialised during initialize()
        self.embeddings: Optional[HuggingFaceEmbeddings] = None
        self.llm: Optional[OllamaLLM] = None
        self.vectorstore: Optional[FAISS] = None
        self.retriever = None
        self.rag_chain = None

        # Custom prompt override (set via update_system_prompt)
        self._custom_prompt: Optional[str] = None

        logger.info(
            "%s instance created (model=%s, vectorstore=%s)",
            self.__class__.__name__, self.model_name, self.vectorstore_path,
        )

    # ================================================================
    # Public API
    # ================================================================

    def initialize(self, force_rebuild: bool = False) -> None:
        """Orchestrate full system setup (template method).

        1. Init embeddings
        2. Init LLM
        3. Load or build vectorstore
        4. Setup retriever
        5. Build RAG chain
        """
        logger.info("Initializing %s...", self.__class__.__name__)

        # Step 1 – Embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name,
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info("Embedding model loaded")

        # Step 2 – LLM
        llm_kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "temperature": self.temperature,
            "base_url": self.ollama_base_url,
        }
        if self.llm_format:
            llm_kwargs["format"] = self.llm_format
        self.llm = OllamaLLM(**llm_kwargs)
        logger.info("LLM initialized (model=%s)", self.model_name)

        # Step 3 – Vectorstore
        if not force_rebuild and self._vectorstore_exists():
            logger.info("Loading existing vectorstore from disk")
            self._load_vectorstore()
        else:
            reason = "force_rebuild=True" if force_rebuild else "no existing index"
            logger.info("Building vectorstore (%s)", reason)
            docs = self._ingest_documents()
            if not docs:
                logger.warning("No documents ingested — RAG will be non-functional")
                return
            self._build_vectorstore(docs)

        # Step 4 – Retriever
        self._setup_retriever()

        # Step 5 – Chain
        self._build_chain()
        logger.info("%s ready for queries", self.__class__.__name__)

    def ask(self, query: str) -> str:
        """Query the RAG system and return the answer string."""
        if not self.rag_chain:
            logger.error("RAG chain not initialized — call initialize() first")
            return "System not initialized. Call initialize() first."

        logger.debug("Processing query: %s...", query[:80])
        response = self.rag_chain.invoke({"input": query})
        return response.get("answer", "No response generated.")

    def add_documents(self, file_path: str) -> int:
        """Process a single file and merge into the existing vectorstore.

        Returns the number of document chunks added.
        Requires that initialize() has been called first (embeddings must exist).
        """
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
        """Load all raw data and return a list of LangChain Documents.

        This is the main extension point for batch ingestion.
        """

    @abstractmethod
    def _ingest_single_file(self, file_path: str) -> List[Document]:
        """Load a single file and return a list of LangChain Documents.

        Used by add_documents() for incremental ingestion.
        """

    # ================================================================
    # System prompt (concrete — subclasses define SYSTEM_PROMPT)
    # ================================================================

    def _get_system_prompt(self) -> str:
        """Return the active system prompt.

        Uses _custom_prompt if set via update_system_prompt(),
        otherwise falls back to the class-level SYSTEM_PROMPT.
        """
        if self._custom_prompt is not None:
            return self._custom_prompt
        return self.SYSTEM_PROMPT

    # ================================================================
    # Vectorstore management (overridable)
    # ================================================================

    def _vectorstore_exists(self) -> bool:
        """Check whether a saved vectorstore exists on disk."""
        return self.vectorstore_path.exists()

    def _build_vectorstore(self, documents: List[Document]) -> None:
        """Create a FAISS vectorstore from documents and persist to disk."""
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
        """Load an existing FAISS vectorstore from disk."""
        logger.info("Loading vectorstore from %s", self.vectorstore_path)
        self.vectorstore = FAISS.load_local(
            folder_path=str(self.vectorstore_path),
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True,
        )
        logger.info("Vectorstore loaded (%d vectors)", self.vectorstore.index.ntotal)

    def _merge_into_vectorstore(self, documents: List[Document]) -> None:
        """Embed new documents and merge into the existing FAISS index."""
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
        """Wrap the vectorstore into a retriever.

        Override this to use a custom retriever (e.g. SmartRetriever,
        EnsembleRetriever, BM25 hybrid, etc.).
        """
        if self.vectorstore is None:
            logger.warning("No vectorstore available — retriever not created")
            return
        self.retriever = self.vectorstore.as_retriever()

    # ================================================================
    # Chain building (overridable)
    # ================================================================

    def _build_chain(self) -> None:
        """Assemble the RAG chain from retriever + LLM + prompt."""
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
        """Path to the JSON sidecar that tracks indexed files."""
        return self.vectorstore_path.parent / f"{self.vectorstore_path.name}_indexed.json"

    def _get_indexed_files(self) -> Set[str]:
        """Read the set of already-indexed file paths from disk."""
        meta_path = self._index_meta_path()
        if not meta_path.exists():
            return set()
        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return set(data.get("files", []))

    def _save_indexed_files(self, files: Set[str]) -> None:
        """Persist the set of indexed file paths to disk."""
        meta_path = self._index_meta_path()
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({"files": sorted(files)}, f, indent=2)

    # ================================================================
    # Utilities
    # ================================================================

    def update_system_prompt(self, new_prompt: str) -> None:
        """Replace the system prompt and rebuild the chain."""
        if "{input}" not in new_prompt or "{context}" not in new_prompt:
            raise ValueError("Prompt must contain {input} and {context} placeholders")
        self._custom_prompt = new_prompt
        self._build_chain()
        logger.info("System prompt updated and chain rebuilt")

    def get_stats(self) -> Dict[str, Any]:
        """Return basic system statistics."""
        stats: Dict[str, Any] = {
            "class": self.__class__.__name__,
            "model": self.model_name,
            "temperature": self.temperature,
            "vectorstore_path": str(self.vectorstore_path),
            "status": "initialized" if self.rag_chain else "not_initialized",
        }
        if self.vectorstore:
            stats["vector_count"] = self.vectorstore.index.ntotal
        return stats
