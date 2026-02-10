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
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Optional

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
        _ingest_documents()  – load raw data and return List[Document]
        _get_system_prompt() – return the system prompt (must contain {input} and {context})

    Subclasses may override:
        _build_vectorstore()  – custom vectorstore creation logic
        _load_vectorstore()   – custom vectorstore loading logic
        _setup_retriever()    – custom retriever (e.g. SmartRetriever)
        _build_chain()        – custom chain assembly
    """

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

        logger.info(
            f"{self.__class__.__name__} instance created "
            f"(model={self.model_name}, vectorstore={self.vectorstore_path})"
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
        logger.info(f"Initializing {self.__class__.__name__}...")

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
        logger.info(f"LLM initialized (model={self.model_name})")

        # Step 3 – Vectorstore
        if not force_rebuild and self._vectorstore_exists():
            logger.info("Loading existing vectorstore from disk")
            self._load_vectorstore()
        else:
            reason = "force_rebuild=True" if force_rebuild else "no existing index"
            logger.info(f"Building vectorstore ({reason})")
            docs = self._ingest_documents()
            if not docs:
                logger.warning("No documents ingested — RAG will be non-functional")
                return
            self._build_vectorstore(docs)

        # Step 4 – Retriever
        self._setup_retriever()

        # Step 5 – Chain
        self._build_chain()
        logger.info(f"{self.__class__.__name__} ready for queries")

    def ask(self, query: str) -> str:
        """Query the RAG system and return the answer string."""
        if not self.rag_chain:
            logger.error("RAG chain not initialized — call initialize() first")
            return "System not initialized. Call initialize() first."

        logger.debug(f"Processing query: {query[:80]}...")
        response = self.rag_chain.invoke({"input": query})
        return response.get("answer", "No response generated.")

    # ================================================================
    # Hooks – override in subclasses
    # ================================================================

    @abstractmethod
    def _ingest_documents(self) -> List[Document]:
        """Load raw data and return a list of LangChain Documents.

        This is the main extension point: PDF loading, CSV parsing, etc.
        """

    @abstractmethod
    def _get_system_prompt(self) -> str:
        """Return the system prompt string.

        Must contain ``{input}`` and ``{context}`` placeholders.
        """

    # ================================================================
    # Vectorstore management (overridable)
    # ================================================================

    def _vectorstore_exists(self) -> bool:
        """Check whether a saved vectorstore exists on disk."""
        return self.vectorstore_path.exists()

    def _build_vectorstore(self, documents: List[Document]) -> None:
        """Create a FAISS vectorstore from documents and persist to disk."""
        logger.info(f"Creating FAISS index from {len(documents)} documents")
        self.vectorstore = FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings,
            distance_strategy=DistanceStrategy.COSINE,
        )
        self.vectorstore_path.parent.mkdir(parents=True, exist_ok=True)
        self.vectorstore.save_local(str(self.vectorstore_path))
        logger.info(f"Vectorstore saved to {self.vectorstore_path}")

    def _load_vectorstore(self) -> None:
        """Load an existing FAISS vectorstore from disk."""
        self.vectorstore = FAISS.load_local(
            folder_path=str(self.vectorstore_path),
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True,
        )
        logger.info(f"Vectorstore loaded ({self.vectorstore.index.ntotal} vectors)")

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
    # Utilities
    # ================================================================

    def update_system_prompt(self, new_prompt: str) -> None:
        """Replace the system prompt and rebuild the chain."""
        if "{input}" not in new_prompt or "{context}" not in new_prompt:
            raise ValueError("Prompt must contain {input} and {context} placeholders")
        # Stash the new prompt so _get_system_prompt can return it
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
