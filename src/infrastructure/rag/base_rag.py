"""
infrastructure.rag.base_rag - Abstract base class for all RAG systems.

Migrated from rags/base_rag.py — kept largely intact as it's the strongest
abstraction in the original codebase. Changes:
    - Imports updated (no sys.path.insert)
    - ask() wrapped for async compatibility in subclasses
"""

import json
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel

from domain.exceptions import RAGError
from infrastructure.config import DEFAULT_EMBEDDING_MODEL
from infrastructure.llm.llm_builder import build_llm as _build_llm_central
from infrastructure.rag.index_meta import (
    META_FILENAME,
    IndexMeta,
    describe_mismatch,
    read_meta,
    stale_sources,
    write_meta,
)

if TYPE_CHECKING:
    from infrastructure.rag.retrieval import Reranker

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

    # Optional Pydantic schema. When set, the chain returns a populated model
    # instead of a string, via the provider's structured-output API.
    OUTPUT_SCHEMA: type[BaseModel] | None = None

    def __init__(
        self,
        vectorstore_path: str,
        model_name: str = "llama3.2",
        temperature: float = 0.3,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        ollama_base_url: str = "http://localhost:11434/",
        llm_format: str | None = None,
        llm_provider: str = "ollama",  # NEW: "ollama", "groq", or "openai"
        openai_api_key: str | None = None,  # NEW: API key for OpenAI
        groq_api_key: str | None = None,  # NEW: API key for Groq
        use_hybrid: bool = False,  # dense + BM25 hybrid retrieval
        reranker: Optional["Reranker"] = None,  # cross-encoder rerank pass
        fetch_k: int = 30,  # candidate pool size before rerank
        rerank_top_n: int = 8,  # docs returned after rerank
        embeddings: HuggingFaceEmbeddings | None = None,  # shared, prebuilt instance
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
        self.use_hybrid = use_hybrid
        self.reranker = reranker
        self.fetch_k = fetch_k
        self.rerank_top_n = rerank_top_n
        self._injected_embeddings = embeddings

        self.embeddings: HuggingFaceEmbeddings | None = None
        self.llm: BaseChatModel | None = None  # CHANGED: more generic type
        self.vectorstore: FAISS | None = None
        self.retriever = None
        self.rag_chain = None

        self._custom_prompt: str | None = None

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

        # Reuse a shared embeddings instance when one is injected (so a large
        # model like BAAI/bge-m3 is loaded once, not once per RAG).
        self.embeddings = self._injected_embeddings or HuggingFaceEmbeddings(
            model_name=self.embedding_model_name,
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info("Embedding model ready (%s)", self.embedding_model_name)

        # Build LLM based on configured provider
        self.llm = self._build_llm()
        logger.info("LLM initialized (provider=%s, model=%s)", self.llm_provider, self.model_name)

        needs_build = force_rebuild or not self._vectorstore_exists()
        if not needs_build:
            # Check provenance BEFORE loading — refusing early avoids paying to
            # deserialize an index we already know is wrong.
            self._verify_index_provenance()
            logger.info("Loading existing vectorstore from disk")
            self._load_vectorstore()
            if not self._embeddings_match_index():
                logger.warning(
                    "Vectorstore dimension mismatch detected — rebuilding index "
                    "with current embedding model (%s)",
                    self.embedding_model_name,
                )
                needs_build = True

        if needs_build:
            reason = "force_rebuild=True" if force_rebuild else "rebuild/no existing index"
            logger.info("Building vectorstore (%s)", reason)
            self._require_writable_vectorstore_path()
            self._prepare_rebuild()
            docs = self._ingest_documents()
            if not docs:
                logger.warning("No documents ingested — RAG will be non-functional")
                return
            self._build_vectorstore(docs)
            self._write_index_meta(doc_count=len(docs))

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
        # A bound OUTPUT_SCHEMA needs ChatOllama: with_structured_output is
        # not implemented on OllamaLLM, so without this the schema would be
        # silently dropped and every response parsed as a string. The schema
        # also supersedes format="json" — both set the same Ollama parameter.
        structured = self.OUTPUT_SCHEMA is not None
        return _build_llm_central(
            provider=self.llm_provider,
            model=self.model_name,
            temperature=self.temperature,
            ollama_base_url=self.ollama_base_url,
            openai_api_key=self.openai_api_key or "",
            groq_api_key=self.groq_api_key or "",
            json_mode=bool(self.llm_format) and not structured,
            chat_model=structured,
        )


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
    def _ingest_documents(self) -> list[Document]:
        """Load all raw data and return a list of LangChain Documents."""

    @abstractmethod
    def _ingest_single_file(self, file_path: str) -> list[Document]:
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

    def _require_writable_vectorstore_path(self) -> None:
        """Fail fast, and legibly, when an index rebuild cannot be written.

        docker-compose mounts vector_databases read-only. Without this guard a
        missing or stale index makes startup re-ingest the whole corpus and then
        die inside save_local with a bare OSError, minutes later.
        """
        path = self.vectorstore_path
        probe = path if path.exists() else path.parent
        if os.access(probe, os.W_OK):
            return
        raise RAGError(
            f"{self.__class__.__name__} needs to build its vector index at '{path}', "
            f"but that path is read-only. Either build the index in a writable "
            f"environment and ship it, or mount the directory read-write "
            f"(docker-compose.yml mounts ./vector_databases with ':ro')."
        )

    # ---------------- index provenance ----------------

    def _meta_dir(self) -> Path:
        """Directory holding this RAG's index_meta.json. Override if the layout differs."""
        return self.vectorstore_path

    def _source_files(self) -> list[Path]:
        """Source data this index was built from. Override to enable staleness checks."""
        return []

    def _index_dimension(self) -> int | None:
        """Embedding dimension of the built index, read from FAISS itself."""
        if self.vectorstore is None:
            return None
        return int(self.vectorstore.index.d)

    def _verify_index_provenance(self) -> None:
        """Refuse to load an index built by a different embedding model or chunking.

        A same-dimension model swap (bge-m3 -> another 1024-d model) loads fine
        and returns near-random neighbours, which the dimension probe cannot
        detect. Failing loudly beats degrading silently.

        Indexes built before provenance existed have no sidecar; those are
        allowed through with a warning so upgrades don't break.
        """
        existing = read_meta(self._meta_dir())
        if existing is None:
            logger.warning(
                "%s index at %s has no %s — it predates provenance tracking. "
                "Run `nutriai-index rebuild` to stamp it; until then a silent "
                "embedding-model mismatch cannot be detected.",
                self.__class__.__name__, self._meta_dir(), META_FILENAME,
            )
            return

        reason = describe_mismatch(
            existing,
            embedding_model=self.embedding_model_name,
            chunk_size=getattr(self, "chunk_size", None),
            chunk_overlap=getattr(self, "chunk_overlap", None),
            source_files=self._source_files(),
        )
        if reason:
            raise RAGError(
                f"{self.__class__.__name__} index at '{self._meta_dir()}' is stale: "
                f"{reason}. Retrieval would silently return wrong results, so the "
                f"index will not be loaded. Rebuild it with `nutriai-index rebuild`."
            )

        changed = stale_sources(existing, self._source_files())
        if changed:
            logger.warning(
                "%s index is valid but its source data changed since %s: %s. "
                "Run `nutriai-index rebuild` to pick up the new content.",
                self.__class__.__name__, existing.built_at, ", ".join(changed),
            )

    def _write_index_meta(self, doc_count: int) -> None:
        """Stamp provenance beside the index just built."""
        dim = self._index_dimension()
        if dim is None:
            logger.warning("Could not determine index dimension — skipping provenance")
            return
        write_meta(
            self._meta_dir(),
            IndexMeta.build(
                embedding_model=self.embedding_model_name,
                embedding_dim=dim,
                doc_count=doc_count,
                rag_class=self.__class__.__name__,
                chunk_size=getattr(self, "chunk_size", None),
                chunk_overlap=getattr(self, "chunk_overlap", None),
                source_files=self._source_files(),
            ),
        )

    def _index_dim_ok(self, vectorstore) -> bool:
        """True if `vectorstore` accepts a query from the current embeddings.

        A dimension mismatch (e.g. after switching EMBEDDING_MODEL) makes FAISS
        raise an AssertionError on search — that signals a rebuild is needed.
        """
        try:
            vectorstore.similarity_search("dimension check", k=1)
            return True
        except AssertionError:
            return False
        except Exception:
            # Unrelated error (e.g. an empty index) — don't trigger a rebuild loop.
            return True

    def _embeddings_match_index(self) -> bool:
        """Hook: True if the loaded index matches the current embedding model."""
        if self.vectorstore is None:
            return True
        return self._index_dim_ok(self.vectorstore)

    def _prepare_rebuild(self) -> None:  # noqa: B027 - optional hook, not abstract
        """Hook: wipe stale on-disk artifacts before a (re)build.

        Default is a no-op — `_build_vectorstore` overwrites the single store.
        Subclasses with multiple stores / resumable ingestion override this.
        """

    def _build_vectorstore(self, documents: list[Document]) -> None:
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

    def _merge_into_vectorstore(self, documents: list[Document]) -> None:
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
        if self.use_hybrid or self.reranker is not None:
            from infrastructure.rag.retrieval import HybridRetriever

            self.retriever = HybridRetriever(
                self.vectorstore,
                final_k=self.rerank_top_n,
                fetch_k=self.fetch_k,
                use_hybrid=self.use_hybrid,
                reranker=self.reranker,
            )
            logger.info(
                "HybridRetriever active (hybrid=%s, rerank=%s, fetch_k=%d, top_n=%d)",
                self.use_hybrid, self.reranker is not None, self.fetch_k, self.rerank_top_n,
            )
        else:
            self.retriever = self.vectorstore.as_retriever()

    # ================================================================
    # Chain building
    # ================================================================

    @staticmethod
    def _format_docs(docs: list[Document]) -> str:
        """Join retrieved documents into the prompt's ``{context}`` string.

        Matches the default document separator used by the legacy
        ``create_stuff_documents_chain`` (a blank line between chunks).
        """
        return "\n\n".join(doc.page_content for doc in docs)

    def _build_chain(self) -> None:
        """Assemble the retrieval-augmented LCEL chain.

        Replaces the legacy ``create_retrieval_chain`` /
        ``create_stuff_documents_chain`` helpers (removed from the core
        ``langchain`` package in 1.0). The contract is preserved: invoke with
        ``{"input": query}`` and read the ``"answer"`` key off the result, which
        also carries the retrieved ``"context"`` documents.
        """
        if self.retriever is None:
            logger.warning("No retriever available — chain not built")
            return

        system_prompt = self._get_system_prompt()
        prompt = ChatPromptTemplate.from_template(system_prompt)

        # input dict -> retrieved documents (retrievers expect the query string)
        retrieve_docs: Runnable = RunnableLambda(lambda x: x["input"]) | self.retriever

        # {input, context: [Document]} -> answer.
        #
        # When a subclass declares OUTPUT_SCHEMA, bind it through the provider's
        # native structured-output API instead of parsing a free-form string.
        # format="json" only guarantees syntax; a schema guarantees shape, which
        # is what small local models actually get wrong.
        generate: Runnable = self.llm | StrOutputParser()
        if self.OUTPUT_SCHEMA is not None:
            try:
                generate = self.llm.with_structured_output(self.OUTPUT_SCHEMA)
            except (NotImplementedError, AttributeError) as e:
                # A provider without structured-output support must not take the
                # whole app down at startup. Subclasses keep a string-parsing
                # fallback for exactly this case.
                logger.warning(
                    "%s does not support structured output (%s) — falling back to "
                    "string parsing for %s",
                    type(self.llm).__name__, e, self.__class__.__name__,
                )

        answer = (
            RunnableLambda(
                lambda x: {"input": x["input"], "context": self._format_docs(x["context"])}
            )
            | prompt
            | generate
        )

        self.rag_chain = RunnablePassthrough.assign(
            context=retrieve_docs
        ) | RunnablePassthrough.assign(answer=answer)

    # ================================================================
    # Indexed file tracking
    # ================================================================

    def _index_meta_path(self) -> Path:
        return self.vectorstore_path.parent / f"{self.vectorstore_path.name}_indexed.json"

    def _get_indexed_files(self) -> set[str]:
        meta_path = self._index_meta_path()
        if not meta_path.exists():
            return set()
        with open(meta_path, encoding="utf-8") as f:
            data = json.load(f)
        return set(data.get("files", []))

    def _save_indexed_files(self, files: set[str]) -> None:
        meta_path = self._index_meta_path()
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({"files": sorted(files)}, f, indent=2)

    # ================================================================
    # Utilities
    # ================================================================


