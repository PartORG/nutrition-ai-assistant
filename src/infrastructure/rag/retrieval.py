"""
infrastructure.rag.retrieval - Hybrid retrieval + cross-encoder reranking.

Upgrades the RAG retrieval layer from naive dense-only similarity search to:

  1. Hybrid search  — dense (FAISS) + sparse (BM25), fused with Reciprocal
                      Rank Fusion (RRF). Catches both semantic matches and
                      exact keyword/term matches (ingredient names, conditions).
  2. Reranking      — a cross-encoder re-scores the fused candidates so the
                      most relevant chunks reach the LLM first. This is the
                      single highest-impact RAG quality lever.

Everything degrades gracefully: if BM25 (rank_bm25) or the cross-encoder model
is unavailable (e.g. offline, model not downloaded), the component logs a
warning and falls back to dense-only / unreranked order. Retrieval never
hard-fails because of these additions.

These helpers live entirely behind RecipeRAGPort / MedicalRAGPort — no
application, domain, or agent code changes are required to use them.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence

from langchain_core.documents import Document
from langchain_core.runnables import Runnable, RunnableConfig

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


# ---------------------------------------------------------------------------
# Cross-encoder reranker
# ---------------------------------------------------------------------------

class Reranker:
    """Re-scores (query, document) pairs with a cross-encoder.

    The embedding model is a *bi-encoder*: it encodes query and document
    independently, so it can miss subtle relevance. A *cross-encoder* reads the
    query and document together and is far more accurate at judging relevance.
    We run it as a high-precision second pass over the hybrid-search candidates.

    The model is loaded lazily on first use and cached. Any failure (missing
    model, no network for the one-time download, etc.) is caught and the
    original ordering is returned, so retrieval never crashes.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_n: int = 8,
        max_pairs: int = 128,
    ):
        self._model_name = model_name
        self._top_n = top_n
        self._max_pairs = max_pairs
        self._encoder = None
        self._failed = False

    def _ensure_loaded(self) -> bool:
        if self._encoder is not None:
            return True
        if self._failed:
            return False
        try:
            from langchain_community.cross_encoders import HuggingFaceCrossEncoder

            logger.info("Loading cross-encoder reranker: %s", self._model_name)
            self._encoder = HuggingFaceCrossEncoder(model_name=self._model_name)
            return True
        except Exception as e:  # noqa: BLE001 - degrade, never crash retrieval
            logger.warning(
                "Reranker model '%s' unavailable (%s) — reranking disabled",
                self._model_name, e,
            )
            self._failed = True
            return False

    def rerank(
        self,
        query: str,
        documents: Sequence[Document],
        top_n: int | None = None,
    ) -> list[Document]:
        """Return the `top_n` documents most relevant to `query`."""
        docs = list(documents)
        n = top_n or self._top_n
        if not docs:
            return docs
        if not self._ensure_loaded():
            return docs[:n]
        try:
            pairs = [(query, d.page_content) for d in docs[: self._max_pairs]]
            scores = self._encoder.score(pairs)
            ranked = sorted(
                zip(docs, scores, strict=False), key=lambda pair: pair[1], reverse=True,
            )
            return [doc for doc, _ in ranked[:n]]
        except Exception as e:  # noqa: BLE001
            logger.warning("Reranking failed (%s) — using fused order", e)
            return docs[:n]


# ---------------------------------------------------------------------------
# Sparse (BM25) + fusion helpers
# ---------------------------------------------------------------------------

def docs_from_vectorstore(vectorstore) -> list[Document]:
    """Extract all stored Documents from a FAISS vectorstore's in-memory docstore.

    Lets us build a BM25 index from the same corpus that was embedded, without
    re-reading the source files or changing the ingestion path.
    """
    store = getattr(vectorstore, "docstore", None)
    mapping = getattr(store, "_dict", None)
    if not mapping:
        return []
    return list(mapping.values())


def bm25_tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens.

    BM25Retriever's default preprocessor is bare `text.split()`, which keeps
    case and glues punctuation on: a recipe indexes as "Chicken", "Oil," and
    "chopped.", so a lowercase query matches almost nothing. Sparse retrieval
    was returning near-noise, and because RRF weights every input list equally
    that noise was actively displacing good dense hits — hybrid scored *below*
    dense-only on the eval before this.
    """
    return _TOKEN_RE.findall(text.lower())


def build_bm25(documents: Sequence[Document], k: int):
    """Build an in-memory BM25 retriever from documents. Returns None on failure."""
    docs = list(documents)
    if not docs:
        return None
    try:
        from langchain_community.retrievers import BM25Retriever

        retriever = BM25Retriever.from_documents(docs, preprocess_func=bm25_tokenize)
        retriever.k = k
        logger.info("BM25 retriever built over %d documents (k=%d)", len(docs), k)
        return retriever
    except Exception as e:  # noqa: BLE001
        logger.warning("BM25 unavailable (%s) — sparse retrieval disabled", e)
        return None


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[Document]],
    rrf_k: int = 60,
    top_k: int | None = None,
) -> list[Document]:
    """Fuse several ranked document lists via Reciprocal Rank Fusion.

    RRF score for a document = sum over lists of 1 / (rrf_k + rank). It needs no
    score calibration between the dense and sparse retrievers (their scores are
    on different scales), which is why RRF is the standard fusion method for
    hybrid search.
    """
    scores: dict[str, float] = {}
    doc_by_key: dict[str, Document] = {}
    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked):
            key = doc.page_content
            doc_by_key.setdefault(key, doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)
    fused = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    docs = [doc_by_key[key] for key, _ in fused]
    return docs[:top_k] if top_k else docs


def hybrid_search(vectorstore, query: str, fetch_k: int, bm25) -> list[Document]:
    """Dense + (optional) BM25 retrieval over one store, fused via RRF.

    Falls back to dense-only if BM25 is None or fusion fails.
    """
    dense = vectorstore.similarity_search(query, k=fetch_k)
    if bm25 is None:
        return dense
    try:
        sparse = bm25.invoke(query)
    except Exception as e:  # noqa: BLE001
        logger.warning("BM25 query failed (%s) — dense only", e)
        return dense
    return reciprocal_rank_fusion([dense, sparse], top_k=fetch_k)


# ---------------------------------------------------------------------------
# Hybrid retriever (single FAISS store) — used by BaseRAG / MedicalRAG
# ---------------------------------------------------------------------------

class HybridRetriever(Runnable):
    """Dense + BM25 hybrid retrieval over one FAISS store, with optional rerank.

    `invoke(query)` returns a `List[Document]`, so it is a drop-in replacement
    for `vectorstore.as_retriever()` inside `create_retrieval_chain` (which
    calls the retriever with the query string).
    """

    def __init__(
        self,
        vectorstore,
        final_k: int = 8,
        fetch_k: int = 30,
        use_hybrid: bool = True,
        reranker: Reranker | None = None,
    ):
        self.vectorstore = vectorstore
        self.final_k = final_k
        self.fetch_k = fetch_k
        self.use_hybrid = use_hybrid
        self.reranker = reranker
        self._bm25 = None
        self._bm25_built = False

    def _get_bm25(self):
        """Lazily build the BM25 index on first query (keeps startup fast)."""
        if not self.use_hybrid:
            return None
        if not self._bm25_built:
            self._bm25 = build_bm25(docs_from_vectorstore(self.vectorstore), self.fetch_k)
            self._bm25_built = True
        return self._bm25

    def invoke(self, input, config: RunnableConfig = None) -> list[Document]:  # noqa: A002
        query = input.get("input", "") if isinstance(input, dict) else input
        return self.retrieve(query)

    def retrieve(self, query: str) -> list[Document]:
        candidates = hybrid_search(self.vectorstore, query, self.fetch_k, self._get_bm25())
        if self.reranker is not None:
            return self.reranker.rerank(query, candidates, self.final_k)
        return candidates[: self.final_k]
