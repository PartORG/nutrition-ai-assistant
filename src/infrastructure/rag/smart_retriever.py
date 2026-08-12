"""
infrastructure.rag.smart_retriever - Recipe retrieval with hybrid search + rerank.

Retrieves from the recipe corpus with dense (FAISS) + sparse (BM25) search fused
by RRF, then an optional cross-encoder rerank over the candidates.

This class used to also route between a recipe store and a nutrition store based
on keyword matching. That routing is gone with the nutrition FAISS index: food
composition is now an exact SQL lookup (see nutrition_facts_repo), so the whole
fetch_k budget goes to recipes. Previously the router matched "both" for almost
every query — the augmented query always contains "Nutrition Guidelines" — and
spent half the candidate pool on nutrition rows that crowded recipes out of a
small model's context.

Hybrid and rerank are controlled by the caller; when disabled, or when the
underlying models are unavailable, this degrades to plain dense similarity
search.
"""

from __future__ import annotations

import logging

from langchain_core.documents import Document
from langchain_core.runnables import Runnable, RunnableConfig

from infrastructure.rag.retrieval import (
    Reranker,
    build_bm25,
    docs_from_vectorstore,
    hybrid_search,
)

logger = logging.getLogger(__name__)


class SmartRetriever(Runnable):
    """Hybrid dense+sparse retrieval over the recipe corpus, with optional rerank."""

    def __init__(
        self,
        vectorstore_recipes,
        k: int = 14,
        use_hybrid: bool = False,
        reranker: Reranker | None = None,
        fetch_k: int | None = None,
    ):
        self.vectorstore_recipes = vectorstore_recipes
        self.k = k
        self.use_hybrid = use_hybrid
        self.reranker = reranker
        # Pull a larger candidate pool than `k` so the reranker has room to work.
        self.fetch_k = fetch_k or max(k * 2, 30)
        self._bm25 = None
        self._bm25_built = False

    def warm_up(self) -> None:
        """Build the BM25 index now rather than inside the first user request.

        Tokenising the corpus takes seconds; doing it lazily put that entirely
        on whoever asked the first question after a restart.
        """
        if self.use_hybrid:
            self._bm25_index()

    def invoke(
        self, input: dict | str, config: RunnableConfig = None,  # noqa: A002
    ) -> list[Document]:
        query = input.get("input", "") if isinstance(input, dict) else input
        return self._retrieve(query)

    def _bm25_index(self):
        """Lazily build (and cache) the BM25 index on first use."""
        if not self.use_hybrid:
            return None
        if not self._bm25_built:
            self._bm25 = build_bm25(
                docs_from_vectorstore(self.vectorstore_recipes), self.fetch_k,
            )
            self._bm25_built = True
        return self._bm25

    def _retrieve(self, query: str) -> list[Document]:
        results = hybrid_search(
            self.vectorstore_recipes, query, self.fetch_k, self._bm25_index(),
        )
        if self.reranker is not None:
            return self.reranker.rerank(query, results, self.k)
        return results[: self.k]
