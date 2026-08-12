"""
infrastructure.rag.smart_retriever - Intelligent query routing across vectorstores.

Routes queries to the appropriate vectorstore (recipes/nutrition/both) based on
keyword matching, then retrieves with hybrid search (dense + BM25 via RRF) and
an optional cross-encoder rerank over the merged candidates.

Hybrid + rerank are controlled by the caller; when disabled (or when the
underlying models/libraries are unavailable) it degrades to plain dense
similarity search, preserving the original behaviour.
"""

from __future__ import annotations

import logging
from typing import List, Optional

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
    """Routes queries to appropriate vector collections, then hybrid-retrieves."""

    def __init__(
        self,
        vectorstore_recipes,
        vectorstore_nutrition,
        k: int = 14,
        use_hybrid: bool = False,
        reranker: Optional[Reranker] = None,
        fetch_k: Optional[int] = None,
    ):
        self.vectorstore_recipes = vectorstore_recipes
        self.vectorstore_nutrition = vectorstore_nutrition
        self.k = k
        self.use_hybrid = use_hybrid
        self.reranker = reranker
        # Pull a larger candidate pool than `k` so the reranker has room to work.
        self.fetch_k = fetch_k or max(k * 2, 30)
        self._bm25: dict[str, object] = {}
        self._bm25_built: set[str] = set()

    def invoke(
        self, input: dict | str, config: RunnableConfig = None,  # noqa: A002
    ) -> List[Document]:
        query = input.get("input", "") if isinstance(input, dict) else input
        return self._smart_retrieve(query)

    def _bm25_for(self, name: str, vectorstore):
        """Lazily build (and cache) a BM25 index for a store on first use."""
        if not self.use_hybrid:
            return None
        if name not in self._bm25_built:
            self._bm25[name] = build_bm25(docs_from_vectorstore(vectorstore), self.fetch_k)
            self._bm25_built.add(name)
        return self._bm25.get(name)

    def _retrieve_store(self, name: str, vectorstore, query: str, fetch_k: int) -> List[Document]:
        return hybrid_search(vectorstore, query, fetch_k, self._bm25_for(name, vectorstore))

    def _smart_retrieve(self, query: str) -> List[Document]:
        query_type = self._determine_query_type(query)

        if query_type == "recipes":
            results = self._retrieve_store(
                "recipes", self.vectorstore_recipes, query, self.fetch_k,
            )
        elif query_type == "nutrition":
            results = self._retrieve_store(
                "nutrition", self.vectorstore_nutrition, query, self.fetch_k,
            )
        else:
            half = self.fetch_k // 2 + 1
            results_recipes = self._retrieve_store(
                "recipes", self.vectorstore_recipes, query, half,
            )
            results_nutrition = self._retrieve_store(
                "nutrition", self.vectorstore_nutrition, query, half,
            )
            results = results_recipes + results_nutrition

        # Rerank the merged candidates so the best chunks across both stores win.
        if self.reranker is not None:
            return self.reranker.rerank(query, results, self.k)
        return results[: self.k]

    @staticmethod
    def _determine_query_type(query: str) -> str:
        query_lower = query.lower()

        recipe_keywords = [
            "recipe", "meal", "cook", "prepare", "make", "dish",
            "breakfast", "lunch", "dinner", "snack",
            "vegetarian", "vegan", "keto", "paleo",
            "cuisine", "italian", "chinese", "indian",
        ]
        nutrition_keywords = [
            "nutrition", "nutrient", "vitamin", "mineral",
            "calorie", "protein", "carb", "fat", "fiber",
            "healthy", "good source", "rich in",
            "ingredient", "food",
        ]

        recipe_match = any(kw in query_lower for kw in recipe_keywords)
        nutrition_match = any(kw in query_lower for kw in nutrition_keywords)

        if recipe_match and not nutrition_match:
            return "recipes"
        elif nutrition_match and not recipe_match:
            return "nutrition"
        return "both"
