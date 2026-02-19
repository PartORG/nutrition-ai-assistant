"""
infrastructure.rag.smart_retriever - Intelligent query routing across vectorstores.

Extracted from recipes_nutrition_rag.py into its own module.
Routes queries to the appropriate vectorstore (recipes/nutrition/both)
based on keyword matching.
"""

from __future__ import annotations

import logging
from typing import List

from langchain.schema import Document
from langchain_core.runnables import Runnable, RunnableConfig

logger = logging.getLogger(__name__)


class SmartRetriever(Runnable):
    """Routes queries to appropriate vector collections based on content."""

    def __init__(self, vectorstore_recipes, vectorstore_nutrition, k: int = 14):
        self.vectorstore_recipes = vectorstore_recipes
        self.vectorstore_nutrition = vectorstore_nutrition
        self.k = k

    def invoke(self, input: dict | str, config: RunnableConfig = None) -> List[Document]:
        if isinstance(input, dict):
            query = input.get("input", "")
        else:
            query = input
        return self._smart_retrieve(query)

    def _smart_retrieve(self, query: str) -> List[Document]:
        query_type = self._determine_query_type(query)

        if query_type == "recipes":
            results = self.vectorstore_recipes.similarity_search(query, k=self.k)
        elif query_type == "nutrition":
            results = self.vectorstore_nutrition.similarity_search(query, k=self.k)
        else:
            results_recipes = self.vectorstore_recipes.similarity_search(query, k=self.k // 2 + 1)
            results_nutrition = self.vectorstore_nutrition.similarity_search(query, k=self.k // 2 + 1)
            results = results_recipes + results_nutrition

        return results[:self.k]

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
