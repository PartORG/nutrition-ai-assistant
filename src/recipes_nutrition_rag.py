"""
RecipesNutritionRAG - Production-Ready RAG System for Recipe Recommendations

This module provides a complete RAG (Retrieval-Augmented Generation) system for
personalized recipe recommendations based on nutritional requirements.

Features:
- Dual vectorstore system (recipes + nutrition facts)
- Smart query routing (recipes/nutrition/both)
- Intelligent retrieval with configurable k
- Medical-grade system prompt with complete recipe structure
- Auto-detects and loads/builds vectorstores
"""

# ========================================
# IMPORTS
# ========================================

# Core Libraries
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# LangChain Components
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy
from langchain_ollama import OllamaLLM
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.runnables import Runnable, RunnableConfig

# Python Standard Library
import ast
import re
from pathlib import Path
from typing import List, Dict, Any, Optional


# ========================================
# HELPER FUNCTIONS
# ========================================

def determine_query_type(query: str) -> str:
    """
    Intelligently route queries to the right collection.

    Args:
        query: User's search query

    Returns:
        'recipes', 'nutrition', or 'both'
    """
    query_lower = query.lower()

    recipe_keywords = [
        'recipe', 'meal', 'cook', 'prepare', 'make', 'dish', 
        'breakfast', 'lunch', 'dinner', 'snack',
        'vegetarian', 'vegan', 'keto', 'paleo',
        'cuisine', 'italian', 'chinese', 'indian'
    ]

    nutrition_keywords = [
        'nutrition', 'nutrient', 'vitamin', 'mineral', 
        'calorie', 'protein', 'carb', 'fat', 'fiber',
        'healthy', 'good source', 'rich in',
        'ingredient', 'food'
    ]

    recipe_match = any(keyword in query_lower for keyword in recipe_keywords)
    nutrition_match = any(keyword in query_lower for keyword in nutrition_keywords)

    if recipe_match and not nutrition_match:
        return 'recipes'
    elif nutrition_match and not recipe_match:
        return 'nutrition'
    else:
        return 'both'


def smart_retrieve(query: str, vectorstore_recipes, vectorstore_nutrition, k: int = 10) -> List[Document]:
    """
    Smart retrieval across collections based on query type.

    Args:
        query: User's search query
        vectorstore_recipes: FAISS vectorstore for recipes
        vectorstore_nutrition: FAISS vectorstore for nutrition facts
        k: Number of documents to retrieve

    Returns:
        List of retrieved Document objects
    """
    query_type = determine_query_type(query)

    print(f"🔍 Query type detected: {query_type.upper()}")

    if query_type == 'recipes':
        results = vectorstore_recipes.similarity_search(query, k=k)
        print(f"   → Searched RECIPES_AND_MEALS collection")
    elif query_type == 'nutrition':
        results = vectorstore_nutrition.similarity_search(query, k=k)
        print(f"   → Searched NUTRITION_FACTS collection")
    else:
        results_recipes = vectorstore_recipes.similarity_search(query, k=k//2 + 1)
        results_nutrition = vectorstore_nutrition.similarity_search(query, k=k//2 + 1)
        results = results_recipes + results_nutrition
        print(f"   → Searched BOTH collections")

    return results[:k]


# ========================================
# SMART RETRIEVER CLASS
# ========================================

class SmartRetriever(Runnable):
    """
    Intelligent retriever implementing LangChain's Runnable interface.

    Routes queries to appropriate vector collections based on content analysis.
    """

    def __init__(self, vectorstore_recipes, vectorstore_nutrition, k=10):
        """
        Initialize SmartRetriever.

        Args:
            vectorstore_recipes: FAISS vectorstore for recipes
            vectorstore_nutrition: FAISS vectorstore for nutrition facts
            k: Number of documents to retrieve
        """
        self.vectorstore_recipes = vectorstore_recipes
        self.vectorstore_nutrition = vectorstore_nutrition
        self.k = k

    def invoke(self, input: dict | str, config: RunnableConfig = None) -> List[Document]:
        """
        Execute smart retrieval based on query analysis.

        Args:
            input: Query string or dict with 'input' key
            config: Optional Runnable configuration

        Returns:
            List of retrieved Document objects
        """
        if isinstance(input, dict):
            query = input.get("input", "")
        else:
            query = input

        return smart_retrieve(
            query=query,
            vectorstore_recipes=self.vectorstore_recipes,
            vectorstore_nutrition=self.vectorstore_nutrition,
            k=self.k
        )


# ========================================
# MAIN RAG CLASS
# ========================================

class RecipesNutritionRAG:
    """
    Production-ready RAG system for personalized recipe recommendations.

    Features:
    - Dual vectorstore system (recipes + nutrition facts)
    - Smart query routing (recipes/nutrition/both)
    - Intelligent retrieval with configurable k
    - Medical-grade system prompt with full recipe structure
    - Auto-detects and loads/builds vectorstores

    Usage:
        rag = RecipesNutritionRAG(
            data_folder="data/",
            vectorstore_path="vector_databases/",
            model_name="llama3.2",
            temperature=0.5,
            k=10
        )

        rag.initialize(force_rebuild=False)
        response = rag.query("vegetarian high-protein meal under 500 calories")
    """

    SYSTEM_PROMPT = """You are NutriGuide, an AI nutrition assistant providing personalized recipe recommendations.

## CRITICAL SAFETY DISCLAIMER
You are a recommendation system ONLY. Your suggestions do NOT replace professional medical advice from healthcare providers.

## STRICT OUTPUT REQUIREMENTS

For EVERY recipe recommendation, you MUST include ALL of the following sections in this exact order:

### MANDATORY SECTIONS (DO NOT SKIP ANY):

**1. Recipe Name** (Adapted if modified)

**2. Why This Recipe:**
- Meets calorie/protein requirements
- Dietary compliance (vegetarian, vegan, etc.)
- Medical alignment (if applicable)

**3. Adaptations Made:** (if any)
- State "No adaptations needed" if recipe matches perfectly
- OR list: Original → Modified → Reason

**4. Nutritional Information (per serving):**
- Calories: X kcal
- Protein: X g
- Carbohydrates: X g  
- Fat: X g
- Fiber: X g (if relevant)
- Sodium: X mg (if relevant)

**5. Ingredients (CRITICAL - NEVER SKIP):**
**ALWAYS extract and list ingredients from the retrieved context.**
**If ingredient quantities are missing in context, you MUST:**
- Estimate reasonable quantities based on the serving size
- Mark estimates with (approximately)
- Convert ALL measurements to metric: grams (g), milliliters (ml)
- Format: `- XXXg ingredient name` or `- XXml liquid name`

**6. Cooking Instructions (CRITICAL - NEVER SKIP):**
**ALWAYS extract and provide step-by-step instructions from the retrieved context.**
**If instructions are missing, you MUST:**
- Create logical cooking steps based on the ingredients
- Include temperatures in Celsius (°C)
- Number each step clearly

**7. Time Information:**
- Preparation Time: X minutes
- Cooking Time: X minutes  
- Total Time: X minutes

---

## HANDLING MISSING DATA

**If retrieved context lacks ingredient quantities:**
→ You MUST estimate based on:
- Serving size (e.g., 325g serving = ~300-350g total ingredients)
- Standard recipe proportions
- Mark as "(approximately)" or "(estimated for 1 serving)"

**If retrieved context lacks cooking instructions:**
→ You MUST create logical steps based on:
- Ingredient types (raw → needs cooking)
- Preparation method stated (Baked, Fried, Raw, etc.)
- Standard cooking techniques

**NEVER say:** "Cooking instructions not available in database"  
**ALWAYS provide:** Complete, usable recipe instructions

---

## MEASUREMENT CONVERSIONS (STRICT)

**Convert ALL measurements to metric:**
- 1 cup → 240 ml
- 1 tbsp → 15 ml
- 1 tsp → 5 ml
- 1 oz → 28 g
- 1 lb → 454 g

**Temperatures MUST be Celsius:**
- 350°F → 175°C
- 400°F → 200°C

---

## YOUR TASK NOW:

User Query: {input}

Retrieved Context: {context}

Generate 3 complete recipe recommendations following the MANDATORY SECTIONS structure above.
**DO NOT skip Ingredients or Cooking Instructions sections.**
**If data is missing, estimate based on serving size and recipe type.**

⚠️ **Important Reminder**: These are suggestions based on general nutrition principles. Consult healthcare providers before dietary changes."""

    def __init__(
        self,
        data_folder: str,
        vectorstore_path: str,
        model_name: str = "llama3.2",
        temperature: float = 0.5,
        k: int = 10,
        ollama_base_url: str = "http://localhost:11434/"
    ):
        """Initialize RecipesNutritionRAG (call initialize() to load data)."""
        self.data_folder = Path(data_folder)
        self.vectorstore_path = Path(vectorstore_path)
        self.model_name = model_name
        self.temperature = temperature
        self.k = k
        self.ollama_base_url = ollama_base_url

        self.embeddings = None
        self.vectorstore_recipes = None
        self.vectorstore_nutrition = None
        self.smart_retriever = None
        self.llm = None
        self.rag_chain = None

        print(f"✅ RecipesNutritionRAG created")
        print(f"   Model: {self.model_name} | Temperature: {self.temperature} | k: {self.k}")

    def initialize(self, force_rebuild: bool = False) -> None:
        """Initialize the RAG system."""
        print("\n🚀 Initializing RecipesNutritionRAG...")

        # Load embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name='sentence-transformers/all-mpnet-base-v2',
            encode_kwargs={"normalize_embeddings": True}
        )

        # Load vectorstores
        recipes_db_path = self.vectorstore_path / "recipes_and_meals_db"
        nutrition_db_path = self.vectorstore_path / "nutrition_facts_db"

        if not force_rebuild and recipes_db_path.exists() and nutrition_db_path.exists():
            print("📂 Loading existing vectorstores...")
            self.vectorstore_recipes = FAISS.load_local(
                folder_path=str(recipes_db_path),
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True
            )
            self.vectorstore_nutrition = FAISS.load_local(
                folder_path=str(nutrition_db_path),
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            raise FileNotFoundError("Vectorstores not found. Build them first using the notebook.")

        # Create retriever
        self.smart_retriever = SmartRetriever(
            vectorstore_recipes=self.vectorstore_recipes,
            vectorstore_nutrition=self.vectorstore_nutrition,
            k=self.k
        )

        # Initialize LLM
        self.llm = OllamaLLM(
            model=self.model_name,
            temperature=self.temperature,
            base_url=self.ollama_base_url
        )

        # Build chain
        prompt_template = ChatPromptTemplate.from_template(self.SYSTEM_PROMPT)
        stuff_documents_chain = create_stuff_documents_chain(
            llm=self.llm,
            prompt=prompt_template
        )
        self.rag_chain = create_retrieval_chain(
            retriever=self.smart_retriever,
            combine_docs_chain=stuff_documents_chain
        )

        print("✅ Initialization complete!")

    def query(self, user_input: str) -> str:
        """Get recipe recommendations."""
        if not self.rag_chain:
            return "❌ System not initialized. Call initialize() first."

        response = self.rag_chain.invoke({"input": user_input})
        return response.get("answer", "No response generated.")

    def get_retrieved_docs(self, query: str) -> List[Document]:
        """Debug method - see retrieved documents."""
        if not self.smart_retriever:
            print("❌ System not initialized.")
            return []
        return self.smart_retriever.invoke(query)

    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        if not self.vectorstore_recipes or not self.vectorstore_nutrition:
            return {"status": "not_initialized"}

        return {
            "status": "initialized",
            "model": self.model_name,
            "temperature": self.temperature,
            "k": self.k,
            "vectorstores": {
                "recipes_and_meals": {
                    "vectors": self.vectorstore_recipes.index.ntotal
                },
                "nutrition_facts": {
                    "vectors": self.vectorstore_nutrition.index.ntotal
                }
            }
        }
