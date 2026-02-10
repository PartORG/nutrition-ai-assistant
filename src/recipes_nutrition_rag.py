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
import logging
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

# Setup module logger
logger = logging.getLogger(__name__)


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
        query_type = 'recipes'
    elif nutrition_match and not recipe_match:
        query_type = 'nutrition'
    else:
        query_type = 'both'
    
    logger.debug(f"Query type detected: {query_type.upper()}")
    return query_type


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

    if query_type == 'recipes':
        results = vectorstore_recipes.similarity_search(query, k=k)
        logger.debug("Searched RECIPES_AND_MEALS collection")
    elif query_type == 'nutrition':
        results = vectorstore_nutrition.similarity_search(query, k=k)
        logger.debug("Searched NUTRITION_FACTS collection")
    else:
        results_recipes = vectorstore_recipes.similarity_search(query, k=k//2 + 1)
        results_nutrition = vectorstore_nutrition.similarity_search(query, k=k//2 + 1)
        results = results_recipes + results_nutrition
        logger.debug("Searched both collections")

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
        logger.debug(f"SmartRetriever initialized with k={k}")

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
        ollama_base_url: str = "http://localhost:11434/",
        log_level: str = "INFO"
    ):
        """
        Initialize RecipesNutritionRAG (call initialize() to load data).
        
        Args:
            data_folder: Path to folder containing CSVs
            vectorstore_path: Path to vectorstore folder
            model_name: Ollama model name
            temperature: LLM temperature (0=deterministic, 1=creative)
            k: Number of documents to retrieve
            ollama_base_url: Ollama server URL
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        """
        # Setup logging
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
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

        logger.info(f"RecipesNutritionRAG instance created (model={self.model_name}, k={self.k})")

    def initialize(self, force_rebuild: bool = False) -> None:
        """
        Initialize the RAG system: load or build vectorstores, create chain.

        Args:
            force_rebuild: If True, rebuild vectorstores from CSVs even if they exist
        """
        logger.info("Initializing RecipesNutritionRAG system")

        # Step 1: Initialize embeddings
        logger.debug("Loading embedding model")
        self.embeddings = HuggingFaceEmbeddings(
            model_name='sentence-transformers/all-mpnet-base-v2',
            encode_kwargs={"normalize_embeddings": True}
        )
        logger.info("Embedding model loaded successfully")

        # Step 2: Check vectorstore existence
        recipes_db_path = self.vectorstore_path / "recipes_and_meals_db"
        nutrition_db_path = self.vectorstore_path / "nutrition_facts_db"

        vectorstores_exist = recipes_db_path.exists() and nutrition_db_path.exists()

        # Step 3 & 4: Load or Build
        if vectorstores_exist and not force_rebuild:
            logger.info("Loading existing vectorstores from disk")
            self._load_vectorstores()
        else:
            if force_rebuild:
                logger.info("force_rebuild=True, building vectorstores from scratch")
            else:
                logger.info("Vectorstores not found, building from CSV files")
            self._build_vectorstores()

        # Step 5: Create SmartRetriever
        logger.debug("Creating SmartRetriever")
        self.smart_retriever = SmartRetriever(
            vectorstore_recipes=self.vectorstore_recipes,
            vectorstore_nutrition=self.vectorstore_nutrition,
            k=self.k
        )
        logger.info(f"SmartRetriever created (k={self.k})")

        # Step 6: Initialize LLM
        logger.debug(f"Initializing Ollama LLM (model={self.model_name})")
        self.llm = OllamaLLM(
            model=self.model_name,
            temperature=self.temperature,
            base_url=self.ollama_base_url
        )

        # Test LLM connection
        try:
            test_response = self.llm.invoke("Say 'OK' if you can read this.")
            logger.info(f"LLM connected successfully (model={self.model_name})")
        except Exception as e:
            logger.error(f"LLM connection failed: {e}")
            raise

        # Step 7: Build RAG chain
        logger.debug("Building RAG chain")
        self._build_chain()
        logger.info("RAG chain created successfully")

        logger.info(
            f"Initialization complete - "
            f"Recipes: {self.vectorstore_recipes.index.ntotal} vectors, "
            f"Nutrition: {self.vectorstore_nutrition.index.ntotal} vectors"
        )

    # ========================================
    # Private Methods: Vectorstore Management
    # ========================================

    def _load_vectorstores(self) -> None:
        """Load existing vectorstores from disk."""
        recipes_db_path = self.vectorstore_path / "recipes_and_meals_db"
        nutrition_db_path = self.vectorstore_path / "nutrition_facts_db"

        logger.debug(f"Loading RECIPES_AND_MEALS from {recipes_db_path}")
        self.vectorstore_recipes = FAISS.load_local(
            folder_path=str(recipes_db_path),
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True
        )

        logger.debug(f"Loading NUTRITION_FACTS from {nutrition_db_path}")
        self.vectorstore_nutrition = FAISS.load_local(
            folder_path=str(nutrition_db_path),
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True
        )

        logger.info(
            f"Vectorstores loaded - "
            f"Recipes: {self.vectorstore_recipes.index.ntotal}, "
            f"Nutrition: {self.vectorstore_nutrition.index.ntotal}"
        )

    def _build_vectorstores(self) -> None:
        """Build vectorstores from CSV files."""
        logger.debug("Loading CSV files")

        # Load all documents
        recipes1 = self._load_recipes_csv(str(self.data_folder / "cleaned_recipes.csv"))
        recipes2 = self._load_recipes_data_sample_csv(str(self.data_folder / "cleaned_recipes_data_sample.csv"))
        meals = self._load_healthy_meals_csv(str(self.data_folder / "cleaned_healthy_meals.csv"))
        nutrition = self._load_nutrition_csv(str(self.data_folder / "cleaned_nutrition.csv"))

        # Combine
        recipes_and_meals_docs = recipes1 + recipes2 + meals
        nutrition_facts_docs = nutrition

        logger.info(
            f"Loaded documents - "
            f"Recipes/Meals: {len(recipes_and_meals_docs)}, "
            f"Nutrition: {len(nutrition_facts_docs)}"
        )

        # Create vectorstores
        logger.debug("Creating FAISS vectorstores (this may take a few minutes)")
        self.vectorstore_recipes = FAISS.from_documents(
            documents=recipes_and_meals_docs,
            embedding=self.embeddings,
            distance_strategy=DistanceStrategy.COSINE
        )

        self.vectorstore_nutrition = FAISS.from_documents(
            documents=nutrition_facts_docs,
            embedding=self.embeddings,
            distance_strategy=DistanceStrategy.COSINE
        )

        # Save to disk
        logger.debug("Saving vectorstores to disk")
        self.vectorstore_path.mkdir(parents=True, exist_ok=True)

        recipes_db_path = self.vectorstore_path / "recipes_and_meals_db"
        nutrition_db_path = self.vectorstore_path / "nutrition_facts_db"

        self.vectorstore_recipes.save_local(str(recipes_db_path))
        self.vectorstore_nutrition.save_local(str(nutrition_db_path))

        logger.info(f"Vectorstores built and saved to {self.vectorstore_path}")

    # ========================================
    # Private Methods: Data Loading
    # ========================================

    def _load_recipes_csv(self, csv_path: str) -> List[Document]:
        """Load cleaned_recipes.csv with structured nutrition parsing."""
        logger.debug(f"Loading {csv_path}")
        df = pd.read_csv(csv_path)
        documents = []

        for idx, row in df.iterrows():
            if pd.isna(row['recipe_name']):
                continue

            # Build text
            text_parts = [
                f"Recipe: {row['recipe_name']}",
                f"\nCuisine: {row.get('cuisine_path', 'Not specified')}",
                f"\nIngredients:\n{row['ingredients']}",
                f"\nDirections:\n{row['directions']}"
            ]

            if pd.notna(row.get('prep_time')):
                text_parts.append(f"\nPrep Time: {row['prep_time']}")
            if pd.notna(row.get('cook_time')):
                text_parts.append(f"\nCook Time: {row['cook_time']}")
            if pd.notna(row.get('nutrition')):
                text_parts.append(f"\nNutrition Facts: {row['nutrition']}")

            full_text = "".join(text_parts)

            # Metadata
            metadata = {
                'doc_type': 'recipe',
                'source_file': 'cleaned_recipes',
                'recipe_name': row['recipe_name'],
                'servings': row.get('servings', 'Not specified'),
            }

            # Parse cuisine
            if pd.notna(row.get('cuisine_path')):
                cuisine = row['cuisine_path'].split('/')[-1] if '/' in str(row['cuisine_path']) else row['cuisine_path']
                metadata['cuisine'] = cuisine

            # Parse timing
            if pd.notna(row.get('prep_time')):
                prep_str = str(row['prep_time']).lower()
                prep_mins = sum([int(s) * (60 if 'hr' in prep_str else 1) 
                               for s in re.findall(r'\d+', prep_str)])
                metadata['prep_time_min'] = prep_mins

            if pd.notna(row.get('cook_time')):
                cook_str = str(row['cook_time']).lower()
                cook_mins = sum([int(s) * (60 if 'hr' in cook_str else 1) 
                               for s in re.findall(r'\d+', cook_str)])
                metadata['cook_time_min'] = cook_mins

            # Extract allergens
            ingredients_lower = str(row['ingredients']).lower()
            allergens = []
            if any(word in ingredients_lower for word in ['milk', 'cheese', 'butter', 'cream', 'yogurt']):
                allergens.append('dairy')
            if any(word in ingredients_lower for word in ['egg']):
                allergens.append('eggs')
            if any(word in ingredients_lower for word in ['wheat', 'flour', 'bread']):
                allergens.append('gluten')
            if any(word in ingredients_lower for word in ['nuts', 'almond', 'peanut', 'walnut']):
                allergens.append('nuts')
            metadata['allergens'] = allergens

            # Diet tags
            diet_tags = []
            if 'vegetarian' in ingredients_lower or 'veggie' in ingredients_lower:
                diet_tags.append('vegetarian')
            if 'vegan' in ingredients_lower:
                diet_tags.append('vegan')
            if not any(meat in ingredients_lower for meat in ['chicken', 'beef', 'pork', 'fish', 'meat']):
                diet_tags.append('vegetarian')
            metadata['diet_tags'] = diet_tags

            documents.append(Document(page_content=full_text, metadata=metadata))

        logger.debug(f"Loaded {len(documents)} documents from {csv_path}")
        return documents

    def _load_recipes_data_sample_csv(self, csv_path: str) -> List[Document]:
        """Load cleaned_recipes_data_sample.csv with NER parsing."""
        logger.debug(f"Loading {csv_path}")
        df = pd.read_csv(csv_path)
        documents = []

        for idx, row in df.iterrows():
            if pd.isna(row['title']):
                continue

            # Parse ingredients
            try:
                ingredients_list = ast.literal_eval(row['ingredients'])
                ingredients_text = "\n".join([f"- {ing}" for ing in ingredients_list])
            except:
                ingredients_text = row['ingredients']

            # Parse directions
            try:
                directions_list = ast.literal_eval(row['directions'])
                directions_text = "\n".join([f"{i+1}. {step}" for i, step in enumerate(directions_list)])
            except:
                directions_text = row['directions']

            # Parse NER
            try:
                ner_list = ast.literal_eval(row['NER'])
                ner_text = ", ".join(ner_list)
            except:
                ner_list = []
                ner_text = ""

            # Build text
            text_parts = [
                f"Recipe: {row['title']}",
                f"\nIngredients:\n{ingredients_text}",
                f"\nDirections:\n{directions_text}",
                f"\nKey Ingredients: {ner_text}"
            ]

            full_text = "".join(text_parts)

            # Metadata
            metadata = {
                'doc_type': 'recipe',
                'source_file': 'cleaned_recipes_data_sample',
                'recipe_name': row['title'],
                'ingredient_list': ner_list if ner_list else None
            }

            # Extract allergens
            ingredients_lower = str(row['ingredients']).lower()
            allergens = []
            if any(word in ingredients_lower for word in ['milk', 'cheese', 'butter', 'cream', 'yogurt']):
                allergens.append('dairy')
            if any(word in ingredients_lower for word in ['egg']):
                allergens.append('eggs')
            if any(word in ingredients_lower for word in ['wheat', 'flour', 'bread']):
                allergens.append('gluten')
            if any(word in ingredients_lower for word in ['nuts', 'almond', 'peanut', 'walnut']):
                allergens.append('nuts')
            metadata['allergens'] = allergens

            # Diet tags
            diet_tags = []
            if not any(meat in ingredients_lower for meat in ['chicken', 'beef', 'pork', 'fish', 'meat', 'lamb']):
                diet_tags.append('vegetarian')
            metadata['diet_tags'] = diet_tags

            documents.append(Document(page_content=full_text, metadata=metadata))

        logger.debug(f"Loaded {len(documents)} documents from {csv_path}")
        return documents

    def _load_healthy_meals_csv(self, csv_path: str) -> List[Document]:
        """Load cleaned_healthy_meals.csv with numeric nutrition metadata."""
        logger.debug(f"Loading {csv_path}")
        df = pd.read_csv(csv_path)
        documents = []

        for idx, row in df.iterrows():
            if pd.isna(row['meal_name']):
                continue

            # Build text
            text = f"""Meal: {row['meal_name']} ({row['cuisine']} {row['meal_type']})
Diet Type: {row['diet_type']}

Nutrition per {row['serving_size_g']}g serving:
- Calories: {row['calories']} kcal
- Protein: {row['protein_g']}g | Carbs: {row['carbs_g']}g | Fat: {row['fat_g']}g
- Fiber: {row['fiber_g']}g | Sugar: {row['sugar_g']}g
- Sodium: {row['sodium_mg']}mg | Cholesterol: {row['cholesterol_mg']}mg

Preparation: {row['cooking_method']} (Prep: {row['prep_time_min']}min, Cook: {row['cook_time_min']}min)
"""

            # Metadata
            metadata = {
                'doc_type': 'meal',
                'source_file': 'cleaned_healthy_meals',
                'recipe_name': row['meal_name'],
                'cuisine': row['cuisine'],
                'meal_type': row['meal_type'],
                'diet_type': row['diet_type'],
                'calories': int(row['calories']),
                'protein_g': float(row['protein_g']),
                'carbs_g': float(row['carbs_g']),
                'fat_g': float(row['fat_g']),
                'fiber_g': float(row['fiber_g']),
                'sugar_g': float(row['sugar_g']),
                'sodium_mg': int(row['sodium_mg']),
                'cholesterol_mg': int(row['cholesterol_mg']),
                'serving_size_g': int(row['serving_size_g']),
                'cooking_method': row['cooking_method'],
                'prep_time_min': int(row['prep_time_min']),
                'cook_time_min': int(row['cook_time_min'])
            }

            # Diet tags
            diet_tags = [row['diet_type'].lower()]
            if row['diet_type'].lower() in ['vegan', 'vegetarian']:
                diet_tags.append('vegetarian')
            metadata['diet_tags'] = diet_tags

            # Allergens
            allergens = []
            meal_lower = row['meal_name'].lower()
            if any(word in meal_lower for word in ['cheese', 'yogurt', 'milk']):
                allergens.append('dairy')
            metadata['allergens'] = allergens

            documents.append(Document(page_content=text, metadata=metadata))

        logger.debug(f"Loaded {len(documents)} documents from {csv_path}")
        return documents

    def _load_nutrition_csv(self, csv_path: str) -> List[Document]:
        """Load cleaned_nutrition.csv - detailed ingredient nutrition database."""
        logger.debug(f"Loading {csv_path}")
        df = pd.read_csv(csv_path)
        documents = []

        for idx, row in df.iterrows():
            if pd.isna(row['name']):
                continue

            # Build text
            text = f"""Ingredient: {row['name']} (per {row['serving_size']})

Macronutrients:
- Calories: {row['calories']} kcal
- Protein: {row['protein']}
- Carbohydrates: {row['carbohydrate']}
- Total Fat: {row['total_fat']}
- Fiber: {row['fiber']}
- Sugars: {row['sugars']}

Key Vitamins:
- Vitamin A: {row['vitamin_a']}
- Vitamin C: {row['vitamin_c']}
- Vitamin D: {row['vitamin_d']}
- Vitamin B12: {row['vitamin_b12']}
- Folate: {row['folate']}

Key Minerals:
- Calcium: {row['calcium']}
- Iron: {row['irom']}
- Magnesium: {row['magnesium']}
- Sodium: {row['sodium']}
- Potassium: {row['potassium']}

Cholesterol: {row['cholesterol']} | Saturated Fat: {row['saturated_fat']}
"""

            # Metadata
            metadata = {
                'doc_type': 'nutrition_fact',
                'source_file': 'cleaned_nutrition',
                'food_name': row['name'],
                'serving_size': row['serving_size']
            }

            # Extract numeric values
            def parse_numeric(val):
                if pd.isna(val):
                    return None
                try:
                    return float(re.sub(r'[^\d.]', '', str(val)))
                except:
                    return None

            metadata['calories'] = parse_numeric(row['calories'])
            metadata['protein_g'] = parse_numeric(row['protein'])
            metadata['carbs_g'] = parse_numeric(row['carbohydrate'])
            metadata['fat_g'] = parse_numeric(row['total_fat'])
            metadata['fiber_g'] = parse_numeric(row['fiber'])
            metadata['sugar_g'] = parse_numeric(row['sugars'])

            # Allergens
            food_lower = row['name'].lower()
            allergens = []
            if any(word in food_lower for word in ['milk', 'cheese', 'yogurt', 'cream', 'butter']):
                allergens.append('dairy')
            if any(word in food_lower for word in ['egg']):
                allergens.append('eggs')
            if any(word in food_lower for word in ['wheat', 'flour', 'bread']):
                allergens.append('gluten')
            if any(word in food_lower for word in ['nuts', 'almond', 'peanut', 'walnut', 'pecan']):
                allergens.append('nuts')
            metadata['allergens'] = allergens

            documents.append(Document(page_content=text, metadata=metadata))

        logger.debug(f"Loaded {len(documents)} documents from {csv_path}")
        return documents

    # ========================================
    # Private Methods: RAG Chain
    # ========================================

    def _build_chain(self) -> None:
        """Build RAG chain with system prompt."""
        prompt_template = ChatPromptTemplate.from_template(self.SYSTEM_PROMPT)

        stuff_documents_chain = create_stuff_documents_chain(
            llm=self.llm,
            prompt=prompt_template
        )

        self.rag_chain = create_retrieval_chain(
            retriever=self.smart_retriever,
            combine_docs_chain=stuff_documents_chain
        )

    # ========================================
    # Public Methods
    # ========================================

    def ask(self, user_input: str) -> str:
        """
        Main interface - get recipe recommendations.

        Args:
            user_input: User's query

        Returns:
            AI-generated recipe recommendations
        """
        if not self.rag_chain:
            logger.error("System not initialized - call initialize() first")
            raise RuntimeError("System not initialized. Call initialize() first.")

        logger.debug(f"Processing query: {user_input[:50]}...")
        response = self.rag_chain.invoke({"input": user_input})
        logger.debug("Query processed successfully")
        return response.get("answer", "No response generated.")

    def get_retrieved_docs(self, query: str) -> List[Document]:
        """
        Debug method - see which documents are retrieved for a query.

        Args:
            query: Search query

        Returns:
            List of retrieved Document objects
        """
        if not self.smart_retriever:
            logger.error("System not initialized")
            return []

        logger.debug(f"Retrieving documents for: {query[:50]}...")
        return self.smart_retriever.invoke(query)

    def reload_vectorstores(self) -> None:
        """
        Reload vectorstores from disk (useful after manual updates).
        """
        logger.info("Reloading vectorstores from disk")
        self._load_vectorstores()

        # Recreate SmartRetriever
        self.smart_retriever = SmartRetriever(
            vectorstore_recipes=self.vectorstore_recipes,
            vectorstore_nutrition=self.vectorstore_nutrition,
            k=self.k
        )

        # Rebuild chain
        self._build_chain()
        logger.info("Vectorstores reloaded and chain rebuilt successfully")

    def update_system_prompt(self, new_prompt: str) -> None:
        """
        Update the system prompt and rebuild the chain.

        Args:
            new_prompt: New system prompt (must contain {input} and {context} placeholders)
        """
        if "{input}" not in new_prompt or "{context}" not in new_prompt:
            logger.error("Invalid prompt - missing {input} or {context} placeholders")
            raise ValueError("Prompt must contain {input} and {context} placeholders")

        logger.info("Updating system prompt")
        self.SYSTEM_PROMPT = new_prompt
        self._build_chain()
        logger.info("System prompt updated and chain rebuilt")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get system statistics and configuration.

        Returns:
            Dictionary with system info
        """
        if not self.vectorstore_recipes or not self.vectorstore_nutrition:
            return {"status": "not_initialized"}

        return {
            "status": "initialized",
            "model": self.model_name,
            "temperature": self.temperature,
            "k": self.k,
            "vectorstores": {
                "recipes_and_meals": {
                    "vectors": self.vectorstore_recipes.index.ntotal,
                    "path": str(self.vectorstore_path / "recipes_and_meals_db")
                },
                "nutrition_facts": {
                    "vectors": self.vectorstore_nutrition.index.ntotal,
                    "path": str(self.vectorstore_path / "nutrition_facts_db")
                }
            },
            "data_folder": str(self.data_folder),
            "ollama_url": self.ollama_base_url
        }
    
if __name__ == "__main__":
    my_rag = RecipesNutritionRAG(
        data_folder="C:\\Users\\tranq\\Desktop\\neue_fische\\nutrition-ai-assistant\\data\\",
        vectorstore_path="C:\\Users\\tranq\\Desktop\\neue_fische\\nutrition-ai-assistant\\vector_databases\\",
        model_name="llama3.2",
        log_level="INFO"
        )
    my_rag.initialize()
    answer = my_rag.ask("I'm vegetarian and need recommendations for my next meal")
    print(answer)