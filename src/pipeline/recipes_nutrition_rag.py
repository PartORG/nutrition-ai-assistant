"""
RecipesNutritionRAG - RAG system for personalized recipe recommendations.

Inherits from BaseRAG and adds:
    - Dual vectorstore system (recipes + nutrition facts)
    - Smart query routing (recipes/nutrition/both)
    - CSV-based data ingestion (4 different CSV formats)
    - Medical-grade system prompt with complete recipe structure
"""

import ast
import re
import logging
import warnings
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy
from langchain_core.runnables import Runnable, RunnableConfig

import sys


project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
from pipeline.base_rag import BaseRAG

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


# ========================================
# HELPER FUNCTIONS
# ========================================

def determine_query_type(query: str) -> str:
    """Intelligently route queries to the right collection."""
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


def smart_retrieve(query: str, vectorstore_recipes, vectorstore_nutrition, k: int = 10) -> List[Document]:
    """Smart retrieval across collections based on query type."""
    query_type = determine_query_type(query)

    if query_type == "recipes":
        results = vectorstore_recipes.similarity_search(query, k=k)
    elif query_type == "nutrition":
        results = vectorstore_nutrition.similarity_search(query, k=k)
    else:
        results_recipes = vectorstore_recipes.similarity_search(query, k=k // 2 + 1)
        results_nutrition = vectorstore_nutrition.similarity_search(query, k=k // 2 + 1)
        results = results_recipes + results_nutrition

    return results[:k]


# ========================================
# SMART RETRIEVER CLASS
# ========================================

class SmartRetriever(Runnable):
    """Intelligent retriever that routes queries to appropriate vector collections."""

    def __init__(self, vectorstore_recipes, vectorstore_nutrition, k=10):
        self.vectorstore_recipes = vectorstore_recipes
        self.vectorstore_nutrition = vectorstore_nutrition
        self.k = k

    def invoke(self, input: dict | str, config: RunnableConfig = None) -> List[Document]:
        if isinstance(input, dict):
            query = input.get("input", "")
        else:
            query = input

        return smart_retrieve(
            query=query,
            vectorstore_recipes=self.vectorstore_recipes,
            vectorstore_nutrition=self.vectorstore_nutrition,
            k=self.k,
        )


# ========================================
# MAIN RAG CLASS
# ========================================

class RecipesNutritionRAG(BaseRAG):
    """RAG system for personalized recipe recommendations with dual vectorstores."""

    SYSTEM_PROMPT = """You are NutriGuide, an AI nutrition assistant providing personalized recipe recommendations based on medical needs and user preferences.

## CRITICAL SAFETY DISCLAIMER
You are a recommendation system ONLY. Your suggestions do NOT replace professional medical advice from qualified healthcare providers.

## INPUT CONTEXT YOU RECEIVE
- {context}: Pre-filtered recipes and nutrition facts from vector database
- {input}: User's current query with preferences/restrictions/instructions
- User may specify "I have X in my fridge/ at home" → PRIORITIZE using those ingredients

## PRIORITIZATION RULES (STRICT ORDER)
1. **CRITICAL**: Life-threatening allergies (e.g., nut allergy, celiac disease)
2. **HIGH**: Medical dietary needs (e.g., diabetes → low sugar, hypertension → low sodium)
3. **MEDIUM**: Non-life-threatening allergies/intolerances
4. **LOW**: Personal preferences (cuisine, taste)
If conflicts occur, ALWAYS prioritize higher levels and explain why.

## OUTPUT REQUIREMENTS

### RECIPE QUANTITY
- **DEFAULT**: Always provide **3 recipes** for user choice
- Only provide 1 recipe if user explicitly says "give me ONE recipe" or "just one"
- "I need a recipe" → Still provide 3 options
- For ambiguous requests → Provide 3 recipes

### MANDATORY STRUCTURE FOR EACH RECIPE

**1. Recipe Name**

**2. Why This Recipe:**
- Meets dietary requirements (vegetarian, low-carb, etc.)
- Matches medical needs (if applicable)
- Uses available ingredients (if user mentioned any)
- Calorie/macronutrient alignment

**3. Nutritional Information (per serving):**
- Calories: X kcal
- Protein: X g | Carbs: X g | Fat: X g
- Fiber: X g (if relevant) | Sodium: X mg (if relevant)

**4. Ingredients (METRIC ONLY - CRITICAL):**
**MANDATORY CONVERSIONS:**
- 1 cup → 240 ml (or 240g for solids)
- 1 tbsp → 15 ml
- 1 tsp → 5 ml
- 1 oz → 28 g
- 1 lb → 454 g
- Temperatures: 350°F → 175°C, 400°F → 200°C

**FORMAT:**
- 200g chicken breast
- 150ml olive oil
- 500g tomatoes

**If quantities missing in context:**
- Estimate based on serving size
- Mark as "(approximately)"
- Example: 1 serving = 325g → estimate ~300-350g total ingredients

**5. Cooking Instructions (CRITICAL):**
- Extract from {context} if available
- If missing: Create logical steps based on ingredients and cooking method
- Include temperatures in Celsius (°C)
- Number steps clearly (1., 2., 3., ...)

**NEVER say:** "Instructions not available"
**ALWAYS provide:** Complete, usable step-by-step instructions

**6. Time Information:**
- Prep Time: X min | Cook Time: X min | Total: X min

---

## HANDLING SPECIAL CASES

**User Says "I have X/Y/Z in my fridge":**
→ PRIORITIZE recipes using those ingredients
→ If no exact match, note which ingredients were used: "This recipe uses your spinach and tomatoes."

**No Perfect Match in Context:**
→ Select closest recipe from {context}
→ Silently adapt (substitute ingredients, adjust portions)
→ DO NOT include "Adaptations Made" section (user doesn't need to know)

**Missing Nutritional Data:**
→ Calculate from individual ingredients
→ Show calculation: "Estimated: 200g chicken (330 kcal) + 100g rice (130 kcal) = 460 kcal"

**Conflicting Requirements:**
→ Follow prioritization hierarchy (medical > preference)
→ Explain: "I prioritized low-sodium recipes for your hypertension over your Italian cuisine preference. Here are the best options..."

---

## FINAL OUTPUT FORMAT

User Query: {input}
Retrieved Context: {context}

**Generate 3 complete recipes following the MANDATORY STRUCTURE above.**
**Convert ALL measurements to metric.**
**Provide complete cooking instructions (never skip this section).**

**Reminder**: These are general recommendations. Consult healthcare providers before significant dietary changes."""

    def __init__(
        self,
        data_folder: str,
        vectorstore_path: str,
        model_name: str = "llama3.2",
        temperature: float = 0.5,
        k: int = 10,
        ollama_base_url: str = "http://localhost:11434/",
        log_level: str = "INFO",
    ):
        # Setup logging
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

        super().__init__(
            vectorstore_path=vectorstore_path,
            model_name=model_name,
            temperature=temperature,
            ollama_base_url=ollama_base_url,
        )
        self.data_folder = Path(data_folder)
        self.k = k

        # Dual vectorstores (set during initialize)
        self.vectorstore_recipes: Optional[FAISS] = None
        self.vectorstore_nutrition: Optional[FAISS] = None
        self.smart_retriever: Optional[SmartRetriever] = None

    # ================================================================
    # BaseRAG abstract method implementations
    # ================================================================

    def _get_system_prompt(self) -> str:
        if hasattr(self, "_custom_prompt"):
            return self._custom_prompt
        return self.SYSTEM_PROMPT

    def _ingest_documents(self) -> List[Document]:
        """Load all CSV data. Returns combined list — actual splitting into
        two vectorstores happens in _build_vectorstore.
        Missing CSV files are skipped with a warning."""
        csv_loaders = [
            ("cleaned_recipes.csv", self._load_recipes_csv),
            ("cleaned_recipes_data_sample.csv", self._load_recipes_data_sample_csv),
            ("cleaned_healthy_meals.csv", self._load_healthy_meals_csv),
            ("cleaned_nutrition.csv", self._load_nutrition_csv),
        ]

        all_loaded: Dict[str, List[Document]] = {}
        for filename, loader_fn in csv_loaders:
            csv_path = self.data_folder / filename
            if not csv_path.exists():
                logger.warning(f"CSV not found, skipping: {csv_path}")
                all_loaded[filename] = []
                continue
            all_loaded[filename] = loader_fn(str(csv_path))

        recipes1 = all_loaded["cleaned_recipes.csv"]
        recipes2 = all_loaded["cleaned_recipes_data_sample.csv"]
        meals = all_loaded["cleaned_healthy_meals.csv"]
        nutrition = all_loaded["cleaned_nutrition.csv"]

        # Tag documents so _build_vectorstore can split them
        for doc in recipes1 + recipes2 + meals:
            doc.metadata["_collection"] = "recipes"
        for doc in nutrition:
            doc.metadata["_collection"] = "nutrition"

        all_docs = recipes1 + recipes2 + meals + nutrition
        logger.info(
            f"Loaded documents — Recipes/Meals: {len(recipes1) + len(recipes2) + len(meals)}, "
            f"Nutrition: {len(nutrition)}"
        )
        return all_docs

    # ================================================================
    # BaseRAG overrides (dual vectorstore)
    # ================================================================

    def _vectorstore_exists(self) -> bool:
        recipes_path = self.vectorstore_path / "recipes_and_meals_db"
        nutrition_path = self.vectorstore_path / "nutrition_facts_db"
        return recipes_path.exists() and nutrition_path.exists()

    def _build_vectorstore(self, documents: List[Document]) -> None:
        """Build two separate FAISS indices from the tagged documents."""
        recipe_docs = [d for d in documents if d.metadata.get("_collection") == "recipes"]
        nutrition_docs = [d for d in documents if d.metadata.get("_collection") == "nutrition"]

        logger.info(f"Building FAISS indices (recipes={len(recipe_docs)}, nutrition={len(nutrition_docs)})")

        self.vectorstore_recipes = FAISS.from_documents(
            documents=recipe_docs,
            embedding=self.embeddings,
            distance_strategy=DistanceStrategy.COSINE,
        )
        self.vectorstore_nutrition = FAISS.from_documents(
            documents=nutrition_docs,
            embedding=self.embeddings,
            distance_strategy=DistanceStrategy.COSINE,
        )

        # Save to disk
        self.vectorstore_path.mkdir(parents=True, exist_ok=True)
        self.vectorstore_recipes.save_local(str(self.vectorstore_path / "recipes_and_meals_db"))
        self.vectorstore_nutrition.save_local(str(self.vectorstore_path / "nutrition_facts_db"))
        logger.info(f"Vectorstores saved to {self.vectorstore_path}")

    def _load_vectorstore(self) -> None:
        """Load both FAISS indices from disk."""
        recipes_path = self.vectorstore_path / "recipes_and_meals_db"
        nutrition_path = self.vectorstore_path / "nutrition_facts_db"

        self.vectorstore_recipes = FAISS.load_local(
            folder_path=str(recipes_path),
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True,
        )
        self.vectorstore_nutrition = FAISS.load_local(
            folder_path=str(nutrition_path),
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True,
        )
        logger.info(
            f"Vectorstores loaded — Recipes: {self.vectorstore_recipes.index.ntotal}, "
            f"Nutrition: {self.vectorstore_nutrition.index.ntotal}"
        )

    def _setup_retriever(self) -> None:
        """Create SmartRetriever that routes queries across both vectorstores."""
        self.smart_retriever = SmartRetriever(
            vectorstore_recipes=self.vectorstore_recipes,
            vectorstore_nutrition=self.vectorstore_nutrition,
            k=self.k,
        )
        # BaseRAG._build_chain uses self.retriever
        self.retriever = self.smart_retriever
        logger.info(f"SmartRetriever created (k={self.k})")

    # ================================================================
    # CSV data loaders (private)
    # ================================================================

    def _load_recipes_csv(self, csv_path: str) -> List[Document]:
        """Load cleaned_recipes.csv with structured nutrition parsing."""
        logger.debug(f"Loading {csv_path}")
        df = pd.read_csv(csv_path)
        documents = []

        for idx, row in df.iterrows():
            if pd.isna(row["recipe_name"]):
                continue

            text_parts = [
                f"Recipe: {row['recipe_name']}",
                f"\nCuisine: {row.get('cuisine_path', 'Not specified')}",
                f"\nIngredients:\n{row['ingredients']}",
                f"\nDirections:\n{row['directions']}",
            ]

            if pd.notna(row.get("prep_time")):
                text_parts.append(f"\nPrep Time: {row['prep_time']}")
            if pd.notna(row.get("cook_time")):
                text_parts.append(f"\nCook Time: {row['cook_time']}")
            if pd.notna(row.get("nutrition")):
                text_parts.append(f"\nNutrition Facts: {row['nutrition']}")

            full_text = "".join(text_parts)

            metadata = {
                "doc_type": "recipe",
                "source_file": "cleaned_recipes",
                "recipe_name": row["recipe_name"],
                "servings": row.get("servings", "Not specified"),
            }

            if pd.notna(row.get("cuisine_path")):
                cuisine = row["cuisine_path"].split("/")[-1] if "/" in str(row["cuisine_path"]) else row["cuisine_path"]
                metadata["cuisine"] = cuisine

            if pd.notna(row.get("prep_time")):
                prep_str = str(row["prep_time"]).lower()
                prep_mins = sum(
                    int(s) * (60 if "hr" in prep_str else 1) for s in re.findall(r"\d+", prep_str)
                )
                metadata["prep_time_min"] = prep_mins

            if pd.notna(row.get("cook_time")):
                cook_str = str(row["cook_time"]).lower()
                cook_mins = sum(
                    int(s) * (60 if "hr" in cook_str else 1) for s in re.findall(r"\d+", cook_str)
                )
                metadata["cook_time_min"] = cook_mins

            ingredients_lower = str(row["ingredients"]).lower()
            allergens = []
            if any(w in ingredients_lower for w in ["milk", "cheese", "butter", "cream", "yogurt"]):
                allergens.append("dairy")
            if any(w in ingredients_lower for w in ["egg"]):
                allergens.append("eggs")
            if any(w in ingredients_lower for w in ["wheat", "flour", "bread"]):
                allergens.append("gluten")
            if any(w in ingredients_lower for w in ["nuts", "almond", "peanut", "walnut"]):
                allergens.append("nuts")
            metadata["allergens"] = allergens

            diet_tags = []
            if "vegetarian" in ingredients_lower or "veggie" in ingredients_lower:
                diet_tags.append("vegetarian")
            if "vegan" in ingredients_lower:
                diet_tags.append("vegan")
            if not any(meat in ingredients_lower for meat in ["chicken", "beef", "pork", "fish", "meat"]):
                diet_tags.append("vegetarian")
            metadata["diet_tags"] = diet_tags

            documents.append(Document(page_content=full_text, metadata=metadata))

        logger.debug(f"Loaded {len(documents)} documents from {csv_path}")
        return documents

    def _load_recipes_data_sample_csv(self, csv_path: str) -> List[Document]:
        """Load cleaned_recipes_data_sample.csv with NER parsing."""
        logger.debug(f"Loading {csv_path}")
        df = pd.read_csv(csv_path)
        documents = []

        for idx, row in df.iterrows():
            if pd.isna(row["title"]):
                continue

            try:
                ingredients_list = ast.literal_eval(row["ingredients"])
                ingredients_text = "\n".join([f"- {ing}" for ing in ingredients_list])
            except Exception:
                ingredients_text = row["ingredients"]

            try:
                directions_list = ast.literal_eval(row["directions"])
                directions_text = "\n".join([f"{i + 1}. {step}" for i, step in enumerate(directions_list)])
            except Exception:
                directions_text = row["directions"]

            try:
                ner_list = ast.literal_eval(row["NER"])
                ner_text = ", ".join(ner_list)
            except Exception:
                ner_list = []
                ner_text = ""

            text_parts = [
                f"Recipe: {row['title']}",
                f"\nIngredients:\n{ingredients_text}",
                f"\nDirections:\n{directions_text}",
                f"\nKey Ingredients: {ner_text}",
            ]
            full_text = "".join(text_parts)

            metadata = {
                "doc_type": "recipe",
                "source_file": "cleaned_recipes_data_sample",
                "recipe_name": row["title"],
                "ingredient_list": ner_list if ner_list else None,
            }

            ingredients_lower = str(row["ingredients"]).lower()
            allergens = []
            if any(w in ingredients_lower for w in ["milk", "cheese", "butter", "cream", "yogurt"]):
                allergens.append("dairy")
            if any(w in ingredients_lower for w in ["egg"]):
                allergens.append("eggs")
            if any(w in ingredients_lower for w in ["wheat", "flour", "bread"]):
                allergens.append("gluten")
            if any(w in ingredients_lower for w in ["nuts", "almond", "peanut", "walnut"]):
                allergens.append("nuts")
            metadata["allergens"] = allergens

            diet_tags = []
            if not any(meat in ingredients_lower for meat in ["chicken", "beef", "pork", "fish", "meat", "lamb"]):
                diet_tags.append("vegetarian")
            metadata["diet_tags"] = diet_tags

            documents.append(Document(page_content=full_text, metadata=metadata))

        logger.debug(f"Loaded {len(documents)} documents from {csv_path}")
        return documents

    def _load_healthy_meals_csv(self, csv_path: str) -> List[Document]:
        """Load cleaned_healthy_meals.csv with numeric nutrition metadata."""
        logger.debug(f"Loading {csv_path}")
        df = pd.read_csv(csv_path)
        documents = []

        for idx, row in df.iterrows():
            if pd.isna(row["meal_name"]):
                continue

            text = f"""Meal: {row['meal_name']} ({row['cuisine']} {row['meal_type']})
Diet Type: {row['diet_type']}

Nutrition per {row['serving_size_g']}g serving:
- Calories: {row['calories']} kcal
- Protein: {row['protein_g']}g | Carbs: {row['carbs_g']}g | Fat: {row['fat_g']}g
- Fiber: {row['fiber_g']}g | Sugar: {row['sugar_g']}g
- Sodium: {row['sodium_mg']}mg | Cholesterol: {row['cholesterol_mg']}mg

Preparation: {row['cooking_method']} (Prep: {row['prep_time_min']}min, Cook: {row['cook_time_min']}min)
"""

            metadata = {
                "doc_type": "meal",
                "source_file": "cleaned_healthy_meals",
                "recipe_name": row["meal_name"],
                "cuisine": row["cuisine"],
                "meal_type": row["meal_type"],
                "diet_type": row["diet_type"],
                "calories": int(row["calories"]),
                "protein_g": float(row["protein_g"]),
                "carbs_g": float(row["carbs_g"]),
                "fat_g": float(row["fat_g"]),
                "fiber_g": float(row["fiber_g"]),
                "sugar_g": float(row["sugar_g"]),
                "sodium_mg": int(row["sodium_mg"]),
                "cholesterol_mg": int(row["cholesterol_mg"]),
                "serving_size_g": int(row["serving_size_g"]),
                "cooking_method": row["cooking_method"],
                "prep_time_min": int(row["prep_time_min"]),
                "cook_time_min": int(row["cook_time_min"]),
            }

            diet_tags = [row["diet_type"].lower()]
            if row["diet_type"].lower() in ["vegan", "vegetarian"]:
                diet_tags.append("vegetarian")
            metadata["diet_tags"] = diet_tags

            allergens = []
            meal_lower = row["meal_name"].lower()
            if any(w in meal_lower for w in ["cheese", "yogurt", "milk"]):
                allergens.append("dairy")
            metadata["allergens"] = allergens

            documents.append(Document(page_content=text, metadata=metadata))

        logger.debug(f"Loaded {len(documents)} documents from {csv_path}")
        return documents

    def _load_nutrition_csv(self, csv_path: str) -> List[Document]:
        """Load cleaned_nutrition.csv - detailed ingredient nutrition database."""
        logger.debug(f"Loading {csv_path}")
        df = pd.read_csv(csv_path)
        documents = []

        def parse_numeric(val):
            if pd.isna(val):
                return None
            try:
                return float(re.sub(r"[^\d.]", "", str(val)))
            except Exception:
                return None

        for idx, row in df.iterrows():
            if pd.isna(row["name"]):
                continue

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

            metadata = {
                "doc_type": "nutrition_fact",
                "source_file": "cleaned_nutrition",
                "food_name": row["name"],
                "serving_size": row["serving_size"],
                "calories": parse_numeric(row["calories"]),
                "protein_g": parse_numeric(row["protein"]),
                "carbs_g": parse_numeric(row["carbohydrate"]),
                "fat_g": parse_numeric(row["total_fat"]),
                "fiber_g": parse_numeric(row["fiber"]),
                "sugar_g": parse_numeric(row["sugars"]),
            }

            food_lower = row["name"].lower()
            allergens = []
            if any(w in food_lower for w in ["milk", "cheese", "yogurt", "cream", "butter"]):
                allergens.append("dairy")
            if any(w in food_lower for w in ["egg"]):
                allergens.append("eggs")
            if any(w in food_lower for w in ["wheat", "flour", "bread"]):
                allergens.append("gluten")
            if any(w in food_lower for w in ["nuts", "almond", "peanut", "walnut", "pecan"]):
                allergens.append("nuts")
            metadata["allergens"] = allergens

            documents.append(Document(page_content=text, metadata=metadata))

        logger.debug(f"Loaded {len(documents)} documents from {csv_path}")
        return documents

    # ================================================================
    # Domain-specific public methods
    # ================================================================

    def get_retrieved_docs(self, query: str) -> List[Document]:
        """Debug method — see which documents are retrieved for a query."""
        if not self.smart_retriever:
            logger.error("System not initialized")
            return []
        return self.smart_retriever.invoke(query)

    def reload_vectorstores(self) -> None:
        """Reload vectorstores from disk (useful after manual updates)."""
        logger.info("Reloading vectorstores from disk")
        self._load_vectorstore()
        self._setup_retriever()
        self._build_chain()
        logger.info("Vectorstores reloaded and chain rebuilt")

    def get_stats(self) -> Dict[str, Any]:
        """Return system statistics including both vectorstores."""
        stats = super().get_stats()
        if self.vectorstore_recipes and self.vectorstore_nutrition:
            stats["vectorstores"] = {
                "recipes_and_meals": {
                    "vectors": self.vectorstore_recipes.index.ntotal,
                    "path": str(self.vectorstore_path / "recipes_and_meals_db"),
                },
                "nutrition_facts": {
                    "vectors": self.vectorstore_nutrition.index.ntotal,
                    "path": str(self.vectorstore_path / "nutrition_facts_db"),
                },
            }
        stats["k"] = self.k
        stats["data_folder"] = str(self.data_folder)
        return stats


if __name__ == "__main__":
    from pipeline.config import DATA_DIR, RECIPES_NUTRITION_VECTOR_PATH, LLM_MODEL

    my_rag = RecipesNutritionRAG(
        data_folder=str(DATA_DIR),
        vectorstore_path=str(RECIPES_NUTRITION_VECTOR_PATH),
        model_name=LLM_MODEL,
        log_level="INFO",
    )
    my_rag.initialize()
    answer = my_rag.ask("I'm vegetarian and need a recipe")
    print(answer)
