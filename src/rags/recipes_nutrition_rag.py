"""
RecipesNutritionRAG - RAG system for personalized recipe recommendations.

Inherits from BaseRAG and adds:
    - Dual vectorstore system (recipes + nutrition facts)
    - Smart query routing (recipes/nutrition/both)
    - CSV-based data ingestion (4 different CSV formats)
    - Batched CSV ingestion for large files (2.2M+ rows)
    - Medical-grade system prompt with complete recipe structure
"""
import sys
from pathlib import Path

# Add src directory to path
src_dir = Path(__file__).parent.parent
sys.path.insert(0, str(src_dir))

import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI #imports until here for testing purpose, TODO: remove if not needed

import ast
import json
import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Set

import pandas as pd
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy
from langchain_core.runnables import Runnable, RunnableConfig

from rags.base_rag import BaseRAG

logger = logging.getLogger(__name__)


class SmartRetriever(Runnable):
    """Intelligent retriever that routes queries to appropriate vector collections."""

    def __init__(self, vectorstore_recipes, vectorstore_nutrition, k=14):
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
        """Retrieve documents from the appropriate vectorstore(s) based on query type."""
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
        """Route queries to the right collection based on keywords."""
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


class RecipesNutritionRAG(BaseRAG):
    """RAG system for personalized recipe recommendations with dual vectorstores."""

    LARGE_FILE_THRESHOLD_MB = 50

    SYSTEM_PROMPT = """You are a recipe recommendation engine in a nutrition assistant pipeline. Your task is to select and adapt recipes based on structured user requirements.

## INPUT FORMAT YOU RECEIVE

{{input}} will contain:
- User Query: Original user request
- Dietary Restrictions: Hard constraints (vegetarian, gluten-free, etc.)
- Preferences: Soft preferences (cuisine type, meal speed, etc.)
- Nutrition Guidelines: Calorie and macronutrient targets
- Foods to Avoid: Complete exclusions (allergies, medical)
- Foods to Limit: Partial restrictions with limits
- Instructions: Special requests (use fridge ingredients, specific foods wanted)

{{context}} contains pre-filtered recipe and nutrition data from vector database (may be in metric or imperial units).

## PRIORITIZATION RULES (STRICT ORDER)

When requirements conflict, follow this hierarchy:
1. **CRITICAL**: Foods to Avoid (allergies, medical contraindications)
2. **HIGH**: Dietary Restrictions (vegetarian, vegan, gluten-free, etc.)
3. **MEDIUM**: Nutrition Guidelines (calorie limits, macronutrient targets)
4. **LOW**: Preferences (cuisine, cooking time, taste)

If no recipe in {{context}} meets all critical requirements, silently adapt the closest matches to user requirements.

## RECIPE QUANTITY

- **DEFAULT**: Always provide 3 recipes (unless specified in Instructions)
- If Instructions say "I want ONE recipe" → Provide 1
- If Instructions say "give me 5 options" → Provide 5
- For ambiguous requests → Provide 3

## RECIPE ADAPTATION PROTOCOL

**When to Adapt:**
- Recipe contains Foods to Avoid → Replace with safe alternatives
- Recipe exceeds Nutrition Guidelines → Adjust portions or substitute ingredients
- Recipe missing Dietary Restrictions compliance → Substitute non-compliant ingredients
- User requests specific ingredients in Instructions → Incorporate them if possible

**Adaptation Rules:**
- Make adaptations SILENTLY (do not mention original recipe or changes made)
- Maintain recipe authenticity (don't change cuisine character drastically)
- Preserve cooking method when possible (oven → oven, not oven → microwave)
- Calculate new nutritional values after adaptations

**Example:**
- User: vegetarian, Recipe: chicken pasta
- **Adapt:** Replace chicken with chickpeas or tofu (same protein content)
- **Output:** Present as "Chickpea Pasta" (not "Adapted Chicken Pasta")

## MEASUREMENT CONVERSIONS (MANDATORY)

**Imperial to Metric Conversions:**
- 1 cup → 240 ml (liquids) or 240 g (solids like flour/sugar)
- 1 tbsp → 15 ml
- 1 tsp → 5 ml
- 1 oz → 28 g
- 1 lb → 454 g
- 1 fl oz → 30 ml

**Temperature Conversions:**
- 350°F → 175°C
- 375°F → 190°C
- 400°F → 200°C
- 425°F → 220°C

**If Context Has Metric:** Keep as-is
**If Context Has Imperial:** Convert to metric in output

## HANDLING MISSING DATA

**Missing Ingredient Quantities:**
- Use context nutritional info to estimate
- Base on serving size (e.g., 325g serving → ~300-350g total ingredients)
- For seasonings: Use standard amounts (salt: 5g, pepper: 2g, herbs: 10g)

**Missing Cooking Instructions:**
- Create logical steps based on:
  - Ingredient types (raw meat → cook, vegetables → sauté)
  - Cooking method stated (Baked, Fried, Steamed, Raw)
  - Standard techniques for cuisine type
- Include temperatures in Celsius
- Number steps clearly

**Missing Nutritional Values:**
- Calculate from individual ingredients
- Use standard USDA values for common foods
- Include in output (never leave nutrition fields empty)

## SPECIAL CASES

**Instructions: "use ingredients from fridge/home [list]"**
→ PRIORITIZE recipes using listed ingredients
→ If no exact match, incorporate ingredients into closest recipe
→ Example: User has spinach, tomatoes → Add to pasta recipe

**Instructions: "I want [specific food]"**
→ MUST include that food in at least one recipe
→ Example: "I want rice and fish" → One recipe must contain both

**Foods to Limit: "sugar (max 10g)"**
→ Select recipes ≤10g sugar per serving
→ If context recipe exceeds limit, reduce sugar-containing ingredients proportionally

## JSON OUTPUT FORMAT (STRICT)

Return ONLY valid JSON (no markdown, no explanations)

## ⚠️ CRITICAL OUTPUT RULES - READ CAREFULLY ⚠️

1. **ONLY use these field names**: name, why_recommended, servings, prep_time, cook_time, ingredients, cook_instructions, nutrition
2. **FORBIDDEN fields**: Do NOT include "Meal", "Diet Type", "Preparation", "Key Vitamins", "Key Minerals", "Macronutrients"
3. **Structure**: Root object MUST have single key "recipes" with array value
4. **Ingredients**: Must be flat string array, NOT nested objects with "Name", "Macronutrients"
5. **Validation**: If unsure, refer to the example format below and follow it EXACTLY. Do NOT deviate from the structure or field names under any circumstances.

## EXAMPLE (MANDATORY REFERENCE)

{{
  "recipes": [
    {{
      "name": "string",
      "why_recommended": "string (max 150 chars)",
      "servings": integer,
      "prep_time": "X minutes",
      "cook_time": "X minutes",
      "ingredients": ["200g item1", "150ml item2"],
      "cook_instructions": "1. Step\\n2. Step\\n3. Step",
      "nutrition": {{
        "calories": integer,
        "protein_g": float,
        "carbs_g": float,
        "fat_g": float,
        "fiber_g": float,
        "sodium_mg": integer,
        "sugar_g": float,
        "saturated_fat_g": float
      }}
    }}
  ]
}}

**Field Requirements:**

- **`name`**: String (adapted recipe name, not original if modified)
- **`why_recommended`**: Single sentence explaining match to user requirements (max 150 characters)
- **`servings`**: Integer (number of servings this recipe makes)
- **`prep_time`**: String in format "X minutes" or "X hours Y minutes" (always include unit)
- **`cook_time`**: String in format "X minutes" or "X hours Y minutes" (always include unit)
- **`ingredients`**: Array of strings, each formatted as:
  - Metric quantities: "200g chicken breast", "150ml olive oil"
  - Whole items: "2 cloves garlic", "1 onion"
  - Seasonings: "Salt to taste", "1 tsp pepper"
- **`cook_instructions`**: Single string with numbered steps separated by `\\n` (newline character)
  - Include temperatures in Celsius with unit: "180°C"
  - Number each step: "1. ...", "2. ...", "3. ..."
- **`nutrition`**: Object with numeric values:
  - `calories`: Integer (kcal per serving)
  - `protein_g`, `carbs_g`, `fat_g`, `fiber_g`, `sugar_g`, `saturated_fat_g`: Float (grams per serving)
  - `sodium_mg`: Integer (milligrams per serving)

**Validation Rules:**

1. **Metric System ONLY**:
   - Ingredients must use metric: "200g", "150ml" (never "1 cup", "2 tbsp")
   - Temperatures in Celsius: "180°C" (never "350°F")
   
2. **String Formatting**:
   - Time strings must include unit: "15 minutes" (not just "15")
   - Instructions must use `\\n` between steps: "1. Step one\\n2. Step two"
   - Ingredient strings must start with quantity: "200g chicken" (not "chicken 200g")

3. **Completeness**:
   - Never return null values (estimate if data missing)
   - Recipe count MUST match requested quantity (default 3)
   - All nutrition fields are required (if unknown, calculate from ingredients)

4. **why_recommended Rules**:
   - Must reference at least ONE user requirement (dietary restriction, nutrition goal, or available ingredient)
   - Keep concise (one sentence, max 150 characters)
   - Examples:
     * "Vegetarian, high-protein meal under 500 calories using your spinach and tomatoes"
     * "Gluten-free breakfast rich in fiber, matches your Italian cuisine preference"

---

## YOUR TASK NOW

User Input:
{input}

Retrieved Context:
{context}

**Generate JSON with the exact number of recipes requested (default 3), adapting recipes silently to meet all requirements from highest to lowest priority. Convert all measurements to metric. Return ONLY the JSON object.**"""

    def __init__(
        self,
        data_folder: str,
        vectorstore_path: str,
        model_name: str = "llama3.2",
        temperature: float = 0.3,
        k: int = 14,
        ollama_base_url: str = "http://localhost:11434/",
    ):
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

    def _ingest_documents(self) -> List[Document]:
        """Load all CSV data. Returns combined list — actual splitting into
        two vectorstores happens in _build_vectorstore.

        Large files (above LARGE_FILE_THRESHOLD_MB) are automatically routed
        to batched ingestion via _ingest_large_csv() and ingested directly
        into the vectorstore, bypassing the in-memory return path.
        Missing CSV files are skipped with a warning."""
        csv_sources = [
            ("cleaned_recipes.csv", self._load_recipes_csv, "recipes"),
            ("cleaned_recipes_data_sample.csv", self._load_recipes_data_sample_csv, "recipes"),
            ("cleaned_healthy_meals.csv", self._load_healthy_meals_csv, "recipes"),
            ("cleaned_nutrition.csv", self._load_nutrition_csv, "nutrition"),
        ]

        threshold_bytes = self.LARGE_FILE_THRESHOLD_MB * 1024 * 1024
        all_loaded: Dict[str, List[Document]] = {}
        large_files_ingested: List[str] = []

        for filename, loader_fn, collection in csv_sources:
            csv_path = self.data_folder / filename
            if not csv_path.exists():
                logger.warning("CSV not found, skipping: %s", csv_path)
                all_loaded[filename] = []
                continue

            file_size = csv_path.stat().st_size
            if file_size > threshold_bytes:
                logger.info(
                    "Large file detected: %s (%.1f MB) — using batched ingestion",
                    filename, file_size / (1024 * 1024),
                )
                self._ingest_large_csv(str(csv_path), loader_fn, collection)
                large_files_ingested.append(filename)
                all_loaded[filename] = []
            else:
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

        if large_files_ingested:
            logger.info(
                "Large files ingested via batched path: %s",
                ", ".join(large_files_ingested),
            )
        logger.info(
            "Loaded documents (standard path) — Recipes/Meals: %d, Nutrition: %d",
            len(recipes1) + len(recipes2) + len(meals), len(nutrition),
        )
        return all_docs

    def _ingest_single_file(self, file_path: str) -> List[Document]:
        """Load a single CSV file and return Documents.

        Delegates to the appropriate loader based on filename.
        """
        path = Path(file_path)
        if not path.exists():
            logger.warning("File does not exist: %s", file_path)
            return []

        loader_map = {
            "cleaned_recipes.csv": self._load_recipes_csv,
            "cleaned_recipes_data_sample.csv": self._load_recipes_data_sample_csv,
            "cleaned_healthy_meals.csv": self._load_healthy_meals_csv,
            "cleaned_nutrition.csv": self._load_nutrition_csv,
        }

        loader_fn = loader_map.get(path.name)
        if loader_fn is None:
            logger.warning("No loader for file: %s", path.name)
            return []

        return loader_fn(file_path)

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

        logger.info("Building FAISS indices (recipes=%d, nutrition=%d)", len(recipe_docs), len(nutrition_docs))

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
        logger.info("Vectorstores saved to %s", self.vectorstore_path)

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
            "Vectorstores loaded — Recipes: %d, Nutrition: %d",
            self.vectorstore_recipes.index.ntotal,
            self.vectorstore_nutrition.index.ntotal,
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
        logger.info("SmartRetriever created (k=%d)", self.k)

    # ================================================================
    # Batched CSV ingestion (for large files)
    # ================================================================

    def _ingest_large_csv(
        self,
        csv_path: str,
        loader_fn,
        collection: str,
        chunksize: int = 5000,
    ) -> None:
        """Ingest a large CSV in batches, merging into the target vectorstore incrementally.

        Args:
            csv_path: Path to the CSV file.
            loader_fn: A method that accepts a DataFrame and returns List[Document].
            collection: Which vectorstore to merge into ("recipes" or "nutrition").
            chunksize: Number of CSV rows per batch.
        """
        path = Path(csv_path)
        if not path.exists():
            logger.error("CSV file not found: %s", csv_path)
            return

        if self.embeddings is None:
            raise RuntimeError("Call initialize() before _ingest_large_csv()")

        # Determine target vectorstore
        if collection == "recipes":
            target = self.vectorstore_recipes
        elif collection == "nutrition":
            target = self.vectorstore_nutrition
        else:
            raise ValueError(f"Unknown collection: {collection}")

        # Load progress metadata
        meta = self._get_batch_meta(csv_path)
        start_offset = meta.get("rows_indexed", 0)

        # Count total rows for progress logging
        total_rows = meta.get("total_rows")
        if total_rows is None:
            total_rows = sum(1 for _ in open(csv_path, encoding="utf-8")) - 1  # minus header
            meta["total_rows"] = total_rows

        total_batches = (total_rows + chunksize - 1) // chunksize
        rows_processed = start_offset
        batch_num = start_offset // chunksize

        logger.info(
            "Starting batched ingestion of %s (%d total rows, chunksize=%d, resuming from row %d)",
            path.name, total_rows, chunksize, start_offset,
        )

        reader = pd.read_csv(csv_path, chunksize=chunksize)
        for chunk_df in reader:
            # Skip already-processed batches
            if rows_processed + len(chunk_df) <= start_offset:
                rows_processed += len(chunk_df)
                batch_num += 1
                continue

            # Process this batch
            docs = loader_fn(chunk_df)
            if docs:
                new_store = FAISS.from_documents(
                    documents=docs,
                    embedding=self.embeddings,
                    distance_strategy=DistanceStrategy.COSINE,
                )
                if target is None:
                    target = new_store
                else:
                    target.merge_from(new_store)

                # Save checkpoint
                if collection == "recipes":
                    self.vectorstore_recipes = target
                    self.vectorstore_path.mkdir(parents=True, exist_ok=True)
                    target.save_local(str(self.vectorstore_path / "recipes_and_meals_db"))
                else:
                    self.vectorstore_nutrition = target
                    self.vectorstore_path.mkdir(parents=True, exist_ok=True)
                    target.save_local(str(self.vectorstore_path / "nutrition_facts_db"))

            rows_processed += len(chunk_df)
            batch_num += 1

            # Save progress
            meta["rows_indexed"] = rows_processed
            self._save_batch_meta(csv_path, meta)

            logger.info(
                "Batch %d/%d — %d / %d rows indexed (%d docs in this batch)",
                batch_num, total_batches, rows_processed, total_rows, len(docs),
            )

        logger.info("Batched ingestion complete: %s (%d rows)", path.name, rows_processed)

    def _batch_meta_path(self, csv_path: str) -> Path:
        """Path to the JSON file tracking batch ingestion progress."""
        name = Path(csv_path).stem
        return self.vectorstore_path / f"{name}_batch_meta.json"

    def _get_batch_meta(self, csv_path: str) -> Dict[str, Any]:
        meta_path = self._batch_meta_path(csv_path)
        if not meta_path.exists():
            return {}
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_batch_meta(self, csv_path: str, meta: Dict[str, Any]) -> None:
        meta_path = self._batch_meta_path(csv_path)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    # ================================================================
    # CSV data loaders (private)
    # ================================================================

    @staticmethod
    def _detect_allergens(text: str) -> List[str]:
        """Detect common allergens from ingredient/food text."""
        text_lower = text.lower()
        allergens = []
        if any(w in text_lower for w in ["milk", "cheese", "butter", "cream", "yogurt"]):
            allergens.append("dairy")
        if "egg" in text_lower:
            allergens.append("eggs")
        if any(w in text_lower for w in ["wheat", "flour", "bread"]):
            allergens.append("gluten")
        if any(w in text_lower for w in ["nuts", "almond", "peanut", "walnut", "pecan"]):
            allergens.append("nuts")
        return allergens

    @staticmethod
    def _detect_diet_tags(text: str) -> List[str]:
        """Detect diet tags from ingredient text."""
        text_lower = text.lower()
        tags = []
        if "vegan" in text_lower:
            tags.append("vegan")
            tags.append("vegetarian")
        elif not any(meat in text_lower for meat in ["chicken", "beef", "pork", "fish", "meat", "lamb"]):
            tags.append("vegetarian")
        return tags

    def _load_recipes_csv(self, csv_path_or_df) -> List[Document]:
        """Load cleaned_recipes.csv with structured nutrition parsing.

        Accepts either a file path (str) or a DataFrame (for batched ingestion).
        """
        if isinstance(csv_path_or_df, pd.DataFrame):
            df = csv_path_or_df
        else:
            logger.debug("Loading %s", csv_path_or_df)
            df = pd.read_csv(csv_path_or_df)

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

            ingredients_text = str(row["ingredients"])
            metadata["allergens"] = self._detect_allergens(ingredients_text)
            metadata["diet_tags"] = self._detect_diet_tags(ingredients_text)

            documents.append(Document(page_content=full_text, metadata=metadata))

        logger.debug("Loaded %d documents from recipes CSV", len(documents))
        return documents

    def _load_recipes_data_sample_csv(self, csv_path_or_df) -> List[Document]:
        """Load cleaned_recipes_data_sample.csv with NER parsing.

        Accepts either a file path (str) or a DataFrame (for batched ingestion).
        """
        if isinstance(csv_path_or_df, pd.DataFrame):
            df = csv_path_or_df
        else:
            logger.debug("Loading %s", csv_path_or_df)
            df = pd.read_csv(csv_path_or_df)

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

            ingredients_raw = str(row["ingredients"])
            metadata = {
                "doc_type": "recipe",
                "source_file": "cleaned_recipes_data_sample",
                "recipe_name": row["title"],
                "ingredient_list": ner_list if ner_list else None,
                "allergens": self._detect_allergens(ingredients_raw),
                "diet_tags": self._detect_diet_tags(ingredients_raw),
            }

            documents.append(Document(page_content=full_text, metadata=metadata))

        logger.debug("Loaded %d documents from recipes_data_sample CSV", len(documents))
        return documents

    def _load_healthy_meals_csv(self, csv_path_or_df) -> List[Document]:
        """Load cleaned_healthy_meals.csv with numeric nutrition metadata.

        Accepts either a file path (str) or a DataFrame (for batched ingestion).
        """
        if isinstance(csv_path_or_df, pd.DataFrame):
            df = csv_path_or_df
        else:
            logger.debug("Loading %s", csv_path_or_df)
            df = pd.read_csv(csv_path_or_df)

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

            diet_tags = [row["diet_type"].lower()]
            if row["diet_type"].lower() in ["vegan", "vegetarian"]:
                diet_tags.append("vegetarian")

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
                "diet_tags": diet_tags,
                "allergens": self._detect_allergens(row["meal_name"]),
            }

            documents.append(Document(page_content=text, metadata=metadata))

        logger.debug("Loaded %d documents from healthy_meals CSV", len(documents))
        return documents

    def _load_nutrition_csv(self, csv_path_or_df) -> List[Document]:
        """Load cleaned_nutrition.csv - detailed ingredient nutrition database.

        Accepts either a file path (str) or a DataFrame (for batched ingestion).
        """
        if isinstance(csv_path_or_df, pd.DataFrame):
            df = csv_path_or_df
        else:
            logger.debug("Loading %s", csv_path_or_df)
            df = pd.read_csv(csv_path_or_df)

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
                "allergens": self._detect_allergens(row["name"]),
            }

            documents.append(Document(page_content=text, metadata=metadata))

        logger.debug("Loaded %d documents from nutrition CSV", len(documents))
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
    from settings import DATA_DIR, RECIPES_NUTRITION_VECTOR_PATH, LLM_MODEL

    # Configure logging to see debug output
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Initialize RAG system
    print("=" * 80)
    print("Initializing RecipesNutritionRAG...")
    print("=" * 80)
    
    my_rag = RecipesNutritionRAG(
        data_folder=str(DATA_DIR),
        vectorstore_path=str(RECIPES_NUTRITION_VECTOR_PATH),
        model_name="llama3:8b",
        k=14,
        temperature=0.3,
    )

    # Override LLM with OpenAI GPT-4.1 BEFORE initialize()
    my_rag.llm = ChatOpenAI(
        model="gpt-4.1",
        temperature=0.1,
        api_key=OPENAI_API_KEY,
    )

    my_rag.initialize()

    # Override LLM again AFTER initialize() (because initialize() resets it)
    my_rag.llm = ChatOpenAI(
        model="gpt-4.1",
        temperature=0.1,
        api_key=OPENAI_API_KEY,
    )

    # Rebuild the chain with the new LLM
    my_rag._build_chain()

    print("✅ Using OpenAI GPT-4.1")
    
    # Test query
    test_query = """USER REQUEST: I want something with Tofu and Rice
DIETARY RESTRICTIONS: apples, vegetarian
PREFERENCES: carrots
NUTRITION GUIDELINES: General healthy Eating
FOODS TO AVOID: Foods high in fat
FOODS TO LIMIT: sugar
INSTRUCTIONS: Tofu and Rice
Based on the above, recommend specific safe and healthy food options.
Include a brief nutritional breakdown for each."""

    print("\n" + "=" * 80)
    print("TEST QUERY:")
    print("=" * 80)
    print(test_query)
    print("\n" + "=" * 80)
    print("RAG RESPONSE:")
    print("=" * 80)
    
    # Get answer
    answer = my_rag.ask(test_query)
    
    # Print raw output
    print("\n" + answer)
    print("\n" + "=" * 80)
    print("STATS:")
    print("=" * 80)
    stats = my_rag.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")
    print("=" * 80)
