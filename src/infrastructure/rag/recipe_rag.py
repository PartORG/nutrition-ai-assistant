"""
infrastructure.rag.recipe_rag - Personalized recipe recommendations via dual vectorstores.

Implements RecipeRAGPort. Migrated from rags/recipes_nutrition_rag.py with:
    - SmartRetriever extracted to smart_retriever.py
    - ask() has async wrapper
    - No sys.path.insert
    - Imports updated for new structure
"""

from __future__ import annotations

import ast
import asyncio
import json
import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Union

import pandas as pd
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy

from infrastructure.rag.base_rag import BaseRAG
from infrastructure.rag.smart_retriever import SmartRetriever

logger = logging.getLogger(__name__)


class RecipeNutritionRAG(BaseRAG):
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
1. **CRITICAL**: Foods to Avoid (allergies, medical contraindications) — NEVER include these
2. **HIGH**: Dietary Restrictions (vegetarian, vegan, gluten-free, etc.)
3. **MEDIUM**: Nutrition Guidelines (calorie limits, macronutrient targets)
4. **LOW**: Preferences (cuisine, cooking time, taste)

If no recipe in {{context}} meets all critical requirements, silently adapt the closest matches.

## RECIPE QUANTITY

- **DEFAULT**: Always provide 3 recipes (unless specified in Instructions)
- If Instructions say "I want ONE recipe" → Provide 1
- If Instructions say "give me 5 options" → Provide 5
- For ambiguous requests → Provide 3

## RECIPE ADAPTATION PROTOCOL

**When to Adapt:**
- Recipe contains Foods to Avoid → Replace with safe alternatives
- Recipe exceeds Nutrition Guidelines → Adjust portions or substitute ingredients
- Recipe violates Dietary Restrictions → Substitute non-compliant ingredients
- User requests specific ingredients in Instructions → Incorporate them if possible

**Adaptation Rules:**
- Make adaptations SILENTLY (do not mention original recipe or changes)
- Maintain recipe authenticity (don't change cuisine character drastically)
- Preserve cooking method when possible (oven → oven, not oven → microwave)
- Calculate new nutritional values after adaptations

**Example:**
- User: vegetarian, Recipe: chicken pasta
- **Adapt:** Replace chicken with chickpeas or tofu (same protein content)
- **Output:** Present as "Chickpea Pasta" (not "Adapted Chicken Pasta")

## MEASUREMENT CONVERSIONS (MANDATORY)

Convert ALL imperial measurements to metric in the output:
- Cups → ml/g (1 cup ≈ 240 ml liquid, 240 g solid)
- tbsp → ml (1 tbsp = 15 ml), tsp → ml (1 tsp = 5 ml)
- oz → g (1 oz ≈ 28 g), lb → g (1 lb ≈ 454 g)
- °F → °C (350°F→175°C, 375°F→190°C, 400°F→200°C, 425°F→220°C)

If context already uses metric, keep as-is.

## HANDLING MISSING DATA

**Missing Ingredient Quantities:**
- Estimate from context nutritional info and serving size
- Seasonings: use standard amounts (salt: 5g, pepper: 2g, herbs: 10g)

**Missing Cooking Instructions:**
- Create logical steps based on ingredient types and cooking method
- Include temperatures in Celsius, number each step clearly

**Missing Nutritional Values:**
- Calculate from individual ingredients using standard USDA values
- Never leave nutrition fields empty

## SPECIAL CASES

**"use ingredients from fridge/home [list]"**
→ PRIORITIZE recipes using those listed ingredients
→ If no exact match, incorporate ingredients into closest recipe

**"I want [specific food]"**
→ MUST include that food in at least one recipe

**Foods to Limit with amounts (e.g. "sugar max 10g")**
→ Select recipes within limit per serving
→ If context recipe exceeds limit, reduce proportionally

## OUTPUT FORMAT

Return ONLY valid JSON — no markdown fences, no explanations, no extra text.

**CRITICAL OUTPUT RULES:**
1. Root object MUST have single key "recipes" with array value
2. ONLY use these field names: name, why_recommended, servings, prep_time, cook_time, ingredients, cook_instructions, nutrition
3. FORBIDDEN fields: "Meal", "Diet Type", "Preparation", "Key Vitamins", "Key Minerals", "Macronutrients"
4. Ingredients MUST be a flat string array — NOT nested objects
5. All metric, all complete, no null values

**JSON Structure:**
{{
  "recipes": [
    {{
      "name": "Adapted recipe name (not original if modified)",
      "why_recommended": "One sentence referencing user requirements (max 150 chars)",
      "servings": 2,
      "prep_time": "15 minutes",
      "cook_time": "20 minutes",
      "ingredients": ["200g chicken breast", "150ml olive oil", "2 cloves garlic"],
      "cook_instructions": "1. Preheat oven to 180°C.\\n2. Season chicken.\\n3. Bake 25 min.",
      "nutrition": {{
        "calories": 350,
        "protein_g": 25.0,
        "carbs_g": 30.0,
        "fat_g": 12.0,
        "fiber_g": 5.0,
        "sodium_mg": 400,
        "sugar_g": 3.0,
        "saturated_fat_g": 2.0
      }}
    }}
  ]
}}

**Field Format Rules:**
- `ingredients`: Start with quantity: "200g chicken" (not "chicken 200g"). Metric only.
- `cook_instructions`: Single string, numbered steps separated by \\n. Temperatures in °C.
- `prep_time` / `cook_time`: Always include unit: "15 minutes" (not "15").
- `why_recommended`: Reference at least one user requirement. Max 150 characters.
- `nutrition`: All 8 fields required. Estimate if unknown.

---

User Input:
{input}

Retrieved Context:
{context}

Generate JSON with the exact number of recipes requested (default 3), adapting recipes silently to meet all requirements from highest to lowest priority. Convert all measurements to metric. Return ONLY the JSON object."""

    def __init__(
        self,
        data_folder: str,
        vectorstore_path: str,
        model_name: str = "llama3.2",
        temperature: float = 0.3,
        k: int = 14,
        ollama_base_url: str = "http://localhost:11434/",
        llm_provider: str = "ollama",
        openai_api_key: Optional[str] = None,
        groq_api_key: Optional[str] = None,
    ):
        # Only Ollama uses the native format="json" parameter.
        # Cloud providers (Groq, OpenAI) rely on the prompt for JSON output.
        llm_format = "json" if llm_provider == "ollama" else None

        super().__init__(
            vectorstore_path=vectorstore_path,
            model_name=model_name,
            temperature=temperature,
            ollama_base_url=ollama_base_url,
            llm_format=llm_format,
            llm_provider=llm_provider,
            openai_api_key=openai_api_key,
            groq_api_key=groq_api_key,
        )
        self.data_folder = Path(data_folder)
        self.k = k

        self.vectorstore_recipes: Optional[FAISS] = None
        self.vectorstore_nutrition: Optional[FAISS] = None
        self.smart_retriever: Optional[SmartRetriever] = None

    # ================================================================
    # RecipeRAGPort implementation
    # ================================================================

    async def async_ask(self, query: str) -> list:
        """Query the RAG and return structured Recipe objects.

        The LLM is forced to output JSON (llm_format='json').
        We parse that JSON directly into Recipe domain objects —
        no second LLM re-parsing step needed.
        """
        from domain.models import Recipe, NutritionValues
        loop = asyncio.get_event_loop()
        raw_json = await loop.run_in_executor(None, super().ask, query)
        return self._parse_json_to_recipes(raw_json)

    @staticmethod
    def _parse_json_to_recipes(raw_json: str) -> list:
        """Parse the LLM JSON output into a list of Recipe domain objects."""
        from domain.models import Recipe, NutritionValues
        import re

        # Extract JSON object (guard against any stray text)
        match = re.search(r'\{.*\}', raw_json, re.DOTALL)
        if not match:
            logger.warning("Recipe RAG returned no JSON object: %s", raw_json[:200])
            return []

        try:
            data = json.loads(match.group())
        except json.JSONDecodeError as e:
            logger.warning("Recipe RAG JSON parse error: %s", e)
            return []

        recipes = []
        for r in data.get("recipes", []):
            nd = r.get("nutrition") or {}
            nutrition = NutritionValues(
                calories=_to_float(nd.get("calories")),
                protein_g=_to_float(nd.get("protein_g")),
                carbs_g=_to_float(nd.get("carbs_g")),
                fat_g=_to_float(nd.get("fat_g")),
                fiber_g=_to_float(nd.get("fiber_g")),
                sodium_mg=_to_float(nd.get("sodium_mg")),
                sugar_g=_to_float(nd.get("sugar_g")),
                saturated_fat_g=_to_float(nd.get("saturated_fat_g")),
            )
            # cook_instructions may be a string or a list
            instructions = r.get("cook_instructions", "")
            if isinstance(instructions, list):
                instructions = "\n".join(str(s) for s in instructions)

            raw_ingredients = r.get("ingredients", [])
            if isinstance(raw_ingredients, list):
                normalized_ingredients = []
                for ing in raw_ingredients:
                    if isinstance(ing, str):
                        normalized_ingredients.append(ing)
                    elif isinstance(ing, dict):
                        # LLM returned {"name": "...", "amount": "..."} instead of a plain string
                        parts = [str(v) for v in ing.values() if v]
                        normalized_ingredients.append(" ".join(parts))
                    else:
                        normalized_ingredients.append(str(ing))
            else:
                normalized_ingredients = []

            recipes.append(Recipe(
                name=r.get("name", "").strip(),
                ingredients=normalized_ingredients,
                nutrition=nutrition,
                why_recommended=r.get("why_recommended", ""),
                servings=int(r.get("servings") or 1),
                prep_time=str(r.get("prep_time") or ""),
                cook_instructions=instructions,
            ))

        logger.info("Recipe RAG parsed %d recipes from JSON", len(recipes))
        return recipes

    # ================================================================
    # BaseRAG implementations
    # ================================================================

    def _ingest_documents(self) -> List[Document]:
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

        recipes1 = all_loaded.get("cleaned_recipes.csv", [])
        recipes2 = all_loaded.get("cleaned_recipes_data_sample.csv", [])
        meals = all_loaded.get("cleaned_healthy_meals.csv", [])
        nutrition = all_loaded.get("cleaned_nutrition.csv", [])

        for doc in recipes1 + recipes2 + meals:
            doc.metadata["_collection"] = "recipes"
        for doc in nutrition:
            doc.metadata["_collection"] = "nutrition"

        all_docs = recipes1 + recipes2 + meals + nutrition
        logger.info(
            "Loaded documents — Recipes/Meals: %d, Nutrition: %d",
            len(recipes1) + len(recipes2) + len(meals), len(nutrition),
        )
        return all_docs

    def _ingest_single_file(self, file_path: str) -> List[Document]:
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
    # Dual vectorstore overrides
    # ================================================================

    def _vectorstore_exists(self) -> bool:
        recipes_path = self.vectorstore_path / "recipes_and_meals_db"
        nutrition_path = self.vectorstore_path / "nutrition_facts_db"
        return recipes_path.exists() and nutrition_path.exists()

    def _build_vectorstore(self, documents: List[Document]) -> None:
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

        self.vectorstore_path.mkdir(parents=True, exist_ok=True)
        self.vectorstore_recipes.save_local(str(self.vectorstore_path / "recipes_and_meals_db"))
        self.vectorstore_nutrition.save_local(str(self.vectorstore_path / "nutrition_facts_db"))
        logger.info("Vectorstores saved to %s", self.vectorstore_path)

    def _load_vectorstore(self) -> None:
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
        self.smart_retriever = SmartRetriever(
            vectorstore_recipes=self.vectorstore_recipes,
            vectorstore_nutrition=self.vectorstore_nutrition,
            k=self.k,
        )
        self.retriever = self.smart_retriever
        logger.info("SmartRetriever created (k=%d)", self.k)

    # ================================================================
    # Batched CSV ingestion
    # ================================================================

    def _ingest_large_csv(
        self,
        csv_path: str,
        loader_fn,
        collection: str,
        chunksize: int = 5000,
    ) -> None:
        path = Path(csv_path)
        if not path.exists():
            logger.error("CSV file not found: %s", csv_path)
            return

        if self.embeddings is None:
            raise RuntimeError("Call initialize() before _ingest_large_csv()")

        if collection == "recipes":
            target = self.vectorstore_recipes
        elif collection == "nutrition":
            target = self.vectorstore_nutrition
        else:
            raise ValueError(f"Unknown collection: {collection}")

        meta = self._get_batch_meta(csv_path)
        start_offset = meta.get("rows_indexed", 0)

        total_rows = meta.get("total_rows")
        if total_rows is None:
            total_rows = sum(1 for _ in open(csv_path, encoding="utf-8")) - 1
            meta["total_rows"] = total_rows

        total_batches = (total_rows + chunksize - 1) // chunksize
        rows_processed = start_offset
        batch_num = start_offset // chunksize

        logger.info(
            "Starting batched ingestion of %s (%d total rows, resuming from row %d)",
            path.name, total_rows, start_offset,
        )

        reader = pd.read_csv(csv_path, chunksize=chunksize)
        for chunk_df in reader:
            if rows_processed + len(chunk_df) <= start_offset:
                rows_processed += len(chunk_df)
                batch_num += 1
                continue

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

            meta["rows_indexed"] = rows_processed
            self._save_batch_meta(csv_path, meta)

            logger.info(
                "Batch %d/%d — %d/%d rows indexed",
                batch_num, total_batches, rows_processed, total_rows,
            )

        logger.info("Batched ingestion complete: %s (%d rows)", path.name, rows_processed)

    def _batch_meta_path(self, csv_path: str) -> Path:
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
    # CSV data loaders
    # ================================================================

    @staticmethod
    def _detect_allergens(text: str) -> List[str]:
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
        text_lower = text.lower()
        tags = []
        if "vegan" in text_lower:
            tags.append("vegan")
            tags.append("vegetarian")
        elif not any(meat in text_lower for meat in ["chicken", "beef", "pork", "fish", "meat", "lamb"]):
            tags.append("vegetarian")
        return tags

    def _load_recipes_csv(self, csv_path_or_df) -> List[Document]:
        if isinstance(csv_path_or_df, pd.DataFrame):
            df = csv_path_or_df
        else:
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
            metadata: Dict[str, Any] = {
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
                prep_mins = sum(int(s) * (60 if "hr" in prep_str else 1) for s in re.findall(r"\d+", prep_str))
                metadata["prep_time_min"] = prep_mins
            if pd.notna(row.get("cook_time")):
                cook_str = str(row["cook_time"]).lower()
                cook_mins = sum(int(s) * (60 if "hr" in cook_str else 1) for s in re.findall(r"\d+", cook_str))
                metadata["cook_time_min"] = cook_mins

            ingredients_text = str(row["ingredients"])
            metadata["allergens"] = self._detect_allergens(ingredients_text)
            metadata["diet_tags"] = self._detect_diet_tags(ingredients_text)
            documents.append(Document(page_content=full_text, metadata=metadata))

        logger.debug("Loaded %d documents from recipes CSV", len(documents))
        return documents

    def _load_recipes_data_sample_csv(self, csv_path_or_df) -> List[Document]:
        if isinstance(csv_path_or_df, pd.DataFrame):
            df = csv_path_or_df
        else:
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
        if isinstance(csv_path_or_df, pd.DataFrame):
            df = csv_path_or_df
        else:
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
        if isinstance(csv_path_or_df, pd.DataFrame):
            df = csv_path_or_df
        else:
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
        if not self.smart_retriever:
            return []
        return self.smart_retriever.invoke(query)

    def reload_vectorstores(self) -> None:
        self._load_vectorstore()
        self._setup_retriever()
        self._build_chain()

    def get_stats(self) -> Dict[str, Any]:
        stats = super().get_stats()
        if self.vectorstore_recipes and self.vectorstore_nutrition:
            stats["vectorstores"] = {
                "recipes_and_meals": {"vectors": self.vectorstore_recipes.index.ntotal},
                "nutrition_facts": {"vectors": self.vectorstore_nutrition.index.ntotal},
            }
        stats["k"] = self.k
        stats["data_folder"] = str(self.data_folder)
        return stats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_float(value) -> Optional[float]:
    """Safely coerce a JSON value to float, returning None on failure."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
