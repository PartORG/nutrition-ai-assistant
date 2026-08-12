"""
infrastructure.rag.recipe_ingest - CSV -> Document loading for the recipe corpus.

Pure data shaping: read a recipe CSV, produce LangChain Documents carrying the
metadata the retriever needs. Nothing here touches a vectorstore, an LLM or an
embedding model — which is why it lives apart from RecipeNutritionRAG. That
class implements RecipeRAGPort, and ~180 lines of pandas glue made that hard
to see.
"""

from __future__ import annotations

import ast
import json
import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def detect_allergens(text: str) -> list[str]:
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

def detect_diet_tags(text: str) -> list[str]:
    text_lower = text.lower()
    tags = []
    if "vegan" in text_lower:
        tags.append("vegan")
        tags.append("vegetarian")
    elif not any(meat in text_lower for meat in ["chicken", "beef", "pork", "fish", "meat", "lamb"]):
        tags.append("vegetarian")
    return tags

def load_recipes_csv(csv_path_or_df) -> list[Document]:
    if isinstance(csv_path_or_df, pd.DataFrame):
        df = csv_path_or_df
    else:
        df = pd.read_csv(csv_path_or_df)

    documents = []
    for _idx, row in df.iterrows():
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
        metadata: dict[str, Any] = {
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
        metadata["allergens"] = detect_allergens(ingredients_text)
        metadata["diet_tags"] = detect_diet_tags(ingredients_text)
        documents.append(Document(page_content=full_text, metadata=metadata))

    logger.debug("Loaded %d documents from recipes CSV", len(documents))
    return documents

def load_recipes_data_sample_csv(csv_path_or_df) -> list[Document]:
    if isinstance(csv_path_or_df, pd.DataFrame):
        df = csv_path_or_df
    else:
        df = pd.read_csv(csv_path_or_df)

    documents = []
    for _idx, row in df.iterrows():
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
            "allergens": detect_allergens(ingredients_raw),
            "diet_tags": detect_diet_tags(ingredients_raw),
        }
        documents.append(Document(page_content=full_text, metadata=metadata))

    logger.debug("Loaded %d documents from recipes_data_sample CSV", len(documents))
    return documents

def load_healthy_meals_csv(csv_path_or_df) -> list[Document]:
    if isinstance(csv_path_or_df, pd.DataFrame):
        df = csv_path_or_df
    else:
        df = pd.read_csv(csv_path_or_df)

    documents = []
    for _idx, row in df.iterrows():
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
            "allergens": detect_allergens(row["meal_name"]),
        }
        documents.append(Document(page_content=text, metadata=metadata))

    logger.debug("Loaded %d documents from healthy_meals CSV", len(documents))
    return documents


def batch_meta_path(vectorstore_path: Path, csv_path: str) -> Path:
    """Checkpoint file letting a large batched ingest resume where it stopped."""
    name = Path(csv_path).stem
    return Path(vectorstore_path) / f"{name}_batch_meta.json"


def get_batch_meta(vectorstore_path: Path, csv_path: str) -> dict[str, Any]:
    meta_path = batch_meta_path(vectorstore_path, csv_path)
    if not meta_path.exists():
        return {}
    with open(meta_path, encoding="utf-8") as f:
        return json.load(f)


def save_batch_meta(vectorstore_path: Path, csv_path: str, meta: dict[str, Any]) -> None:
    meta_path = batch_meta_path(vectorstore_path, csv_path)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

