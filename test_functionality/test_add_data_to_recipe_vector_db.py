"""
Add new data to RecipeNutritionRAG vectorstore

Instructions:
- This script adds new data to the RecipeNutritionRAG vector database only.
- Use when you want to append new recipe/nutrition data without rebuilding the entire vectorstore.
- Prints confirmation when new data is added.
"""
import sys
import os
import asyncio

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from infrastructure.config import Settings
from factory import ServiceFactory

async def main():
    settings = Settings.from_env()
    factory = ServiceFactory(settings)
    await factory.initialize()
    recipe_rag = factory._recipe_rag
    file_path = input("Enter the path to the recipe/nutrition file to add: ")
    recipe_rag.initialize()
    # Enhancement: check file size and use batched ingestion for large CSVs
    import os
    from pathlib import Path
    path = Path(file_path)
    threshold_bytes = recipe_rag.LARGE_FILE_THRESHOLD_MB * 1024 * 1024
    if path.suffix == ".csv" and path.stat().st_size > threshold_bytes:
        print(f"Large CSV detected ({path.stat().st_size / (1024*1024):.1f} MB), using batched ingestion...")
        loader_map = {
            "cleaned_recipes.csv": recipe_rag._load_recipes_csv,
            "cleaned_recipes_data_sample.csv": recipe_rag._load_recipes_data_sample_csv,
            "cleaned_healthy_meals.csv": recipe_rag._load_healthy_meals_csv,
            "cleaned_nutrition.csv": recipe_rag._load_nutrition_csv,
        }
        loader_fn = loader_map.get(path.name)
        collection = "recipes" if "recipe" in path.name or "meal" in path.name else "nutrition"
        if loader_fn:
            recipe_rag._ingest_large_csv(str(path), loader_fn, collection)
            print(f"Batched ingestion complete for {file_path}.")
        else:
            print(f"No loader found for file: {path.name}. Skipping.")
    else:
        chunks_added = recipe_rag.add_documents(file_path)
        print(f"Added {chunks_added} chunks from {file_path} to RecipeNutritionRAG vectorstore.")

if __name__ == "__main__":
    asyncio.run(main())
