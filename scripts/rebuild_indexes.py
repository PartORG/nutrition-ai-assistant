"""
Recreate vector DBs from CSVs

Instructions:
- This script rebuilds MedicalRAG and RecipeNutritionRAG vector databases from source data.
- Use when you want to refresh or update the vectorstores after changing data files.
- Prints confirmation when vector DBs are recreated.
"""
import asyncio
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from factory import ServiceFactory
from infrastructure.config import Settings


async def main():
    settings = Settings.from_env()
    factory = ServiceFactory(settings)
    await factory.initialize()
    print("Rebuilding MedicalRAG vectorstore...")
    factory._medical_rag.initialize(force_rebuild=True)
    print("Rebuilding RecipeNutritionRAG vectorstore...")
    factory._recipe_rag.initialize(force_rebuild=True)
    print("Vector DBs recreated.")

if __name__ == "__main__":
    asyncio.run(main())
