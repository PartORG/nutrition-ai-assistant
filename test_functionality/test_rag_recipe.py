"""
Test Recipe RAG retrieval

Instructions:
- This script tests the RecipeNutritionRAG system for personalized recipe recommendations.
- Dummy user is ensured in DB.
- You can modify the 'query' string to test different recipe requests (e.g. "Healthy vegan breakfast recipes").
- Prints structured recipe recommendations for the query.

SessionContext fields:
- user_id: int (must match a user in DB)
- conversation_id: str (unique for each session)
- user_data: dict (user profile info, can be minimal for test)
"""
import sys
import os
import asyncio

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from infrastructure.config import Settings
from factory import ServiceFactory
from application.context import SessionContext
from domain.entities import User

async def ensure_dummy_user(factory, user_id=1):
    user_repo = factory.create_user_repository()
    user = await user_repo.get_by_id(user_id)
    if not user:
        dummy_user = User(
            id=user_id,
            name="Test",
            surname="User",
            user_name="testuser",
            caretaker="",
            age=30,
            gender="male",
        )
        await user_repo.save(dummy_user)
        print(f"Dummy user created: {dummy_user}")
    else:
        print(f"Dummy user already exists: {user}")

async def main():
    settings = Settings.from_env()
    factory = ServiceFactory(settings)
    await factory.initialize()
    await ensure_dummy_user(factory, user_id=1)
    ctx = SessionContext(user_id=1, conversation_id="test_convo_1", user_data={"name": "Test User"})
    rag = factory._recipe_rag
    query = "Healthy vegan breakfast recipes"  # Modify this string to test other queries
    results = await rag.async_ask(query)
    print("Recipe RAG Results:", results)

if __name__ == "__main__":
    asyncio.run(main())
