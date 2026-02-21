import sys
import os
import asyncio

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from infrastructure.config import Settings
from factory import ServiceFactory
from application.context import SessionContext
from domain.entities import User
from agent.tools.search_recipes import SearchRecipesTool

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
    tool = SearchRecipesTool(factory.create_recommendation_service())
    query = "Recommend a low-carb dinner"
    result = await tool.execute(ctx, query=query)
    print("Tool Output:", result.output)

if __name__ == "__main__":
    asyncio.run(main())
