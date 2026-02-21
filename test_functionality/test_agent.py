"""
Test Agent Response

Instructions:
- This script allows you to chat interactively with the Nutrition AI Agent.
- The dummy user is created automatically if not present.
- You can type any question or request, e.g. "Recommend a low-carb dinner".
- Type 'quit' or 'exit' to end the chat.

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
    ctx = SessionContext(
        user_id=1,
        conversation_id="test_convo_1",
        user_data={"name": "Test User"},
    )
    agent = factory.create_agent(ctx)
    print("\n--- Nutrition AI Assistant Chat ---")
    while True:
        query = input("You: ")
        if query.strip().lower() in {"quit", "exit"}:
            print("Exiting chat.")
            break
        response = await agent.run(ctx, query)
        print("Agent:", response)

if __name__ == "__main__":
    asyncio.run(main())
