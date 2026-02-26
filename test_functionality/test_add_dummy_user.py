"""
Add dummy user to DB

Instructions:
- This script creates a dummy user in the database for testing purposes.
- You can modify the dummy_user fields to test different user profiles.
- Prints confirmation when user is added.
"""
import sys
import os
import asyncio

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from infrastructure.config import Settings
from factory import ServiceFactory
from domain.entities import User

async def main():
    settings = Settings.from_env()
    factory = ServiceFactory(settings)
    await factory.initialize()
    user_repo = factory.create_user_repository()
    dummy_user = User(
        id=1,
        name="Test",
        surname="User",
        user_name="testuser",
        caretaker="",
        age=30,
        gender="male",
    )  # Modify fields above to test other user profiles
    # Check if user already exists by name and surname
    existing = await user_repo.get_by_name(dummy_user.name, dummy_user.surname)
    if existing:
        print(f"User '{dummy_user.name} {dummy_user.surname}' already exists in DB.")
    else:
        await user_repo.save(dummy_user)
        print("Dummy user added.")

if __name__ == "__main__":
    asyncio.run(main())
