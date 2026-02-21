"""
Test Intent Parser

Instructions:
- This script tests the OllamaIntentParser for extracting structured user intent.
- You can modify the 'text' string to test different user requests (e.g. "Show me recipes for diabetes").
- Prints parsed UserIntent result.
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
    parser = factory._intent_parser
    text = "Show me recipes for diabetes"  # Modify this string to test other requests
    intent = await parser.parse(text)
    print("Parsed Intent:", intent)

if __name__ == "__main__":
    asyncio.run(main())
