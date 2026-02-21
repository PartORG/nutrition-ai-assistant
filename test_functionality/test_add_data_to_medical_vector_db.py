"""
Add new data to MedicalRAG vectorstore

Instructions:
- This script adds new data to the MedicalRAG vector database only.
- Use when you want to append new medical data without rebuilding the entire vectorstore.
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
    medical_rag = factory._medical_rag
    # Specify the file path to add (PDF)
    file_path = input("Enter the path to the medical PDF file to add: ")
    medical_rag.initialize()
    chunks_added = medical_rag.add_documents(file_path)
    print(f"Added {chunks_added} chunks from {file_path} to MedicalRAG vectorstore.")

if __name__ == "__main__":
    asyncio.run(main())
