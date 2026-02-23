"""
infrastructure.rag.medical_rag - Medical dietary constraint extraction from PDFs.

Implements MedicalRAGPort. Migrated from rags/medical_rag.py with:
    - get_constraints() returns NutritionConstraints (typed) instead of raw dict
    - Async wrapper via run_in_executor
    - No sys.path.insert
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from domain.models import NutritionConstraints
from infrastructure.rag.base_rag import BaseRAG

logger = logging.getLogger(__name__)


class MedicalRAG(BaseRAG):
    """RAG system for extracting medical constraints from PDF documents."""

    SYSTEM_PROMPT = """You are a medical nutrition specialist. Based on the medical documents provided, extract dietary constraints for the patient's condition(s).

CONTEXT:
{context}

USER QUERY: {input}

You MUST respond with a valid JSON object using EXACTLY this structure:
{{
  "dietary_goals": "brief summary of dietary goals",
  "foods_to_increase": ["food1", "food2"],
  "avoid": ["foods the patient must completely avoid"],
  "limit": ["foods the patient should reduce or eat in moderation"],
  "constraints": {{
    "sugar_g": {{"max": <number or null>}},
    "sodium_mg": {{"max": <number or null>}},
    "fiber_g": {{"min": <number or null>}},
    "protein_g": {{"max": <number or null>}},
    "saturated_fat_g": {{"max": <number or null>}}
  }},
  "notes": "any additional important dietary notes"
}}

RULES:
- Use numeric daily limits in grams/milligrams where the documents provide them.
- If the documents do not specify an exact number, estimate a reasonable daily limit based on standard medical guidelines.
- "avoid" = foods that must be completely excluded.
- "limit" = foods that should be reduced but are okay in moderation.
- Always return valid JSON, nothing else."""

    def __init__(
        self,
        folder_paths: List[str],
        vectorstore_path: str,
        model_name: str = "llama3.2",
        embedding_model: str = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
        temperature: float = 0.1,
        chunk_size: int = 300,
        chunk_overlap: int = 50,
        ollama_base_url: str = "http://localhost:11434/",
        llm_provider: str = "ollama",
        openai_api_key: Optional[str] = None,
        groq_api_key: Optional[str] = None,
    ):
        # Only Ollama uses the native format="json" parameter.
        # Cloud providers (Groq, OpenAI) rely on the prompt for JSON output.
        llm_format = "json" if llm_provider == "ollama" else None

        super().__init__(
            embedding_model=embedding_model,
            vectorstore_path=vectorstore_path,
            model_name=model_name,
            temperature=temperature,
            llm_format=llm_format,
            ollama_base_url=ollama_base_url,
            llm_provider=llm_provider,
            openai_api_key=openai_api_key,
            groq_api_key=groq_api_key,
        )
        self.folder_paths = folder_paths if isinstance(folder_paths, list) else [folder_paths]
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # ================================================================
    # BaseRAG implementations
    # ================================================================

    def _ingest_documents(self) -> List[Document]:
        documents = []
        logger.info("Starting PDF ingestion from %d folder(s)", len(self.folder_paths))

        for folder_path in self.folder_paths:
            pdf_folder = Path(folder_path)
            if not pdf_folder.exists():
                logger.warning("Folder %s does not exist — skipping", folder_path)
                continue

            pdf_files = list(pdf_folder.rglob("*.pdf"))
            logger.info("Found %d PDFs in %s", len(pdf_files), folder_path)

            for pdf_file in pdf_files:
                try:
                    loader = PyPDFLoader(file_path=str(pdf_file))
                    docs = loader.load()
                    documents.extend(docs)
                except Exception as e:
                    logger.error("Error loading %s: %s", pdf_file.name, e)

        logger.info("Total documents loaded: %d", len(documents))
        return self._chunk_documents(documents)

    def _ingest_single_file(self, file_path: str) -> List[Document]:
        pdf_path = Path(file_path)
        if not pdf_path.exists():
            logger.warning("File does not exist: %s", file_path)
            return []
        try:
            loader = PyPDFLoader(file_path=str(pdf_path))
            documents = loader.load()
        except Exception as e:
            logger.error("Error loading %s: %s", pdf_path.name, e)
            return []
        return self._chunk_documents(documents)

    # ================================================================
    # Domain-specific (async)
    # ================================================================

    async def get_constraints(
        self,
        conditions: list[str],
    ) -> NutritionConstraints:
        """Get nutrition constraints for given medical conditions.

        Async wrapper around the sync RAG chain.
        Returns typed NutritionConstraints instead of raw dict.
        """
        if not conditions:
            return NutritionConstraints.default()

        if not self.rag_chain:
            logger.warning("RAG chain not initialized — returning default constraints")
            return NutritionConstraints.default()

        conditions_str = ", ".join(conditions)
        query = (
            f"Extract dietary constraints and nutrition guidelines for a patient with: "
            f"{conditions_str}. "
            "Return the JSON with avoid list, limit list, and numeric daily constraints."
        )

        logger.info("Retrieving constraints for: %s", conditions_str)

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, self.rag_chain.invoke, {"input": query},
        )
        answer = response.get("answer", "")

        return self._parse_constraints_response(answer)

    # ================================================================
    # Private helpers
    # ================================================================

    def _chunk_documents(self, documents: List[Document]) -> List[Document]:
        if not documents:
            return documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        chunks = text_splitter.split_documents(documents)
        for i, chunk in enumerate(chunks):
            chunk.metadata["id"] = f"chunk_{i}"
        logger.info("Split into %d chunks", len(chunks))
        return chunks

    @staticmethod
    def _parse_constraints_response(answer: Any) -> NutritionConstraints:
        """Parse LLM response into typed NutritionConstraints."""
        if isinstance(answer, str):
            try:
                answer = json.loads(answer)
            except (json.JSONDecodeError, TypeError):
                logger.warning("Could not parse LLM response as JSON")
                return NutritionConstraints(notes=answer)

        if not isinstance(answer, dict):
            return NutritionConstraints.default()

        # Handle dietary_goals as string or list
        goals = answer.get("dietary_goals", [])
        if isinstance(goals, str):
            goals = [goals]

        return NutritionConstraints(
            dietary_goals=goals,
            foods_to_increase=answer.get("foods_to_increase", []),
            avoid=answer.get("avoid", []),
            limit=answer.get("limit", []),
            constraints=answer.get("constraints", {}),
            notes=answer.get("notes", ""),
        )
