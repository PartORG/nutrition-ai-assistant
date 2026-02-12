"""
MedicalRAG - RAG system for extracting medical dietary constraints from PDF documents.

Inherits from BaseRAG and adds:
    - PDF ingestion from multiple folders
    - Chunking with RecursiveCharacterTextSplitter
    - Medical nutrition extraction prompt
    - get_constraints() for structured constraint retrieval
"""

import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Any

from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from pipeline.config import PDF_DIR, MEDICAL_VECTORSTORE_PATH, LLM_MODEL
from pipeline.base_rag import BaseRAG

logger = logging.getLogger(__name__)


class MedicalRAG(BaseRAG):
    """RAG system for extracting medical constraints from medical PDF documents."""

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
        embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
        temperature: float = 0.3,
        chunk_size: int = 300,
        chunk_overlap: int = 50,
    ):
        super().__init__(
            embedding_model=embedding_model,
            vectorstore_path=vectorstore_path,
            model_name=model_name,
            temperature=temperature,
            llm_format="json",
        )
        self.folder_paths = folder_paths if isinstance(folder_paths, list) else [folder_paths]
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._start_time = None

    # ================================================================
    # BaseRAG abstract method implementations
    # ================================================================

    def _get_system_prompt(self) -> str:
        if hasattr(self, "_custom_prompt"):
            return self._custom_prompt
        return self.SYSTEM_PROMPT

    def _ingest_documents(self) -> List[Document]:
        """Load PDFs from multiple directory paths and split into chunks."""
        documents = []

        for folder_path in self.folder_paths:
            pdf_folder = Path(folder_path)

            if not pdf_folder.exists():
                logger.warning(f"Folder {folder_path} does not exist — skipping")
                continue

            pdf_files = list(pdf_folder.rglob("*.pdf"))
            logger.info(f"Found {len(pdf_files)} PDFs in {folder_path}")

            for pdf_file in pdf_files:
                try:
                    logger.debug(f"Loading {pdf_file.name}...")
                    loader = PyPDFLoader(file_path=str(pdf_file))
                    docs = loader.load()
                    documents.extend(docs)
                except Exception as e:
                    logger.error(f"Error loading {pdf_file.name}: {e}")

        logger.info(f"Total documents loaded: {len(documents)}")

        # Chunk the documents
        if documents:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
            chunks = text_splitter.split_documents(documents)
            for i, chunk in enumerate(chunks):
                chunk.metadata["id"] = f"chunk_{i}"
            logger.info(f"Split into {len(chunks)} chunks")
            return chunks

        return documents

    # ================================================================
    # Domain-specific public methods
    # ================================================================

    def get_constraints(self, conditions: List[str]) -> Dict[str, Any]:
        """Get nutrition constraints for given medical conditions."""
        if not conditions:
            return self._default_constraints()

        if not self.rag_chain:
            logger.warning("RAG chain not initialized — returning default constraints")
            return self._default_constraints()

        query = (
            f"Extract dietary constraints and nutrition guidelines for a patient with: "
            f"{', '.join(conditions)}. "
            "Return the JSON with avoid list, limit list, and numeric daily constraints."
        )

        response = self.rag_chain.invoke({"input": query})
        answer = response.get("answer", "")

        # The LLM is configured with format="json", so parse the response
        if isinstance(answer, str):
            try:
                answer = json.loads(answer)
            except (json.JSONDecodeError, TypeError):
                logger.warning("Could not parse LLM response as JSON, wrapping as raw text")
                answer = {
                    "raw_response": answer,
                    "constraints": {},
                    "avoid": [],
                    "limit": [],
                }

        return answer

    def _default_constraints(self) -> Dict[str, Any]:
        """Return default constraints when no conditions are provided."""
        return {
            "dietary_goals": "General healthy eating guidelines",
            "foods_to_increase": ["whole grains", "vegetables", "fruits", "lean proteins"],
            "avoid": [],
            "limit": [],
            "constraints": {
                "sugar_g": {"max": None},
                "sodium_mg": {"max": None},
                "fiber_g": {"min": None},
                "protein_g": {"max": None},
                "saturated_fat_g": {"max": None},
            },
            "notes": "No specific medical conditions provided",
        }


if __name__ == "__main__":
    medical_rag = MedicalRAG(
        folder_paths=[str(PDF_DIR)],
        model_name=LLM_MODEL,
        vectorstore_path=str(MEDICAL_VECTORSTORE_PATH),
    )
    medical_rag.initialize(force_rebuild=True)
    print("\nMedical RAG ready!")
    medical_rag.ask(query="hello")
