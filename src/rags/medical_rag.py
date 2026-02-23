"""
MedicalRAG - RAG system for extracting medical dietary constraints from PDF documents.

Inherits from BaseRAG and adds:
    - PDF ingestion from multiple folders
    - Chunking with RecursiveCharacterTextSplitter
    - Medical nutrition extraction prompt
    - get_constraints() for structured constraint retrieval
    - Single-file ingestion for incremental PDF processing
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any

from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from rags.base_rag import BaseRAG

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

        logger.info(
            "MedicalRAG configured — folders: %s, chunk_size: %d, chunk_overlap: %d",
            self.folder_paths, self.chunk_size, self.chunk_overlap,
        )

    # ================================================================
    # BaseRAG abstract method implementations
    # ================================================================

    def _ingest_documents(self) -> List[Document]:
        """Load PDFs from all configured directory paths and split into chunks."""
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
                    logger.debug("Loading %s...", pdf_file.name)
                    loader = PyPDFLoader(file_path=str(pdf_file))
                    docs = loader.load()
                    documents.extend(docs)
                except Exception as e:
                    logger.error("Error loading %s: %s", pdf_file.name, e)

        logger.info("Total documents loaded: %d", len(documents))
        return self._chunk_documents(documents)

    def _ingest_single_file(self, file_path: str) -> List[Document]:
        """Load a single PDF file and split into chunks."""
        pdf_path = Path(file_path)
        if not pdf_path.exists():
            logger.warning("File does not exist: %s", file_path)
            return []

        try:
            logger.info("Loading PDF: %s", pdf_path.name)
            loader = PyPDFLoader(file_path=str(pdf_path))
            documents = loader.load()
        except Exception as e:
            logger.error("Error loading %s: %s", pdf_path.name, e)
            return []

        return self._chunk_documents(documents)

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

        if isinstance(conditions, list):
            conditions = ", ".join(conditions)

        query = (
            f"Extract dietary constraints and nutrition guidelines for a patient with: "
            f"{conditions}. "
            "Return the JSON with avoid list, limit list, and numeric daily constraints."
        )

        logger.info("Retrieving constraints for: %s", conditions)
        response = self.rag_chain.invoke({"input": query})
        answer = response.get("answer", "")

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

        logger.info("Constraints retrieval completed")
        return answer

    # ================================================================
    # Private helpers
    # ================================================================

    def _chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks using RecursiveCharacterTextSplitter."""
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
    from settings import PDF_DIR, MEDICAL_VECTORSTORE_PATH, LLM_MODEL

    medical_rag = MedicalRAG(
        folder_paths=[str(PDF_DIR)],
        model_name=LLM_MODEL,
        vectorstore_path=str(MEDICAL_VECTORSTORE_PATH),
        embedding_model="sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
    )
    medical_rag.initialize(force_rebuild=True)
    logger.info("Medical RAG ready!")
    medical_rag.ask(query="hello")
