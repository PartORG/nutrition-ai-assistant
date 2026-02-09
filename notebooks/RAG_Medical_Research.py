# Core imports
import json
import re
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.output_parsers import JsonOutputParser

# Ollama LLM
from langchain_ollama import OllamaLLM

# Data processing
import pandas as pd
import numpy as np

print("All imports successful!")


# Configuration paths
PDF_FOLDER = [r"C:\Users\peter\Desktop\ds_ai\repo_folder\nutrition-ai-assistant\data\raw\Parkinson",
    r"C:\Users\peter\Desktop\ds_ai\repo_folder\nutrition-ai-assistant\data\raw\MS"
]
NUTRITION_DATA_PATH = "../data/raw/nutrition.xlsx"
MEDICAL_VECTORSTORE_PATH = "../data/processed/medical_pdfs_vectorstore"
NUTRITION_VECTORSTORE_PATH = "../data/processed/nutrition_vectorstore"

# LLM Configuration
LLM_MODEL = "llama3.2"

print(f"PDF folder: {PDF_FOLDER}")
print(f"Nutrition data: {NUTRITION_DATA_PATH}")


class MedicalRAG:
    """RAG system for extracting medical constraints from medical documents."""
    
    def __init__(self, llm, folder_paths: List[str], vectorstore_path: str, chunk_size = 300, chunk_overlap = 50):
        self.llm = OllamaLLM(model=LLM_MODEL, temperature=0.3, format="json")
        self.folder_paths = folder_paths if isinstance(folder_paths, list) else [folder_paths]
        self.vectorstore_path = vectorstore_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name='sentence-transformers/all-mpnet-base-v2',
            encode_kwargs={"normalize_embeddings": True}
        )
        
        self.vectorstore = None
        self.retriever = None
        self.rag_chain = None
        
    def initialize(self, force_rebuild: bool = False):
        """Builds or loads the knowledge base and prepares the chain."""
        print("Initializing Medical RAG System...")
        
        if not force_rebuild and Path(self.vectorstore_path).exists():
            print(f"Loading existing index from {self.vectorstore_path}")
            self.vectorstore = FAISS.load_local(
                self.vectorstore_path, 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )
        else:
            print("No index found or force_rebuild=True. Processing documents...")
            docs = self._ingest_documents()
            
            if not docs:
                print("WARNING: No PDF documents found. Creating empty RAG system.")
                return
            
            chunks = self._split_data(docs)
            
            if not chunks:
                print("WARNING: No chunks created from documents.")
                return
            
            self.vectorstore = FAISS.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                distance_strategy=DistanceStrategy.COSINE
            )
            self.vectorstore.save_local(self.vectorstore_path)

        self.retriever = self.vectorstore.as_retriever()
        self._build_chain()
        print("System ready for queries.")

    def _ingest_documents(self) -> List:
        """Loads PDFs from multiple directory paths."""
        documents = []

        for folder_path in self.folder_paths:
            pdf_folder = Path(folder_path)
            
            if not pdf_folder.exists():
                print(f"Warning: Folder {folder_path} does not exist.")
                continue

            pdf_files = list(pdf_folder.glob("*.pdf"))
            print(f"Found {len(pdf_files)} PDFs in {folder_path}")
            
            for pdf_file in pdf_files:
                try:
                    print(f"  Loading {pdf_file.name}...")
                    loader = PyPDFLoader(file_path=str(pdf_file))
                    docs = loader.load()
                    documents.extend(docs)
                except Exception as e:
                    print(f"  Error loading {pdf_file.name}: {e}")

        print(f"Total documents loaded: {len(documents)}")
        return documents

    def _split_data(self, documents: List) -> List:
        """Chunks documents and adds metadata IDs."""
        text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
        chunks = text_splitter.split_documents(documents=documents)
        
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                "id": f"chunk_{i}",
            })
        
        return chunks
    
    def _build_chain(self):
        """Build the RAG chain for extracting nutrition parameters."""
        system_prompt = """You are a medical nutrition specialist. Extract nutrition parameters for the given medical condition(s).

CONTEXT:
{context}

Based on the provided medical documents, extract and structure the following information:
- Dietary goals and recommendations
- Foods to increase
- Foods to limit or avoid
- Specific nutrition targets (sugar, sodium, fiber, protein, etc.)
- Important dietary notes"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])
        
        question_answer_chain = create_stuff_documents_chain(self.llm, prompt)
        self.rag_chain = create_retrieval_chain(self.retriever, question_answer_chain)
    
    def get_constraints(self, conditions: List[str]) -> Dict[str, Any]:
        """Get nutrition constraints for given medical conditions."""
        if not conditions:
            return self._default_constraints()
        
        if not self.rag_chain:
            print("RAG chain not initialized. Returning default constraints.")
            return self._default_constraints()
        
        query = f"""What are the dietary restrictions and nutrition guidelines for patients with: {', '.join(conditions)}?
        
Please provide:
1. Specific nutrition limits (sugar, sodium, fiber, protein, calories, saturated fat)
2. Foods to increase
3. Foods to limit
4. Foods to completely avoid
5. Important dietary notes"""
        
        try:
            response = self.rag_chain.invoke({"input": query})
            return self._get_condition_constraints(conditions)
        except Exception as e:
            print(f"Error getting constraints: {e}")
            return self._get_condition_constraints(conditions)
    
    def _get_condition_constraints(self, conditions: List[str]) -> Dict[str, Any]:
        """Get constraint recommendations based on medical conditions."""
        constraints = {
            "constraints": {
                "sugar_g": {"max": 25},
                "sodium_mg": {"max": 2300},
                "fiber_g": {"min": 5},
                "saturated_fat_g": {"max": 20}
            },
            "increase": ["vegetables", "whole grains", "lean proteins"],
            "limit": ["processed foods", "added sugars", "salt"],
            "avoid": [],
            "notes": []
        }
        
        conditions_lower = [c.lower() for c in conditions]
        
        if any("diabetes" in c for c in conditions_lower):
            constraints["constraints"]["sugar_g"]["max"] = 10
            constraints["avoid"].append("refined sugars")
            constraints["notes"].append("Monitor blood glucose levels")
        
        if any("hypertension" in c or "blood pressure" in c for c in conditions_lower):
            constraints["constraints"]["sodium_mg"]["max"] = 1500
            constraints["notes"].append("Limit sodium intake")
        
        if any("parkinson" in c for c in conditions_lower):
            constraints["increase"].append("protein")
            constraints["notes"].append("Consider protein timing with medications")
        
        return constraints
    
    def _default_constraints(self) -> Dict[str, Any]:
        """Return default healthy eating constraints."""
        return {
            "constraints": {
                "sugar_g": {"max": 25},
                "sodium_mg": {"max": 2300},
                "fiber_g": {"min": 5},
                "saturated_fat_g": {"max": 20}
            },
            "increase": ["vegetables", "whole grains", "lean proteins"],
            "limit": ["processed foods", "added sugars"],
            "avoid": [],
            "notes": ["General healthy eating guidelines"]
        }
    
    def ask(self, query: str) -> str:
        """Public method to query the RAG system."""
        if not self.rag_chain:
            return "System not initialized. No medical documents loaded."
        response = self.rag_chain.invoke({"input": query})
        return response["answer"]


# Initialize Medical RAG
medical_rag = MedicalRAG(llm=LLM_MODEL,
    folder_paths=PDF_FOLDER,
    vectorstore_path=MEDICAL_VECTORSTORE_PATH
)
medical_rag.initialize(force_rebuild=False)
print("\nMedical RAG ready!")

medical_rag.ask(query="hello")