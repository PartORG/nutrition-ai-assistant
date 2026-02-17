from fastapi import FastAPI
from contextlib import asynccontextmanager
from pathlib import Path
import sys

# Add src to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

# from pipeline.config import PDF_DIR, DATA_DIR, LLM_MODEL, MEDICAL_VECTORSTORE_PATH, RECIPES_NUTRITION_VECTOR_PATH
# from pipeline.intent_retriever import IntentParser
# from pipeline.medical_rag import MedicalRAG
# from pipeline.recipes_nutrition_rag import RecipesNutritionRAG
# from pipeline.safety_filter import SafetyFilter
# from pipeline.test_pipeline import RAGPipeline

# Global pipeline instance
pipeline = None

# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     # Startup: Initialize pipeline
#     global pipeline
#     print("🚀 Initializing RAG Pipeline...")
#     intent_parser = IntentParser(model_name=LLM_MODEL)
#     medical_rag = MedicalRAG(
#         folder_paths=[str(PDF_DIR)],
#         model_name=LLM_MODEL,
#         vectorstore_path=str(MEDICAL_VECTORSTORE_PATH),
#         embedding_model="sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
#     )
#     medical_rag.initialize(force_rebuild=False)
    
#     nutrition_rag = RecipesNutritionRAG(
#         data_folder=str(DATA_DIR),
#         model_name=LLM_MODEL,
#         vectorstore_path=str(RECIPES_NUTRITION_VECTOR_PATH)
#     )
#     nutrition_rag.initialize()
    
#     safety_filter = SafetyFilter(debug=False)
    
#     pipeline = RAGPipeline(
#         intent_parser=intent_parser,
#         medical_rag=medical_rag,
#         nutrition_rag=nutrition_rag,
#         safety_filter=safety_filter
#     )
#     print("✅ Pipeline ready!")
#     yield
#     # Shutdown
#     print("🛑 Shutting down...")

app = FastAPI(title="Nutrition AI Assistant") #, lifespan=lifespan

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/chat")
async def chat(query: str):
    if not pipeline:
        return {"error": "Pipeline not initialized"}
    result = pipeline.process(query)
    return {
        "intent": str(result.intent),
        "recommendation": result.llm_recommendation,
        "constraints": result.constraints
    }