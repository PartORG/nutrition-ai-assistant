from fastapi import FastAPI #, Depends, HTTPException, status
# from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
# from contextlib import asynccontextmanager
from pathlib import Path
import sys
import os
# from datetime import datetime, timedelta
# from typing import Optional
# from jose import JWTError, jwt
# from passlib.context import CryptContext
from api.routes.pipeline import router

# Add src to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

# Add the routes directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'routes')))

# from pipeline.config import PDF_DIR, DATA_DIR, LLM_MODEL, MEDICAL_VECTORSTORE_PATH, RECIPES_NUTRITION_VECTOR_PATH
# from pipeline.intent_retriever import IntentParser
# from pipeline.medical_rag import MedicalRAG
# from pipeline.recipes_nutrition_rag import RecipesNutritionRAG
# from pipeline.safety_filter import SafetyFilter
# from pipeline.test_pipeline import RAGPipeline

# from pipeline.agent import Agent

# Global pipeline instance
pipeline = None

# Global agent instance
# agent = Agent()

app = FastAPI(title="Nutrition AI Assistant")
app.include_router(router)

