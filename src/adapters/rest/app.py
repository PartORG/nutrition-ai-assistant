"""
FastAPI application — REST adapter for the Nutrition AI Assistant.

Usage:
    python run_api.py

Or directly:
    uvicorn adapters.rest.app:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure src/ is on sys.path when invoked via uvicorn directly
_src_dir = Path(__file__).resolve().parent.parent.parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from infrastructure.config import Settings
from factory import ServiceFactory
from adapters.rest.dependencies import set_factory
from adapters.rest.routers import auth, recommendations, conversations, profile, images, chat_ws, analytics


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize ServiceFactory on startup."""
    project_root = _src_dir.parent
    config = Settings.from_env(project_root=project_root)
    factory = ServiceFactory(config)
    await factory.initialize()
    set_factory(factory)
    yield
    # No teardown needed — aiosqlite connections are per-operation


app = FastAPI(
    title="Nutrition AI Assistant",
    version="0.2.0",
    description="Personalized meal recommendation API powered by LLM + RAG.",
    lifespan=lifespan,
)

# CORS — permissive for development; tighten allowed_origins in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(auth.router)
app.include_router(recommendations.router)
app.include_router(conversations.router)
app.include_router(profile.router)
app.include_router(images.router)
app.include_router(chat_ws.router)
app.include_router(analytics.router)


@app.get("/health", tags=["health"])
async def health():
    return {"status": "ok", "version": "0.2.0"}
