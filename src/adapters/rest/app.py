"""
FastAPI application — REST adapter for the Nutrition AI Assistant.

Usage:
    uv run nutriai-api          # console entry point (dev: port 9000, reload)
    uv run python run_api.py    # equivalent

Or directly:
    uv run uvicorn adapters.rest.app:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from adapters.rest.dependencies import set_factory
from adapters.rest.routers import (
    analytics,
    auth,
    chat_ws,
    conversations,
    images,
    profile,
    recommendations,
)
from factory import ServiceFactory
from infrastructure.config import Settings

# Project root = repo root (two levels above this file's src/ package dir).
_src_dir = Path(__file__).resolve().parent.parent.parent


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

# Reject oversized bodies before Starlette parses them. The 10 MB cap in the
# upload handler runs only after request.form() has already spooled the whole
# multipart body to the system temp dir, so it bounds what lands in uploads/
# but not what a 50 GB POST does to the disk.
#
# Chunked requests send no Content-Length and slip past this — the
# authoritative limit belongs at the reverse proxy (nginx:
# `client_max_body_size 10m`).
MAX_BODY_BYTES = 10 * 1024 * 1024 + 8192   # payload + multipart framing


@app.middleware("http")
async def _cap_request_body(request, call_next):
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > MAX_BODY_BYTES:
        return JSONResponse({"detail": "Request too large."}, status_code=413)
    return await call_next(request)


# Register routers
app.include_router(auth.router, prefix="/api")
app.include_router(recommendations.router, prefix="/api")
app.include_router(conversations.router, prefix="/api")
app.include_router(profile.router, prefix="/api")
app.include_router(images.router, prefix="/api")
app.include_router(chat_ws.router, prefix="/api")
app.include_router(analytics.router, prefix="/api")


@app.get("/health", tags=["health"])
async def health():
    return {"status": "ok", "version": "0.2.0"}


def main() -> None:
    """Console entry point (`nutriai-api`): run the dev server with reload."""
    import uvicorn

    uvicorn.run("adapters.rest.app:app", host="0.0.0.0", port=9000, reload=True)
