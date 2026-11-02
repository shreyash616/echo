"""
Echo FastAPI backend.

Run with:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

import time
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.services.recommender import recommender

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("Starting Echo backend…")
    recommender.load()  # Load CLAP + FAISS index once at startup
    yield
    logger.info("Shutting down Echo backend.")


app = FastAPI(
    title="Echo API",
    version="1.0.0",
    lifespan=lifespan,
)

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time = time.perf_counter() - start_time
    response.headers["X-Process-Time"] = f"{process_time:.6f}"
    return response

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

# The Next.js web app (web/), as a static export the Deploy workflow builds and copies here; not in git
WEB_DIR = Path(__file__).parent / "web"
if WEB_DIR.exists():
    # Last, so /api matches first; html=True serves song/index.html for /song/ and 404.html
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
