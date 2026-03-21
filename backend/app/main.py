"""
PathForge FastAPI Application — Groq Edition
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from app.config import get_settings
from app.utils.logger import configure_logging, get_logger
from app.api.routes import analyze, chat, health, pathway, quiz
from app.dependencies import init_db

settings = get_settings()
configure_logging(debug=settings.DEBUG)
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ───────────────────────────────────────────
    logger.info("pathforge.startup", version=settings.APP_VERSION)

    # Init database tables
    await init_db()

    # Pre-warm embedding model
    from app.utils.embeddings import EmbeddingService
    EmbeddingService.get()
    logger.info("pathforge.embeddings_warmed")

    # Init Groq client
    from app.agent.groq_client import GroqClient
    groq = GroqClient.get()
    logger.info("pathforge.groq_ready", available=groq.available)

    yield

    # ── Shutdown ──────────────────────────────────────────
    logger.info("pathforge.shutdown")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="AI-Adaptive Onboarding — BERT · Groq LLaMA-3.3 · O*NET · Adaptive BFS",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Register CORS middleware directly (no circular import)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router)
app.include_router(analyze.router)
app.include_router(chat.router)
app.include_router(pathway.router)
app.include_router(quiz.router)

# Serve frontend
frontend_dir = Path(__file__).resolve().parents[2] / "frontend"
if frontend_dir.exists():
    app.mount(
        "/",
        StaticFiles(directory=str(frontend_dir), html=True),
        name="frontend",
    )
