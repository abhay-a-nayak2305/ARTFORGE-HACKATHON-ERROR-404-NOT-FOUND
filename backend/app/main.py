"""
PathForge FastAPI Application — Groq Edition
FIXED: CORS middleware order, preflight handling, allowed origins
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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

    await init_db()

    from app.utils.embeddings import EmbeddingService
    EmbeddingService.get()
    logger.info("pathforge.embeddings_warmed")

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

# ── CORS origins ──────────────────────────────────────────────────────────────
# FIX 1: Added all localhost variants (browsers may send either form).
# FIX 2: Removed the manual preflight @app.options route — CORSMiddleware
#         handles OPTIONS correctly on its own. A custom route after the
#         middleware was intercepting responses and causing double-header issues.
# FIX 3: allow_credentials=True requires explicit origins (no "*"), which is
#         already correct here — kept as-is.

ALLOWED_ORIGINS = [
    "https://aionboardingpathforge.netlify.app",
    # Local development — both hostname variants browsers may send
    "http://localhost:3000",
    "http://localhost:5500",
    "http://localhost:8080",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5500",
    "http://127.0.0.1:8080",
]

# IMPORTANT: CORSMiddleware must be added BEFORE any other middleware.
# It will handle OPTIONS preflight automatically — do NOT add a separate
# @app.options route or it will conflict and cause CORS failures.
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=600,  # Cache preflight for 10 minutes
)

# ── Include routers ───────────────────────────────────────────────────────────
app.include_router(health.router)
app.include_router(analyze.router)
app.include_router(chat.router)
app.include_router(pathway.router)
app.include_router(quiz.router)


# ── Global exception handler (keeps CORS headers on 500s) ────────────────────
# FIX 4: Using FastAPI's exception_handler instead of a custom HTTP middleware.
# The old middleware approach ran after CORSMiddleware had already processed the
# response, leading to duplicate or missing CORS headers on error responses.
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error("pathforge.unhandled_error", error=str(exc))
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )
