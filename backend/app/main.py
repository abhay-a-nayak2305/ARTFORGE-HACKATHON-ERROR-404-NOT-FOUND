"""
PathForge FastAPI Application — Groq Edition
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
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

# ── CORS origins ──────────────────────────────────────────
ALLOWED_ORIGINS = [
    "https://aionboardingpathforge.netlify.app",
    "http://localhost:3000",
    "http://127.0.0.1:5500",
]

# Register CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Guarantee CORS headers even on error responses ────────
@app.middleware("http")
async def add_cors_to_errors(request: Request, call_next):
    try:
        response = await call_next(request)
    except Exception as exc:
        logger.error("pathforge.unhandled_error", error=str(exc))
        response = JSONResponse({"detail": str(exc)}, status_code=500)

    origin = request.headers.get("origin")
    if origin in ALLOWED_ORIGINS:
        response.headers["Access-Control-Allow-Origin"] = origin
    else:
        response.headers["Access-Control-Allow-Origin"] = ALLOWED_ORIGINS[0]

    response.headers["Access-Control-Allow-Credentials"] = "true"
    response.headers["Vary"] = "Origin"
    return response


# ── Preflight handler ─────────────────────────────────────
@app.options("/{rest_of_path:path}")
async def preflight_handler(rest_of_path: str):
    return Response(status_code=200)


# ── Include routers ───────────────────────────────────────
app.include_router(health.router)
app.include_router(analyze.router)
app.include_router(chat.router)
app.include_router(pathway.router)
app.include_router(quiz.router)
