"""
PathForge FastAPI Application — Groq Edition
FIXES:
1. CORS origins now read from settings.allowed_origins_list — single source of truth.
   No more three separate hardcoded lists in config.py, main.py, and render.yaml.
2. Removed conflicting custom OPTIONS preflight handler.
3. Removed custom add_cors_to_errors HTTP middleware (conflicted with CORSMiddleware).
4. Uses FastAPI exception_handler for clean 500 error handling with CORS intact.
5. Heavy imports (routers, init_db) deferred inside lifespan so a Groq/DB init
   failure doesn't prevent the app object itself from being created.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.config import get_settings
from app.utils.logger import configure_logging, get_logger

settings = get_settings()
configure_logging(debug=settings.DEBUG)
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ───────────────────────────────────────────
    logger.info("pathforge.startup version=%s", settings.APP_VERSION)

    # Lazy import: if DB/Groq init fails, at least the app boots
    from app.dependencies import init_db
    await init_db()

    from app.agent.groq_client import GroqClient
    groq = GroqClient.get()
    logger.info("pathforge.groq_ready available=%s", groq.available)

    yield

    logger.info("pathforge.shutdown")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="AI-Adaptive Onboarding — BERT · Groq LLaMA-3.3 · O*NET · Adaptive BFS",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ─────────────────────────────────────────────────────────────────────
# FIX: Single source of truth — read from settings, which reads ALLOWED_ORIGINS
# env var set in render.yaml. No more three separate hardcoded lists.
ALLOWED_ORIGINS = settings.allowed_origins_list
logger.info("pathforge.cors origins=%s", ALLOWED_ORIGINS)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=600,
)

# ── Routers ───────────────────────────────────────────────────────────────────
from app.api.routes import analyze, chat, health, pathway, quiz  # noqa: E402

app.include_router(health.router)
app.include_router(analyze.router)
app.include_router(chat.router)
app.include_router(pathway.router)
app.include_router(quiz.router)


# ── Global error handler ─────────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error("pathforge.unhandled_error error=%s", str(exc))
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Please try again."},
    )
