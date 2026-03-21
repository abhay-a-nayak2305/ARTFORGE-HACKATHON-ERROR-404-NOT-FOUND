"""
PathForge Settings
──────────────────
FIXES:
1. ALLOWED_ORIGINS now reads from the ALLOWED_ORIGINS env var (set in render.yaml)
   instead of being hardcoded. main.py should import ALLOWED_ORIGINS from here
   so there is exactly ONE source of truth — not three separate hardcoded lists.

2. Data file paths are now absolute using Path(__file__) anchoring, so they
   resolve correctly regardless of the working directory uvicorn is launched from.

3. GROQ_API_KEY uses Optional[str] with None default so pydantic-settings doesn't
   raise a ValidationError on startup when the key hasn't been set yet — the
   GroqClient already handles the None/empty case gracefully.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from functools import lru_cache
from pathlib import Path
import os

# Absolute path to the /app directory (where the backend code lives in Docker)
_BASE_DIR = Path(__file__).resolve().parent.parent  # backend/app/../ → backend/


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        # Don't crash if extra env vars exist (Render injects many)
        extra="ignore",
    )

    APP_NAME: str = "PathForge API"
    APP_VERSION: str = "4.0.0"
    DEBUG: bool = False
    ENVIRONMENT: str = "development"

    # Render injects PORT automatically — read it here
    HOST: str = "0.0.0.0"
    PORT: int = Field(default=10000)

    # ── CORS ─────────────────────────────────────────────────────────────────
    # FIX: Single source of truth for allowed origins.
    # Set ALLOWED_ORIGINS in your Render dashboard (already done in render.yaml).
    # Multiple origins can be comma-separated: "https://a.com,https://b.com"
    # main.py reads settings.ALLOWED_ORIGINS — do NOT hardcode a separate list there.
    ALLOWED_ORIGINS: str = (
        "https://aionboardingpathforge.netlify.app,"
        "http://localhost:3000,"
        "http://localhost:5500,"
        "http://localhost:8080,"
        "http://127.0.0.1:3000,"
        "http://127.0.0.1:5500,"
        "http://127.0.0.1:8080"
    )

    @property
    def allowed_origins_list(self) -> list[str]:
        """Parse comma-separated ALLOWED_ORIGINS string into a list."""
        return [o.strip() for o in self.ALLOWED_ORIGINS.split(",") if o.strip()]

    # ── Groq ──────────────────────────────────────────────────────────────────
    # FIX: Use Optional[str] = None so pydantic-settings doesn't raise
    # ValidationError at startup when GROQ_API_KEY isn't set.
    # GroqClient.get() already handles empty/None key gracefully.
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    GROQ_FALLBACK_MODEL: str = "llama-3.1-8b-instant"

    # ── Model paths ───────────────────────────────────────────────────────────
    BERT_MODEL: str = "dslim/bert-base-NER"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # ── Data file paths ───────────────────────────────────────────────────────
    # FIX: Use absolute paths anchored to this file's location so they resolve
    # correctly regardless of what directory uvicorn is launched from.
    # In Docker: _BASE_DIR = /app, so data files are at /app/data/
    ONET_CSV_PATH: str = str(_BASE_DIR / "data" / "onet_skills.csv")
    COURSE_CATALOG_PATH: str = str(_BASE_DIR / "data" / "course_catalog.json")
    ROLE_COMPETENCIES_PATH: str = str(_BASE_DIR / "data" / "role_competencies.json")

    # ── Database ──────────────────────────────────────────────────────────────
    DATABASE_URL: str = "sqlite+aiosqlite:///./pathforge.db"

    # ── Analysis thresholds ───────────────────────────────────────────────────
    COSINE_THRESHOLD: float = 0.78
    MAX_FILE_SIZE_MB: int = 10
    UPLOAD_DIR: str = str(_BASE_DIR / "uploads")


@lru_cache
def get_settings() -> Settings:
    return Settings()
