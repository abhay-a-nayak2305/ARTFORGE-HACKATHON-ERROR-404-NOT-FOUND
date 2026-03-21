from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from typing import Literal


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # App
    APP_NAME: str = "PathForge API"
    APP_VERSION: str = "4.0.0"
    DEBUG: bool = False
    ENVIRONMENT: Literal["development", "staging", "production"] = "development"

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    ALLOWED_ORIGINS: list[str] = [
        "http://localhost:3000",
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "null",
    ]

    # ── Groq (primary LLM) ────────────────────────────────
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    GROQ_FALLBACK_MODEL: str = "llama-3.1-8b-instant"

    # Models (local NLP — lightweight, no GPU needed)
    BERT_MODEL: str = "dslim/bert-base-NER"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # Data paths
    ONET_CSV_PATH: str = "data/onet_skills.csv"
    COURSE_CATALOG_PATH: str = "data/course_catalog.json"
    ROLE_COMPETENCIES_PATH: str = "data/role_competencies.json"

    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./pathforge.db"

    # Analysis
    COSINE_THRESHOLD: float = 0.78
    MAX_FILE_SIZE_MB: int = 10
    UPLOAD_DIR: str = "uploads"


@lru_cache
def get_settings() -> Settings:
    return Settings()