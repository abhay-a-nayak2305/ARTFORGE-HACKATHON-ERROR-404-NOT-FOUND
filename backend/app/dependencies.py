"""
FastAPI dependency injection providers.
FIXES:
1. AgentOrchestrator and ChatService are now lazy-imported inside their
   provider functions, not at module top-level. This means if groq_client.py
   or any service fails to initialise, only those specific endpoints fail —
   not the entire app. Without this fix, a Groq key error would crash the
   import of dependencies.py, which crashes main.py, which makes Render
   think the whole deployment failed (health check never responds).

2. get_orchestrator and get_chat_service use functools.lru_cache correctly —
   they still return singletons, just initialised on first request rather
   than at import time.
"""

from functools import lru_cache
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

# ── Async DB engine ───────────────────────────────────────────────────────────
# DB engine is safe to create at import time — it doesn't make network calls
# until the first actual query.
_engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
    future=True,
)

_AsyncSessionLocal = async_sessionmaker(
    bind=_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_db() -> AsyncSession:
    """Yield an async DB session; auto-close on exit."""
    async with _AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db() -> None:
    """Create all tables on startup if they don't exist."""
    # FIX: Import Base lazily so a model import error only affects DB init,
    # not the entire application startup.
    try:
        from app.models.db_models import Base
        async with _engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("db.tables_initialized")
    except Exception as e:
        logger.error("db.init_failed error=%s — continuing without DB", e)


# ── Singleton service providers ───────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_orchestrator():
    """
    FIX: Lazy import — AgentOrchestrator is only imported when first requested,
    not at module load time. This prevents a Groq init failure from crashing
    the entire dependencies module and taking down the health endpoint.
    """
    from app.agent.orchestrator import AgentOrchestrator
    return AgentOrchestrator()


@lru_cache(maxsize=1)
def get_chat_service():
    """
    FIX: Same lazy import pattern as get_orchestrator.
    """
    from app.services.chat_service import ChatService
    return ChatService()
