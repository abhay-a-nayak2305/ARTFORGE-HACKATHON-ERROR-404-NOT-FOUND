"""
FastAPI dependency injection providers.
All heavyweight singletons (DB engine, orchestrator, etc.)
are created once and shared via these dependency functions.
"""
from functools import lru_cache
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

from app.config import get_settings
from app.agent.orchestrator import AgentOrchestrator
from app.services.chat_service import ChatService
from app.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

# ── Async DB engine ───────────────────────────────────────
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
    from app.models.db_models import Base
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("db.tables_initialized")


# ── Singleton service providers ───────────────────────────
@lru_cache(maxsize=1)
def get_orchestrator() -> AgentOrchestrator:
    return AgentOrchestrator()


@lru_cache(maxsize=1)
def get_chat_service() -> ChatService:
    return ChatService()