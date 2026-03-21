"""
Analysis Service
────────────────
FIX: fetch_from_db now wraps AnalysisResponse(**row.payload) in try/except.
     If the DB has old rows stored before new fields were added to the schema
     (e.g. graph_data), Pydantic v2 raises ValidationError. Now returns None
     gracefully so the caller treats it as a cache miss and reruns analysis.
"""

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.agent.orchestrator import AgentOrchestrator
from app.models.db_models import AnalysisSession
from app.models.schemas import AnalysisResponse
from app.services.chat_service import store_session, get_session
from app.utils.logger import get_logger

logger = get_logger(__name__)


class AnalysisService:

    def __init__(self, orchestrator: AgentOrchestrator):
        self._orchestrator = orchestrator

    async def run_analysis(
        self,
        resume_text: str,
        jd_text: str,
        role: str,
        experience_level: str,
        used_demo: bool,
        db: AsyncSession,
    ) -> AnalysisResponse:
        result = await self._orchestrator.run(
            resume_text=resume_text,
            jd_text=jd_text,
            role=role,
            experience_level=experience_level,
        )

        # Cache in memory for chat
        store_session(result.session_id, result)

        # Persist to DB (best-effort, non-blocking)
        try:
            await self._persist(result, used_demo, db)
        except Exception as e:
            logger.warning("analysis_service.persist_failed error=%s", str(e))

        return result

    async def _persist(
        self,
        result: AnalysisResponse,
        used_demo: bool,
        db: AsyncSession,
    ) -> None:
        record = AnalysisSession(
            id=result.session_id,
            role=result.role,
            experience_level=result.experience_level,
            match_score=result.match_score,
            total_training_days=result.total_training_days,
            days_saved=result.days_saved,
            gap_count=len(result.gap_skills),
            known_count=len(result.known_skills),
            used_demo=used_demo,
            payload=result.model_dump(),
        )
        db.add(record)
        await db.commit()
        logger.info(
            "analysis_service.persisted session_id=%s role=%s",
            result.session_id, result.role,
        )

    async def fetch_from_db(
        self,
        session_id: str,
        db: AsyncSession,
    ) -> AnalysisResponse | None:
        """Check memory cache first, then DB."""
        cached = get_session(session_id)
        if cached:
            return cached

        stmt = select(AnalysisSession).where(AnalysisSession.id == session_id)
        row = (await db.execute(stmt)).scalar_one_or_none()
        if row is None:
            return None

        # FIX: wrap in try/except — old DB rows may be missing new schema fields
        # (e.g. graph_data added later), which causes Pydantic ValidationError.
        # Returning None causes the caller to treat it as a cache miss.
        try:
            result = AnalysisResponse(**row.payload)
            store_session(session_id, result)
            return result
        except Exception as e:
            logger.warning(
                "analysis_service.fetch_schema_mismatch session_id=%s error=%s",
                session_id, str(e),
            )
            return None
