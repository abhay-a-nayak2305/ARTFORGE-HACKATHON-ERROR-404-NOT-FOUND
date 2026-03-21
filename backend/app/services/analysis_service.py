"""
Analysis Service
────────────────
Thin service layer that sits between the API route and the agent
orchestrator. Handles:
  - DB persistence of each AnalysisSession
  - Async background saving (non-blocking)
  - Session retrieval with DB fallback
"""
import json
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
        """
        Run the full agent pipeline and persist the result.
        Returns the complete AnalysisResponse.
        """
        result = await self._orchestrator.run(
            resume_text=resume_text,
            jd_text=jd_text,
            role=role,
            experience_level=experience_level,
        )

        # Cache in memory for chat
        store_session(result.session_id, result)

        # Persist to DB (non-blocking best-effort)
        try:
            await self._persist(result, used_demo, db)
        except Exception as e:
            logger.warning("analysis_service.persist_failed", error=str(e))

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
            "analysis_service.persisted",
            session_id=result.session_id,
            role=result.role,
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

        result = AnalysisResponse(**row.payload)
        store_session(session_id, result)
        return result