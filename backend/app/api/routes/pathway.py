"""
GET /api/pathway/{session_id}
─────────────────────────────
Returns the full stored AnalysisResponse for a given session.
Useful for the frontend to re-fetch pathway data without re-running analysis.
"""
from fastapi import APIRouter, HTTPException
from app.services.chat_service import get_session
from app.utils.logger import get_logger

router = APIRouter(prefix="/api", tags=["pathway"])
logger = get_logger(__name__)


@router.get("/pathway/{session_id}")
async def get_pathway(session_id: str):
    analysis = get_session(session_id)
    if not analysis:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found or expired.",
        )
    logger.info("pathway.fetch", session_id=session_id)
    return analysis.model_dump()