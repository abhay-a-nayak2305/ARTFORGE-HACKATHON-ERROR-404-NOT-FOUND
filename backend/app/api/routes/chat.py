"""
POST /api/chat — Groq-powered context-aware chat
"""
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.schemas import ChatRequest, ChatResponse
from app.models.db_models import ChatLog
from app.services.chat_service import ChatService
from app.dependencies import get_db, get_chat_service
from app.utils.logger import get_logger

router = APIRouter(prefix="/api", tags=["chat"])
logger = get_logger(__name__)


@router.post("/chat", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    db: AsyncSession = Depends(get_db),
    chat_service: ChatService = Depends(get_chat_service),
) -> ChatResponse:
    if not req.session_id or not req.message.strip():
        raise HTTPException(status_code=422, detail="session_id and message required")

    response = await chat_service.respond_async(req)

    # Log to DB (best-effort)
    try:
        db.add(ChatLog(
            session_id=req.session_id,
            role="user",
            message=req.message,
            used_groq=response.used_groq if hasattr(response, "used_groq") else False,
        ))
        db.add(ChatLog(
            session_id=req.session_id,
            role="assistant",
            message=response.reply,
            confidence=response.confidence,
            used_groq=response.used_groq if hasattr(response, "used_groq") else False,
        ))
        await db.commit()
    except Exception as e:
        logger.warning("chat.log_failed", error=str(e))

    return response