# backend/app/services/session_store.py

import json
import redis.asyncio as redis
from app.config import get_settings
from app.models.schemas import AnalysisResponse

settings = get_settings()

_redis: redis.Redis | None = None
_local_cache: dict = {}   # fallback if Redis unavailable

SESSION_TTL = 60 * 60 * 24  # 24 hours


async def get_redis() -> redis.Redis | None:
    global _redis
    if _redis is None and settings.REDIS_URL:
        try:
            _redis = redis.from_url(settings.REDIS_URL, decode_responses=True)
            await _redis.ping()
        except Exception:
            _redis = None
    return _redis


async def store_session(session_id: str, analysis: AnalysisResponse) -> None:
    r = await get_redis()
    if r:
        await r.setex(
            f"session:{session_id}",
            SESSION_TTL,
            analysis.model_dump_json(),
        )
    else:
        _local_cache[session_id] = analysis


async def get_session_async(session_id: str) -> AnalysisResponse | None:
    r = await get_redis()
    if r:
        raw = await r.get(f"session:{session_id}")
        if raw:
            return AnalysisResponse.model_validate_json(raw)
        return None
    return _local_cache.get(session_id)