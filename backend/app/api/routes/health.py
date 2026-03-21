"""
Health & utility routes.

FIXES:
1. /api/roles used hardcoded relative Path("data/role_competencies.json")
   which fails on Render. Now uses settings.ROLE_COMPETENCIES_PATH (absolute).
2. Route registered as @router.get("/api/roles") but router has no prefix,
   so it mounted at /api/roles correctly — but confusing. Kept as-is since
   the frontend calls ${API_BASE}/roles which already includes /api.
3. Added error handling on file read so a missing file returns 503 instead
   of crashing the whole process with an unhandled FileNotFoundError.
"""

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from datetime import datetime
import json

router = APIRouter(tags=["health"])


@router.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "PathForge API",
        "version": "4.0.0",
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


@router.get("/")
async def root():
    return {
        "message": "PathForge AI Adaptive Onboarding Engine",
        "docs": "/docs",
        "health": "/health",
    }


# FIX 1: Use settings.ROLE_COMPETENCIES_PATH (absolute, set in config.py)
# instead of hardcoded relative Path("data/role_competencies.json")
@router.get("/api/roles")
async def get_roles():
    from app.config import get_settings
    settings = get_settings()
    try:
        with open(settings.ROLE_COMPETENCIES_PATH) as f:
            data = json.load(f)
        return [
            {"name": role, "onet_soc": meta.get("onet_soc", "")}
            for role, meta in data.items()
        ]
    except FileNotFoundError:
        return JSONResponse(
            status_code=503,
            content={"detail": "Role competencies data not found. Check data files."},
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"detail": f"Failed to load roles: {str(e)}"},
        )
