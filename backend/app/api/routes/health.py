from fastapi import APIRouter
from datetime import datetime

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

# backend/app/api/routes/health.py — add roles endpoint
import json
from pathlib import Path

@router.get("/api/roles")
async def get_roles():
    path = Path("data/role_competencies.json")
    with open(path) as f:
        data = json.load(f)
    return [
        {"name": role, "onet_soc": meta.get("onet_soc", "")}
        for role, meta in data.items()
    ]