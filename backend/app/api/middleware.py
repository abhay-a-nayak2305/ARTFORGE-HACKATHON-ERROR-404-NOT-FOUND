"""
CORS + request logging middleware.
"""
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import time
from app.utils.logger import get_logger
# backend/app/api/middleware.py  — add rate limiting

from collections import defaultdict
from time import time
from fastapi import Request, HTTPException

# Simple in-memory rate limiter
# Use Redis token bucket in production
_request_counts: dict[str, list[float]] = defaultdict(list)
RATE_LIMIT = 10        # max requests
RATE_WINDOW = 60       # per 60 seconds


def check_rate_limit(ip: str) -> None:
    now = time()
    # Remove old timestamps outside window
    _request_counts[ip] = [
        t for t in _request_counts[ip]
        if now - t < RATE_WINDOW
    ]
    if len(_request_counts[ip]) >= RATE_LIMIT:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Max {RATE_LIMIT} requests per {RATE_WINDOW}s.",
        )
    _request_counts[ip].append(now)


# Add this to the HTTP middleware in middleware.py
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    if request.url.path.startswith("/api/analyze"):
        ip = request.client.host or "unknown"
        check_rate_limit(ip)
    return await call_next(request)

logger = get_logger(__name__)


def register_middleware(app: FastAPI, allowed_origins: list[str]) -> None:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        t0 = time.time()
        response = await call_next(request)
        elapsed = round((time.time() - t0) * 1000)
        logger.info(
            "http.request",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            elapsed_ms=elapsed,
        )
        return response