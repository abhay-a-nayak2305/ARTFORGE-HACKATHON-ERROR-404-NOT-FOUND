"""
SQLAlchemy ORM models for persisting analysis sessions.
Uses async SQLite via aiosqlite for zero-infra local dev.
Swap DATABASE_URL to PostgreSQL for production.
"""
from datetime import datetime
from sqlalchemy import (
    Column, String, Integer, Float, Text,
    DateTime, Boolean, JSON,
)
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class AnalysisSession(Base):
    """
    Stores every analysis run.
    The full pathway JSON is stored in `payload` for fast retrieval.
    """
    __tablename__ = "analysis_sessions"

    id = Column(String(36), primary_key=True)          # UUID
    role = Column(String(100), nullable=False)
    experience_level = Column(String(20), nullable=False)
    match_score = Column(Integer, nullable=False)
    total_training_days = Column(Integer, nullable=False)
    days_saved = Column(Integer, nullable=False)
    gap_count = Column(Integer, default=0)
    known_count = Column(Integer, default=0)
    used_demo = Column(Boolean, default=False)
    payload = Column(JSON, nullable=False)             # full AnalysisResponse
    created_at = Column(DateTime, default=datetime.utcnow)


class ChatLog(Base):
    """
    Stores every chat turn for analytics and future fine-tuning.
    """
    __tablename__ = "chat_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(36), nullable=False, index=True)
    role = Column(String(20), nullable=False)          # "user" | "assistant"
    message = Column(Text, nullable=False)
    intent = Column(String(100), nullable=True)        # detected intent
    confidence = Column(Float, default=1.0)
    used_groq = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class SkillFeedback(Base):
    """
    Stores learner feedback on individual modules (1–5 stars).
    Used to improve duration estimates via collaborative filtering.
    """
    __tablename__ = "skill_feedback"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(36), nullable=False, index=True)
    skill_name = Column(String(200), nullable=False)
    node_type = Column(String(20), nullable=False)
    rating = Column(Integer, nullable=False)           # 1–5
    actual_days = Column(Integer, nullable=True)       # self-reported
    comment = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)