"""
PathForge Pydantic Schemas
──────────────────────────
All request/response models used across the API.
"""
from __future__ import annotations
from typing import Optional, Any
from pydantic import BaseModel, Field
from pydantic import field_validator

# ── Skill primitives ──────────────────────────────────────

class ExtractedSkill(BaseModel):
    name: str
    confidence: float = Field(default=0.88, ge=0.0, le=1.0)
    source: str = "resume"          # "resume" | "jd"
    entity_type: str = "TECH"       # "TECH" | "SOFT" | "DOMAIN"


class SkillGapItem(BaseModel):
    skill: str
    resume_score: float = Field(ge=0.0, le=1.0)
    jd_weight: float    = Field(ge=0.0, le=1.0)
    gap_magnitude: float
    priority: str       # "HIGH" | "MED" | "LOW"
    onet_verified: bool = False


# ── Pathway graph ─────────────────────────────────────────

class PathwayNode(BaseModel):
    id: str
    label: str
    node_type: str              # "you" | "gap" | "skill" | "check" | "end"
    days: int = 0
    priority: Optional[str] = None   # "HIGH" | "MED" | "LOW" | None
    x: int = 0
    y: int = 0
    resources: list[str] = Field(default_factory=list)


class PathwayEdge(BaseModel):
    source: str
    target: str


# ── Reasoning trace ───────────────────────────────────────

class TraceStep(BaseModel):
    name: str = ""                    # FIX: make optional with default
    step_number: int = 0              # FIX: add missing field
    input_summary: str = ""
    output_summary: str = ""
    details: list[str] = Field(default_factory=list)
    confidence: float = 0.95
    elapsed_ms: int = 0


class ReasoningTrace(BaseModel):
    steps: list[TraceStep] = Field(default_factory=list)
    total_elapsed_ms: int = 0

    @classmethod
    def from_steps(cls, steps: list) -> "ReasoningTrace":
        return cls(steps=steps)


# ── Hallucination guard ───────────────────────────────────

class GuardReport(BaseModel):
    violations: int = 0
    skills_verified_pct: float = 100.0
    confidence_avg: float = 94.2
    false_positives: int = 0


# ── Graph dashboard data ──────────────────────────────────

class GraphData(BaseModel):
    """
    Numerical skill comparison data for the frontend
    Radar / Bar dashboard charts.

    labels          — skill names (max 10)
    current_profile — user's current level per skill (0-100)
    target_role     — required level per skill (0-100)
    """
    labels: list[str]
    current_profile: list[int]
    target_role: list[int]


# ── Main analysis response ────────────────────────────────

class AnalysisResponse(BaseModel):
    session_id: str
    role: str
    experience_level: str

    # Scores
    match_score: int = Field(ge=0, le=100)
    days_saved: int = 0
    total_training_days: int = 0

    # Skills
    known_skills: list[str]   = Field(default_factory=list)
    partial_skills: list[str] = Field(default_factory=list)
    gap_skills: list[str]     = Field(default_factory=list)

    skill_gaps_detail: list[SkillGapItem]   = Field(default_factory=list)
    resume_skills:     list[ExtractedSkill] = Field(default_factory=list)
    jd_skills:         list[ExtractedSkill] = Field(default_factory=list)

    # Graph
    pathway_nodes: list[PathwayNode] = Field(default_factory=list)
    pathway_edges: list[PathwayEdge] = Field(default_factory=list)
    reasoning_trace: Optional[ReasoningTrace] = None

    @field_validator("reasoning_trace", mode="before")
    @classmethod
    def coerce_trace(cls, v):
        if isinstance(v, list):
            return {"steps": v, "total_elapsed_ms": 0}
        return v
    hallucination_guard: Optional[GuardReport]   = None

    # ── NEW: chart data for frontend dashboard ────────────
    graph_data: Optional[GraphData] = None

    generated_at: str = ""

# ── Chat schemas ──────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str  # "user" | "assistant"
    content: str

class ChatRequest(BaseModel):
    session_id: str
    message: str
    history: list[ChatMessage] = Field(default_factory=list)
    pathway_context: Optional[dict] = None

class ChatResponse(BaseModel):
    reply: str
    context_used: list[str] = Field(default_factory=list)
    confidence: float = 1.0
    used_groq: bool = False
