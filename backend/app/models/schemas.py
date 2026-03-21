from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class ExperienceLevel(str, Enum):
    beginner = "beginner"
    mid = "mid"
    senior = "senior"


class TargetRole(str, Enum):
    software_engineer = "Software Engineer"
    data_scientist = "Data Scientist"
    product_manager = "Product Manager"


# ── Request ──────────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    role: TargetRole
    experience_level: ExperienceLevel
    use_demo: bool = False


# ── Skill objects ─────────────────────────────────────────
class ExtractedSkill(BaseModel):
    name: str
    confidence: float = Field(ge=0.0, le=1.0)
    source: str  # "resume" | "jd"
    entity_type: str  # "TECH" | "SOFT" | "TOOL" | "CERT"
    onet_code: Optional[str] = None


class SkillGapItem(BaseModel):
    skill: str
    resume_score: float
    jd_weight: float
    gap_magnitude: float
    priority: str  # "HIGH" | "MED" | "LOW"
    onet_verified: bool


# ── Graph / Pathway ───────────────────────────────────────
class PathwayNode(BaseModel):
    id: str
    label: str
    node_type: str   # "gap" | "skill" | "check" | "end" | "you"
    days: int
    priority: Optional[str]
    x: int
    y: int
    onet_code: Optional[str] = None
    description: Optional[str] = None
    resources: list[str] = []


class PathwayEdge(BaseModel):
    source: str
    target: str
    weight: float = 1.0


# ── Reasoning trace ───────────────────────────────────────
class TraceStep(BaseModel):
    step_number: int
    step_name: str
    input_summary: str
    output_summary: str
    confidence: float
    duration_ms: int
    details: list[str]


class HallucinationGuardReport(BaseModel):
    violations: int
    skills_verified_pct: float
    catalog_match_pct: float
    confidence_avg: float
    false_positives: int
    flagged_items: list[str] = []


# ── Full analysis response ────────────────────────────────
class AnalysisResponse(BaseModel):
    session_id: str
    role: str
    experience_level: str
    match_score: int
    days_saved: int
    total_training_days: int
    known_skills: list[str]
    partial_skills: list[str]
    gap_skills: list[str]
    skill_gaps_detail: list[SkillGapItem]
    resume_skills: list[ExtractedSkill]
    jd_skills: list[ExtractedSkill]
    pathway_nodes: list[PathwayNode]
    pathway_edges: list[PathwayEdge]
    reasoning_trace: list[TraceStep]
    hallucination_guard: HallucinationGuardReport
    generated_at: str


# ── Chat ─────────────────────────────────────────────────
class ChatMessage(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    session_id: str
    message: str
    history: list[ChatMessage] = []


class ChatResponse(BaseModel):
    reply: str
    context_used: list[str] = []
    confidence: float = 1.0