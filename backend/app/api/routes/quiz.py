"""
Dynamic Quiz Generator — Groq Edition
──────────────────────────────────────
Instead of hardcoded questions, Groq LLaMA-3.3-70B generates
role-specific diagnostic questions on the fly.

Flow:
  1. Frontend requests quiz for a role
  2. Groq generates 8-10 targeted questions grounded to O*NET skills
  3. User answers → scores mapped to skill proficiency levels
  4. Scores converted to synthetic resume text
  5. Full analysis pipeline runs on synthetic resume

Endpoints:
  GET  /api/quiz/{role}?experience_level=mid
  POST /api/quiz/submit
  POST /api/quiz/generate   (force regenerate, bypass cache)
"""

import json
import re
import hashlib
import asyncio
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.agent.groq_client import GroqClient
from app.agent.tools.onet_lookup import load_onet_skills
from app.agent.tools.resume_parser import TECH_SKILLS, SOFT_SKILLS
from app.utils.logger import get_logger

router = APIRouter(prefix="/api", tags=["quiz"])
logger = get_logger(__name__)

# ── In-memory cache (role+level → generated questions) ───
# Prevents regenerating identical quizzes on every request
_quiz_cache: dict[str, dict] = {}

# ── Supported roles (expand as needed) ───────────────────
SUPPORTED_ROLES = [
    "Software Engineer",
    "Data Scientist",
    "Product Manager",
    "DevOps Engineer",
    "UX Designer",
    "Cybersecurity Analyst",
    "Machine Learning Engineer",
    "Backend Developer",
    "Frontend Developer",
    "Data Engineer",
]

SUPPORTED_LEVELS = ["beginner", "mid", "senior"]


# ── Pydantic models ───────────────────────────────────────

class QuizOption(BaseModel):
    label: str = Field(..., description="Human-readable answer text")
    score: float = Field(..., ge=0.0, le=1.0, description="Proficiency score 0-1")
    skill_level: str = Field(..., description="none | basic | intermediate | advanced")


class QuizQuestion(BaseModel):
    id: str
    question: str
    skill: str                          # canonical skill name
    skill_category: str                 # "technical" | "soft" | "domain"
    why_asked: str                      # shown in tooltip / reasoning trace
    options: list[QuizOption]
    weight: float = Field(1.0, ge=0.1, le=2.0)   # JD importance weight


class GeneratedQuiz(BaseModel):
    role: str
    experience_level: str
    questions: list[QuizQuestion]
    generated_at: str
    groq_model: str
    cache_hit: bool = False
    total_questions: int


class QuizAnswer(BaseModel):
    question_id: str
    skill: str
    score: float = Field(..., ge=0.0, le=1.0)
    skill_level: str


class QuizSubmission(BaseModel):
    role: str
    experience_level: str
    answers: list[QuizAnswer]
    time_taken_seconds: Optional[int] = None


class QuizResult(BaseModel):
    synthetic_resume: str
    skill_profile: dict[str, float]     # skill → proficiency score
    strong_skills: list[str]
    weak_skills: list[str]
    estimated_experience: str
    ready_for_analysis: bool
    session_hint: str


# ── Groq prompt builder ───────────────────────────────────

def _build_system_prompt() -> str:
    return """You are an expert L&D assessment designer with 15 years of experience
creating technical skill diagnostics for corporate onboarding programs.

Your job is to generate highly targeted diagnostic quiz questions for a specific
job role and experience level. These questions will be used by an AI onboarding
engine to identify skill gaps without requiring a resume.

STRICT RULES:
1. Generate EXACTLY the number of questions requested — no more, no less.
2. Every question must target a SPECIFIC, VERIFIABLE technical or professional skill.
3. Questions must be answerable honestly by any candidate — no trick questions.
4. Options must follow a clear proficiency gradient: none → basic → intermediate → advanced.
5. Scores MUST be: 0.0 (no knowledge), 0.35 (basic), 0.65 (intermediate), 1.0 (advanced).
6. The "why_asked" field must explain WHY this skill matters for the role in one sentence.
7. Only use skills that are real, verifiable, and relevant to the role.
8. Adapt question difficulty to the experience level provided.
9. Cover a MIX of: core technical skills, tools/frameworks, practices, and soft skills.
10. Return ONLY valid JSON — no markdown, no explanations, no extra text.

OUTPUT FORMAT (strict JSON array):
[
  {
    "id": "q1",
    "question": "How would you describe your experience with [SKILL]?",
    "skill": "CanonicalSkillName",
    "skill_category": "technical",
    "why_asked": "One sentence explaining why this skill matters for the role.",
    "weight": 1.5,
    "options": [
      {"label": "I have no experience with this",        "score": 0.0,  "skill_level": "none"},
      {"label": "I understand the concept but rarely use it", "score": 0.35, "skill_level": "basic"},
      {"label": "I use it regularly in projects",         "score": 0.65, "skill_level": "intermediate"},
      {"label": "I am highly proficient / teach others",  "score": 1.0,  "skill_level": "advanced"}
    ]
  }
]"""


def _build_user_prompt(
    role: str,
    experience_level: str,
    num_questions: int,
    onet_skills: list[str],
    focus_areas: list[str],
) -> str:
    level_context = {
        "beginner": (
            "The candidate is a recent graduate or career switcher (0-2 years experience). "
            "Focus on foundational skills and basic tool familiarity. "
            "Avoid assuming deep professional experience."
        ),
        "mid": (
            "The candidate has 2-5 years of professional experience. "
            "Include both fundamentals and intermediate practices. "
            "Ask about real-world application and project experience."
        ),
        "senior": (
            "The candidate has 5+ years of experience. "
            "Focus on architecture, leadership, advanced tooling, and best practices. "
            "Include questions about mentoring, system design, and trade-offs."
        ),
    }

    onet_hint = ", ".join(onet_skills[:20]) if onet_skills else "Not available"
    focus_hint = ", ".join(focus_areas) if focus_areas else "General skills"

    return f"""Generate {num_questions} diagnostic quiz questions for this profile:

ROLE: {role}
EXPERIENCE LEVEL: {experience_level}
LEVEL CONTEXT: {level_context.get(experience_level, level_context['mid'])}

O*NET VERIFIED SKILLS FOR THIS ROLE (use these as grounding):
{onet_hint}

PRIORITY FOCUS AREAS FOR {role.upper()}:
{focus_hint}

REQUIREMENTS:
- Questions 1-{num_questions // 2}: Core technical skills (weight 1.5-2.0)
- Questions {num_questions // 2 + 1}-{num_questions - 2}: Tools and frameworks (weight 1.0-1.5)
- Last 2 questions: Soft skills and practices relevant to {role} (weight 0.8-1.0)
- Make every question feel natural and respectful — not a test, but a conversation
- Vary the question phrasing (not all "How would you describe your experience with...")

Generate exactly {num_questions} questions now as a JSON array:"""


# ── Role-specific focus areas ─────────────────────────────

ROLE_FOCUS_AREAS: dict[str, list[str]] = {
    "Software Engineer": [
        "Data Structures & Algorithms", "System Design", "REST APIs",
        "Git & Version Control", "Testing & QA", "Docker",
        "Cloud Platforms", "SQL & Databases", "CI/CD", "Code Review",
    ],
    "Data Scientist": [
        "Statistics & Probability", "Machine Learning", "Python & Pandas",
        "Deep Learning", "Feature Engineering", "Model Deployment",
        "SQL", "Data Visualisation", "A/B Testing", "Experiment Design",
    ],
    "Product Manager": [
        "Product Strategy", "Agile & Scrum", "Roadmapping",
        "Stakeholder Management", "OKR Framework", "User Research",
        "Data Analytics", "GTM Strategy", "Prioritisation Frameworks",
        "Competitive Analysis",
    ],
    "DevOps Engineer": [
        "Kubernetes", "Docker", "Terraform", "CI/CD Pipelines",
        "AWS/GCP/Azure", "Monitoring & Alerting", "Linux Administration",
        "Bash Scripting", "Networking", "Security Practices",
    ],
    "UX Designer": [
        "Figma", "User Research Methods", "Wireframing",
        "Usability Testing", "Design Systems", "Prototyping",
        "Accessibility Standards", "Information Architecture",
        "Visual Design", "Stakeholder Presentation",
    ],
    "Cybersecurity Analyst": [
        "Network Security", "SIEM Tools", "Penetration Testing",
        "OWASP Top 10", "Incident Response", "Cryptography",
        "Compliance Frameworks", "Threat Intelligence",
        "SOC Operations", "Vulnerability Assessment",
    ],
    "Machine Learning Engineer": [
        "PyTorch / TensorFlow", "MLOps", "Model Training & Tuning",
        "Feature Engineering", "Distributed Training", "Model Serving",
        "Python", "Experiment Tracking", "Data Pipelines", "Kubernetes",
    ],
    "Backend Developer": [
        "REST API Design", "Database Design", "Authentication & Auth",
        "Caching Strategies", "Message Queues", "Microservices",
        "SQL & NoSQL", "Performance Optimisation", "Docker", "Testing",
    ],
    "Frontend Developer": [
        "React / Vue / Angular", "JavaScript & TypeScript",
        "CSS & Responsive Design", "State Management",
        "Web Performance", "Accessibility", "Testing (Jest/Cypress)",
        "Build Tools", "REST API Integration", "Browser DevTools",
    ],
    "Data Engineer": [
        "Apache Spark", "SQL & Data Modelling", "ETL Pipelines",
        "Airflow / Prefect", "Data Warehousing", "Kafka",
        "Python", "Cloud Data Services", "dbt", "Data Quality",
    ],
}


# ── O*NET skill fetcher for role ──────────────────────────

def _get_onet_skills_for_role(role: str) -> list[str]:
    """
    Pull role-relevant skills from O*NET database.
    Falls back to focus areas if O*NET not loaded.
    """
    try:
        onet_db = load_onet_skills()
        if not onet_db:
            return ROLE_FOCUS_AREAS.get(role, [])

        # Filter O*NET skills relevant to this role using keyword matching
        role_keywords = set(role.lower().split())
        relevant = []
        for skill_name, meta in onet_db.items():
            category = str(meta.get("category", "")).lower()
            if any(kw in category for kw in role_keywords):
                relevant.append(meta.get("title", skill_name))
            if len(relevant) >= 25:
                break

        return relevant if relevant else ROLE_FOCUS_AREAS.get(role, [])
    except Exception as e:
        logger.warning("onet_skills_fetch_failed: %s", e)
        return ROLE_FOCUS_AREAS.get(role, [])


# ── Cache key builder ─────────────────────────────────────

def _cache_key(role: str, experience_level: str) -> str:
    raw = f"{role}::{experience_level}"
    return hashlib.md5(raw.encode()).hexdigest()


# ── Core generator ────────────────────────────────────────

async def _generate_quiz_with_groq(
    role: str,
    experience_level: str,
    num_questions: int = 10,
) -> list[QuizQuestion]:
    """
    Call Groq to generate quiz questions.
    Validates the response and retries once on parse failure.
    """
    groq = GroqClient.get()
    if not groq.available:
        logger.warning("quiz_generator: Groq unavailable, using fallback")
        return _fallback_questions(role, experience_level)

    onet_skills = _get_onet_skills_for_role(role)
    focus_areas = ROLE_FOCUS_AREAS.get(role, onet_skills[:10])

    system_prompt = _build_system_prompt()
    user_prompt = _build_user_prompt(
        role=role,
        experience_level=experience_level,
        num_questions=num_questions,
        onet_skills=onet_skills,
        focus_areas=focus_areas,
    )

    raw_response = ""

    for attempt in range(3):
        try:
            logger.info(
                "quiz_generator.groq_call",
                role=role,
                level=experience_level,
                attempt=attempt + 1,
            )

            raw_response = await groq.call(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=3000,
                temperature=0.4 + (attempt * 0.1),  # slight increase each retry
            )

            questions = _parse_groq_response(raw_response, role)

            if len(questions) >= 5:
                logger.info(
                    "quiz_generator.success",
                    role=role,
                    questions=len(questions),
                    attempt=attempt + 1,
                )
                return questions

            logger.warning(
                "quiz_generator.insufficient_questions",
                count=len(questions),
                attempt=attempt + 1,
            )

        except Exception as e:
            logger.error(
                "quiz_generator.error",
                attempt=attempt + 1,
                error=str(e),
            )
            if attempt < 2:
                await asyncio.sleep(1.5 ** attempt)

    # All retries failed
    logger.error("quiz_generator.all_retries_failed", role=role)
    return _fallback_questions(role, experience_level)


def _parse_groq_response(raw: str, role: str) -> list[QuizQuestion]:
    """
    Robustly parse Groq's JSON response into QuizQuestion objects.
    Handles common LLM output issues:
      - Markdown code fences
      - Extra text before/after JSON
      - Missing fields (filled with defaults)
      - Invalid score values (clamped to 0-1)
    """
    # Strip markdown code fences if present
    cleaned = re.sub(r"```(?:json)?", "", raw).strip()

    # Find the JSON array in the response
    array_match = re.search(r"\[.*\]", cleaned, re.DOTALL)
    if not array_match:
        raise ValueError(f"No JSON array found in Groq response: {cleaned[:200]}")

    raw_list = json.loads(array_match.group(0))
    questions: list[QuizQuestion] = []

    for i, item in enumerate(raw_list):
        try:
            # Validate and clean options
            options = []
            for opt in item.get("options", []):
                score = float(opt.get("score", 0.0))
                score = max(0.0, min(1.0, score))  # clamp to [0, 1]
                options.append(QuizOption(
                    label=str(opt.get("label", "No label")),
                    score=round(score, 2),
                    skill_level=str(opt.get("skill_level", "basic")),
                ))

            # Ensure exactly 4 options
            if len(options) < 4:
                options = _default_options()
            elif len(options) > 4:
                options = options[:4]

            # Ensure scores are in ascending order (none < basic < intermediate < advanced)
            options.sort(key=lambda o: o.score)

            weight = float(item.get("weight", 1.0))
            weight = max(0.1, min(2.0, weight))  # clamp weight

            questions.append(QuizQuestion(
                id=item.get("id", f"q{i+1}"),
                question=str(item.get("question", f"How proficient are you in skill {i+1}?")),
                skill=str(item.get("skill", f"Skill {i+1}")),
                skill_category=str(item.get("skill_category", "technical")),
                why_asked=str(item.get("why_asked", f"Core requirement for {role}.")),
                options=options,
                weight=round(weight, 1),
            ))

        except Exception as e:
            logger.warning("quiz_parser.skip_question index=%d error=%s", i, e)
            continue

    return questions


def _default_options() -> list[QuizOption]:
    """Standard 4-option proficiency scale used as fallback."""
    return [
        QuizOption(label="I have no experience with this",           score=0.0,  skill_level="none"),
        QuizOption(label="I understand it but rarely use it",        score=0.35, skill_level="basic"),
        QuizOption(label="I use it regularly in real projects",      score=0.65, skill_level="intermediate"),
        QuizOption(label="I am highly proficient / mentor others",   score=1.0,  skill_level="advanced"),
    ]


def _fallback_questions(role: str, experience_level: str) -> list[QuizQuestion]:
    """
    Static fallback questions used when Groq is unavailable.
    Covers the most critical skills for each role.
    """
    focus = ROLE_FOCUS_AREAS.get(role, ["Core Skills", "Communication", "Problem Solving"])
    questions = []

    for i, skill in enumerate(focus[:8]):
        questions.append(QuizQuestion(
            id=f"q{i+1}",
            question=f"How would you rate your proficiency with {skill}?",
            skill=skill,
            skill_category="technical" if i < 6 else "soft",
            why_asked=f"{skill} is a core requirement for the {role} role.",
            options=_default_options(),
            weight=1.5 if i < 3 else 1.0,
        ))

    return questions


# ── Score → resume text converter ────────────────────────

def _scores_to_resume(
    answers: list[QuizAnswer],
    role: str,
    experience_level: str,
) -> tuple[str, dict[str, float]]:
    """
    Convert quiz answers into a synthetic resume text that the
    full analysis pipeline can process exactly like a real resume.

    Also returns a skill_profile dict: skill_name → proficiency_score.
    """
    skill_profile: dict[str, float] = {}

    advanced_skills: list[str]      = []
    intermediate_skills: list[str]  = []
    basic_skills: list[str]         = []
    no_skills: list[str]            = []

    for answer in answers:
        skill_profile[answer.skill] = answer.score
        if answer.score >= 0.85:
            advanced_skills.append(answer.skill)
        elif answer.score >= 0.55:
            intermediate_skills.append(answer.skill)
        elif answer.score >= 0.25:
            basic_skills.append(answer.skill)
        else:
            no_skills.append(answer.skill)

    level_label = {
        "beginner": "Entry-Level",
        "mid": "Mid-Level",
        "senior": "Senior",
    }.get(experience_level, "Professional")

    years = {
        "beginner": "1 year",
        "mid": "3 years",
        "senior": "7 years",
    }.get(experience_level, "3 years")

    lines: list[str] = []
    lines.append(f"DIAGNOSTIC ASSESSMENT PROFILE")
    lines.append(f"Role Target: {role}")
    lines.append(f"Experience Level: {level_label} ({years} estimated)")
    lines.append(f"Assessment Date: {datetime.utcnow().strftime('%B %Y')}")
    lines.append("")

    if advanced_skills:
        lines.append("EXPERT PROFICIENCY (actively use in production):")
        for s in advanced_skills:
            lines.append(f"  - {s}: Advanced — used regularly in professional projects")

    if intermediate_skills:
        lines.append("")
        lines.append("WORKING PROFICIENCY (comfortable, some project experience):")
        for s in intermediate_skills:
            lines.append(f"  - {s}: Intermediate — applied in real-world scenarios")

    if basic_skills:
        lines.append("")
        lines.append("FOUNDATIONAL KNOWLEDGE (familiar, limited hands-on experience):")
        for s in basic_skills:
            lines.append(f"  - {s}: Basic — conceptual understanding, limited practical use")

    lines.append("")
    lines.append(f"EXPERIENCE SUMMARY:")
    lines.append(
        f"  {level_label} {role} with approximately {years} of experience. "
        f"Strong in: {', '.join(advanced_skills[:3]) if advanced_skills else 'foundational skills'}."
    )

    if no_skills:
        lines.append("")
        lines.append("AREAS FOR DEVELOPMENT (no current experience):")
        for s in no_skills:
            lines.append(f"  - {s}: Not yet explored")

    return "\n".join(lines), skill_profile


# ═══════════════════════════════════════════════════════════
#  API ENDPOINTS
# ═══════════════════════════════════════════════════════════

@router.get("/quiz/{role}", response_model=GeneratedQuiz)
async def get_quiz(
    role: str,
    experience_level: str = Query(default="mid", enum=SUPPORTED_LEVELS),
    num_questions: int = Query(default=10, ge=5, le=15),
    force_regenerate: bool = Query(default=False),
):
    """
    Get a dynamically generated quiz for a role and experience level.

    - Cached for 1 hour per role+level combination
    - force_regenerate=true bypasses the cache
    - num_questions: 5 to 15 (default 10)
    """
    if role not in SUPPORTED_ROLES:
        raise HTTPException(
            status_code=400,
            detail=f"Role '{role}' not supported. Choose from: {SUPPORTED_ROLES}",
        )

    cache_key = _cache_key(role, experience_level)
    cache_hit = False

    # Check cache first
    if not force_regenerate and cache_key in _quiz_cache:
        cached = _quiz_cache[cache_key]
        logger.info("quiz.cache_hit", role=role, level=experience_level)
        return GeneratedQuiz(
            role=cached["role"],
            experience_level=cached["experience_level"],
            questions=cached["questions"],
            generated_at=cached["generated_at"],
            groq_model=cached["groq_model"],
            cache_hit=True,
            total_questions=len(cached["questions"]),
        )

    # Generate fresh quiz with Groq
    questions = await _generate_quiz_with_groq(
        role=role,
        experience_level=experience_level,
        num_questions=num_questions,
    )

    now = datetime.utcnow().isoformat() + "Z"
    groq_model = "llama-3.3-70b-versatile" if GroqClient.get().available else "fallback"

    # Store in cache
    _quiz_cache[cache_key] = {
        "role": role,
        "experience_level": experience_level,
        "questions": questions,
        "generated_at": now,
        "groq_model": groq_model,
    }

    logger.info(
        "quiz.generated",
        role=role,
        level=experience_level,
        questions=len(questions),
        groq=groq_model,
    )

    return GeneratedQuiz(
        role=role,
        experience_level=experience_level,
        questions=questions,
        generated_at=now,
        groq_model=groq_model,
        cache_hit=False,
        total_questions=len(questions),
    )


@router.post("/quiz/submit", response_model=QuizResult)
async def submit_quiz(submission: QuizSubmission):
    """
    Process quiz answers and return:
      - synthetic resume text (fed directly to the analysis pipeline)
      - skill profile breakdown
      - strong/weak skills
      - readiness assessment
    """
    if not submission.answers:
        raise HTTPException(status_code=422, detail="No answers provided.")

    if submission.role not in SUPPORTED_ROLES:
        raise HTTPException(status_code=400, detail=f"Unsupported role: {submission.role}")

    synthetic_resume, skill_profile = _scores_to_resume(
        answers=submission.answers,
        role=submission.role,
        experience_level=submission.experience_level,
    )

    # Classify skills
    strong = [s for s, score in skill_profile.items() if score >= 0.65]
    weak   = [s for s, score in skill_profile.items() if score < 0.35]

    # Estimate experience label
    avg_score = (
        sum(skill_profile.values()) / len(skill_profile)
        if skill_profile else 0.0
    )
    if avg_score >= 0.75:
        estimated_experience = "Senior (5+ years)"
    elif avg_score >= 0.50:
        estimated_experience = "Mid-Level (2-5 years)"
    elif avg_score >= 0.25:
        estimated_experience = "Junior (0-2 years)"
    else:
        estimated_experience = "Beginner (career switcher)"

    logger.info(
        "quiz.submitted",
        role=submission.role,
        level=submission.experience_level,
        strong_count=len(strong),
        weak_count=len(weak),
        avg_score=round(avg_score, 2),
    )

    return QuizResult(
        synthetic_resume=synthetic_resume,
        skill_profile=skill_profile,
        strong_skills=strong,
        weak_skills=weak,
        estimated_experience=estimated_experience,
        ready_for_analysis=len(submission.answers) >= 5,
        session_hint=(
            f"Quiz complete. {len(strong)} strong skills detected. "
            f"Ready to generate your {submission.role} pathway."
        ),
    )


@router.post("/quiz/generate", response_model=GeneratedQuiz)
async def force_generate_quiz(
    role: str,
    experience_level: str = Query(default="mid", enum=SUPPORTED_LEVELS),
    num_questions: int = Query(default=10, ge=5, le=15),
):
    """
    Force-regenerate a quiz bypassing the cache.
    Useful for testing or getting fresh questions.
    """
    return await get_quiz(
        role=role,
        experience_level=experience_level,
        num_questions=num_questions,
        force_regenerate=True,
    )


@router.get("/quiz/roles/list")
async def list_supported_roles():
    """Return all roles that support quiz mode."""
    return {
        "roles": SUPPORTED_ROLES,
        "levels": SUPPORTED_LEVELS,
        "groq_available": GroqClient.get().available,
    }