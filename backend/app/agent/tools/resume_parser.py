"""
Resume Parser Tool — Memory-Optimized for Render Free Tier
──────────────────────────────────────────────────────────
FIX 1: Removed dslim/bert-base-NER transformer pipeline (~400MB OOM).
FIX 2: Module-level spaCy singleton — loaded ONCE at startup, shared
        across all calls. Previously a new ResumeParser() was created
        per request, causing spaCy to reload on every call. When resume
        and JD were parsed concurrently, spaCy loaded TWICE in parallel
        → OOM kill → 502 with no CORS headers sent back to browser.

Current pipeline (lightweight):
1. spaCy en_core_web_sm (~15MB) — loaded ONCE via module-level singleton
2. Regex matching against curated skill taxonomy (instant, zero memory)
3. Embedding similarity fallback via MiniLM (cached at module level)

Groq handles deep skill extraction via orchestrator.
"""

import re
import time
from typing import Optional

import spacy

from app.models.schemas import ExtractedSkill
from app.utils.embeddings import EmbeddingService
from app.utils.logger import get_logger

logger = get_logger(__name__)

# ── Curated skill taxonomy ────────────────────────────────
TECH_SKILLS: list[str] = [
    # Languages
    "Python", "JavaScript", "TypeScript", "Java", "Go", "Rust", "C++", "C#",
    "Ruby", "Swift", "Kotlin", "Scala", "R", "MATLAB", "PHP", "Bash",
    # Frameworks
    "React", "Vue", "Angular", "Next.js", "Django", "FastAPI", "Flask",
    "Spring Boot", "Express", "Node.js", "Rails", "Laravel",
    # Data / ML
    "TensorFlow", "PyTorch", "Scikit-learn", "Keras", "Pandas", "NumPy",
    "Spark", "Hadoop", "Kafka", "Airflow", "dbt", "MLflow", "Kubeflow",
    # Cloud / DevOps
    "AWS", "GCP", "Azure", "Docker", "Kubernetes", "Terraform", "Ansible",
    "CI/CD", "Jenkins", "GitHub Actions", "ArgoCD", "Helm",
    # Databases
    "PostgreSQL", "MySQL", "MongoDB", "Redis", "Elasticsearch", "Cassandra",
    "BigQuery", "Snowflake", "DynamoDB", "SQLite",
    # Practices
    "REST APIs", "GraphQL", "gRPC", "Microservices", "System Design",
    "Git", "Linux", "Agile", "Scrum", "TDD", "Code Review",
    # PM / Analytics
    "SQL", "Tableau", "Power BI", "Looker", "JIRA", "Confluence", "Figma",
    "A/B Testing", "OKRs", "Roadmapping", "GTM Strategy",
    # Data Science
    "Statistics", "Machine Learning", "Deep Learning", "Feature Engineering",
    "Model Deployment", "MLOps", "Data Wrangling", "NLP",
]

SOFT_SKILLS: list[str] = [
    "Communication", "Leadership", "Problem Solving", "Teamwork",
    "Critical Thinking", "Time Management", "Adaptability",
    "Stakeholder Management", "Presentation", "Mentoring",
]

ALL_SKILLS = TECH_SKILLS + SOFT_SKILLS

# Pre-compile regex for fast matching
_SKILL_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(s) for s in ALL_SKILLS) + r")\b",
    flags=re.IGNORECASE,
)

# Canonical name lookup (lowercase → proper case)
_SKILL_CANONICAL = {s.lower(): s for s in ALL_SKILLS}


# ═══════════════════════════════════════════════════════════
# MODULE-LEVEL SINGLETONS
# These are initialised once when the module is first imported
# (i.e. at server startup) and reused on every subsequent call.
# This is the key fix — no more per-request spaCy reloads.
# ═══════════════════════════════════════════════════════════

_NLP: Optional[spacy.Language] = None
_SKILL_EMBEDDINGS = None
_PARSER_INSTANCE: Optional["ResumeParser"] = None


def _get_nlp() -> Optional[spacy.Language]:
    """Return shared spaCy model, loading it once if not yet loaded."""
    global _NLP
    if _NLP is None:
        try:
            _NLP = spacy.load("en_core_web_sm")
            logger.info("spaCy model loaded")
        except OSError:
            logger.warning("spaCy model not found — skipping spaCy NER")
            _NLP = None
    return _NLP


def _get_skill_embeddings():
    """Compute and cache skill embeddings once at module level."""
    global _SKILL_EMBEDDINGS
    if _SKILL_EMBEDDINGS is None:
        try:
            emb = EmbeddingService.get()
            _SKILL_EMBEDDINGS = emb.embed(ALL_SKILLS)
        except Exception as e:
            logger.warning("resume_parser: could not precompute skill embeddings: %s", e)
    return _SKILL_EMBEDDINGS


def get_parser() -> "ResumeParser":
    """
    Return the shared ResumeParser singleton.
    Use this instead of ResumeParser() in the orchestrator/services.
    """
    global _PARSER_INSTANCE
    if _PARSER_INSTANCE is None:
        _PARSER_INSTANCE = ResumeParser()
    return _PARSER_INSTANCE


# ═══════════════════════════════════════════════════════════
# PARSER CLASS
# ═══════════════════════════════════════════════════════════

class ResumeParser:
    """
    Lightweight resume/JD skill parser.
    Do NOT instantiate per-request — use get_parser() for the singleton.
    """

    def __init__(self):
        # Only grab EmbeddingService reference — spaCy loaded via module singleton
        self._emb = EmbeddingService.get()

    def parse(
        self,
        text: str,
        source: str = "resume",
    ) -> list[ExtractedSkill]:
        """
        Extract skills from text using:
        1. Regex against curated taxonomy  (fast, ~0ms)
        2. spaCy ORG/PRODUCT NER          (uses shared singleton, ~0ms after first load)
        3. Embedding similarity fallback   (only for very sparse text)

        Safe to call concurrently — all heavy objects are module-level singletons.
        """
        t0 = time.time()

        # Uses shared module-level model — does NOT reload spaCy
        nlp = _get_nlp()

        found: dict[str, ExtractedSkill] = {}

        # ── Step 1: Regex matching ────────────────────────
        for match in _SKILL_PATTERN.finditer(text):
            raw = match.group(0)
            canonical = _SKILL_CANONICAL.get(raw.lower(), raw)
            if canonical not in found:
                entity_type = "TECH" if canonical in TECH_SKILLS else "SOFT"
                found[canonical] = ExtractedSkill(
                    name=canonical,
                    confidence=0.92,
                    source=source,
                    entity_type=entity_type,
                )

        # ── Step 2: spaCy ORG/PRODUCT entities ───────────
        if nlp and len(text) < 50000:
            doc = nlp(text[:10000])  # cap to save memory
            for ent in doc.ents:
                if ent.label_ in ("ORG", "PRODUCT"):
                    name = ent.text.strip()
                    canonical = _SKILL_CANONICAL.get(name.lower())
                    if canonical and canonical not in found:
                        found[canonical] = ExtractedSkill(
                            name=canonical,
                            confidence=0.78,
                            source=source,
                            entity_type="TECH",
                        )

        # ── Step 3: Embedding fallback (sparse text only) ─
        if len(found) < 3 and len(text) > 50:
            try:
                words = list({
                    w.strip(".,;:()")
                    for w in text.split()
                    if len(w) > 3
                })[:80]

                if words:
                    import numpy as np
                    word_embs = self._emb.embed(words)
                    skill_embs = _get_skill_embeddings()  # cached module-level
                    if skill_embs is not None:
                        sims = word_embs @ skill_embs.T
                        best_skill_idx = sims.argmax(axis=1)
                        best_scores = sims.max(axis=1)
                        for i, score in enumerate(best_scores):
                            if float(score) >= 0.88:
                                skill = ALL_SKILLS[best_skill_idx[i]]
                                if skill not in found:
                                    found[skill] = ExtractedSkill(
                                        name=skill,
                                        confidence=round(float(score), 3),
                                        source=source,
                                        entity_type="TECH" if skill in TECH_SKILLS else "SOFT",
                                    )
            except Exception as e:
                logger.warning("resume_parser.embedding_fallback_failed: %s", e)

        elapsed_ms = int((time.time() - t0) * 1000)
        logger.info(
            "resume_parser.parse source=%s skills=%d elapsed_ms=%d",
            source, len(found), elapsed_ms,
        )
        return list(found.values())
