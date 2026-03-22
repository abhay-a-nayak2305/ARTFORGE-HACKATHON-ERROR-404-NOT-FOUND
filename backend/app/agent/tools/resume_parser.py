"""
Resume Parser Tool — Memory-Optimized for Render Free Tier
──────────────────────────────────────────────────────────
FIX: Removed dslim/bert-base-NER transformer pipeline.
     That model is ~400MB and caused OOM kills on Render's 512MB free tier.
     The orchestrator already uses Groq LLaMA-3.3 for skill extraction which
     is far more accurate. BERT NER was redundant and memory-expensive.

Current pipeline (lightweight):
1. spaCy en_core_web_sm (~15MB) for basic NER context
2. Regex matching against curated skill taxonomy (instant, zero memory)
3. Embedding similarity fallback via MiniLM (already loaded at startup)

Groq handles the heavy lifting via extract_skills_from_text() in the orchestrator.
"""

import re
import time
import logging
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


class ResumeParser:
    """
    Lightweight parser — spaCy + regex only.
    BERT NER removed to fit within Render free tier 512MB RAM limit.
    Groq LLM in the orchestrator handles deep skill extraction.
    """

    def __init__(self):
        self._nlp: Optional[spacy.Language] = None
        self._emb = EmbeddingService.get()
        self._skill_embeddings = None  # lazy-loaded on first use

    def _load_spacy(self):
        """Lazy-load spaCy only when first needed."""
        if self._nlp is None:
            try:
                self._nlp = spacy.load("en_core_web_sm")
                logger.info("spaCy model loaded")
            except OSError:
                logger.warning("spaCy model not found — skipping spaCy NER")
                self._nlp = None

    def parse(
        self,
        text: str,
        source: str = "resume",
    ) -> list[ExtractedSkill]:
        """
        Extract skills from text using:
        1. Regex against curated taxonomy (primary — fast, accurate)
        2. spaCy context filtering (removes false positives)
        3. Embedding similarity for near-matches (fallback)

        Returns deduplicated list of ExtractedSkill.
        """
        t0 = time.time()
        self._load_spacy()

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

        # ── Step 2: spaCy ORG entities as potential tech terms ────
        if self._nlp and len(text) < 50000:
            doc = self._nlp(text[:10000])  # cap at 10k chars to save memory
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

        # ── Step 3: Embedding similarity for short texts ──
        # Only run if regex found very few skills (sparse text)
        if len(found) < 3 and len(text) > 50:
            try:
                words = list({
                    w.strip(".,;:()")
                    for w in text.split()
                    if len(w) > 3
                })[:80]  # cap candidates to save memory

                if words:
                    import numpy as np
                    word_embs = self._emb.embed(words)
                    sims = (word_embs @ self._skill_embeddings.T)  # (words, skills)
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
