"""
Resume Parser Tool
──────────────────
1. Runs spaCy NER to detect PERSON, ORG, DATE entities.
2. Runs a BERT-based NER (dslim/bert-base-NER) for fine-grained
   entity classification.
3. Matches tokens against a curated skill taxonomy using regex +
   embedding similarity as a fallback.
4. Returns a list of ExtractedSkill with confidence scores.
"""
import re
import time
import logging
from typing import Optional

import spacy
from transformers import pipeline, TokenClassificationPipeline

from app.models.schemas import ExtractedSkill
from app.utils.embeddings import EmbeddingService
from app.utils.logger import get_logger

logger = get_logger(__name__)

# ── Curated skill taxonomy (expandable via CSV/DB) ────────
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


class ResumeParser:
    """Stateful parser — loads models once, reused across requests."""

    def __init__(self):
        self._nlp: Optional[spacy.Language] = None
        self._bert_ner: Optional[TokenClassificationPipeline] = None
        self._emb = EmbeddingService.get()
        self._skill_embeddings = self._emb.embed(ALL_SKILLS)

    def _load_models(self):
        if self._nlp is None:
            try:
                self._nlp = spacy.load("en_core_web_sm")
                logger.info("spaCy model loaded")
            except OSError:
                logger.warning("spaCy model not found; skipping spaCy NER")
                self._nlp = None

        if self._bert_ner is None:
            try:
                self._bert_ner = pipeline(
                    "ner",
                    model="dslim/bert-base-NER",
                    aggregation_strategy="simple",
                    device=-1,  # CPU; change to 0 for GPU
                )
                logger.info("BERT NER model loaded")
            except Exception as e:
                logger.warning("BERT NER unavailable: %s", e)
                self._bert_ner = None

    def parse(self, text: str, source: str = "resume") -> list[ExtractedSkill]:
        """
        Main entry point. Returns deduplicated ExtractedSkill list.

        Strategy (layered, highest confidence wins):
          Layer 1 — Regex match against curated taxonomy  (conf ~0.95)
          Layer 2 — spaCy NER for context-aware entities  (conf ~0.80)
          Layer 3 — Embedding similarity fallback          (conf ~0.72)
        """
        t0 = time.time()
        self._load_models()
        skills: dict[str, ExtractedSkill] = {}

        # ── Layer 1: Regex ────────────────────────────────
        for match in _SKILL_PATTERN.finditer(text):
            raw = match.group(0)
            canonical = self._canonicalize(raw)
            if canonical and canonical not in skills:
                entity_type = "SOFT" if canonical in SOFT_SKILLS else "TECH"
                skills[canonical] = ExtractedSkill(
                    name=canonical,
                    confidence=0.95,
                    source=source,
                    entity_type=entity_type,
                )

        # ── Layer 2: BERT NER ─────────────────────────────
        if self._bert_ner:
            try:
                ner_results = self._bert_ner(text[:512])  # BERT 512-token limit
                for ent in ner_results:
                    word = ent["word"].strip()
                    if len(word) < 2:
                        continue
                    canonical = self._canonicalize(word)
                    if canonical and canonical not in skills:
                        skills[canonical] = ExtractedSkill(
                            name=canonical,
                            confidence=round(float(ent["score"]), 3),
                            source=source,
                            entity_type="TECH",
                        )
            except Exception as e:
                logger.warning("BERT NER inference failed: %s", e)

        # ── Layer 3: Embedding fallback for long noun phrases ─
        noun_phrases = self._extract_noun_phrases(text)
        for phrase in noun_phrases:
            if phrase in skills:
                continue
            phrase_emb = self._emb.embed([phrase])
            from sklearn.metrics.pairwise import cosine_similarity
            import numpy as np
            sims = cosine_similarity(phrase_emb, self._skill_embeddings)[0]
            best_idx = int(np.argmax(sims))
            best_score = float(sims[best_idx])
            if best_score >= 0.82:
                canonical = ALL_SKILLS[best_idx]
                if canonical not in skills:
                    skills[canonical] = ExtractedSkill(
                        name=canonical,
                        confidence=round(best_score, 3),
                        source=source,
                        entity_type="TECH" if canonical in TECH_SKILLS else "SOFT",
                    )

        elapsed_ms = int((time.time() - t0) * 1000)
        logger.info(
            "resume_parser.parse",
            source=source,
            extracted=len(skills),
            elapsed_ms=elapsed_ms,
        )
        return list(skills.values())

    def _canonicalize(self, raw: str) -> Optional[str]:
        """Map raw string to canonical skill name using case-insensitive lookup."""
        raw_lower = raw.lower()
        for skill in ALL_SKILLS:
            if skill.lower() == raw_lower:
                return skill
        return None

    def _extract_noun_phrases(self, text: str) -> list[str]:
        """Use spaCy to extract noun chunks as candidate skill phrases."""
        if self._nlp is None:
            return []
        doc = self._nlp(text[:10000])  # limit for speed
        phrases = []
        for chunk in doc.noun_chunks:
            clean = chunk.text.strip()
            if 3 <= len(clean) <= 40:
                phrases.append(clean)
        return phrases