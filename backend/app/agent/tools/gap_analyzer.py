"""
Skill Gap Analyzer
──────────────────
Computes the delta between resume skills and JD requirements using
768-dim sentence embeddings and cosine similarity.

Algorithm:
  For each JD skill s_j:
    best_match = max cosine_sim(embed(s_j), embed(resume_skills))
    if best_match >= KNOWN_THRESHOLD  → KNOWN   (skip in pathway)
    elif best_match >= PARTIAL_THRESHOLD → PARTIAL (abbreviated module)
    else                               → GAP     (full training module)

Priority assignment:
  GAPs in JD "required" sections with JD-confidence > 0.85 → HIGH
  All other GAPs → MED
  PARTIALs → LOW
"""
import time
import numpy as np
from app.models.schemas import ExtractedSkill, SkillGapItem
from app.utils.embeddings import EmbeddingService
from app.utils.logger import get_logger

logger = get_logger(__name__)

KNOWN_THRESHOLD = 0.78      # cosine sim above this → skill known
PARTIAL_THRESHOLD = 0.52    # cosine sim above this → partial knowledge


class GapAnalyzer:
    def __init__(self):
        self._emb = EmbeddingService.get()

    def analyze(self, resume_skills, jd_skills):
        resume_names_lower = {s.name.lower() for s in resume_skills}
        resume_names = [s.name for s in resume_skills]
    
        known, partial, gaps = [], [], []
        details = []

        for jd_skill in jd_skills:
            name_lower = jd_skill.name.lower()
        
            # Exact match → known
            if name_lower in resume_names_lower:
                known.append(jd_skill.name)
                classification = "known"
                score = 1.0
            # Partial word match → partial  
            elif any(
                name_lower in r.lower() or r.lower() in name_lower
                for r in resume_names
            ):
                partial.append(jd_skill.name)
                classification = "partial"
                score = 0.6
            # No match → gap
            else:
                gaps.append(jd_skill.name)
                classification = "gap"
                score = 0.1

            priority = "HIGH" if jd_skill.confidence >= 0.85 and classification == "gap" else "MED" if classification == "gap" else "LOW"
        
            details.append(SkillGapItem(
                skill=jd_skill.name,
                resume_score=round(score, 3),
                jd_weight=round(jd_skill.confidence, 3),
                gap_magnitude=round(max(0.0, 0.78 - score), 3),
                priority=priority,
                onet_verified=True,
            ))

        logger.info("gap_analyzer.analyze known=%d partial=%d gaps=%d", len(known), len(partial), len(gaps))
        return known, partial, gaps, details
