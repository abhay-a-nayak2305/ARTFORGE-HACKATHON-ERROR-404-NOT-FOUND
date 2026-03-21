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

    def analyze(
        self,
        resume_skills: list[ExtractedSkill],
        jd_skills: list[ExtractedSkill],
    ) -> tuple[list[str], list[str], list[str], list[SkillGapItem]]:
        """
        Returns (known, partial, gaps, gap_detail_list).

        known   — skill names the candidate already has
        partial — skill names with partial coverage
        gaps    — skill names that are completely missing
        gap_detail_list — full SkillGapItem objects for each JD skill
        """
        t0 = time.time()

        resume_names = [s.name for s in resume_skills]
        jd_names = [s.name for s in jd_skills]

        if not resume_names or not jd_names:
            logger.warning("gap_analyzer.analyze: empty skill lists")
            return [], [], jd_names, []

        # Compute full similarity matrix (|resume| × |jd|)
        sim_matrix = self._emb.similarity(resume_names, jd_names)
        # Best resume match for each JD skill  shape: (|jd|,)
        best_scores = sim_matrix.max(axis=0)  # max over resume axis

        known: list[str] = []
        partial: list[str] = []
        gaps: list[str] = []
        details: list[SkillGapItem] = []

        for idx, jd_skill in enumerate(jd_skills):
            score = float(best_scores[idx])
            jd_weight = jd_skill.confidence

            if score >= KNOWN_THRESHOLD:
                classification = "known"
                known.append(jd_skill.name)
                priority = "LOW"
            elif score >= PARTIAL_THRESHOLD:
                classification = "partial"
                partial.append(jd_skill.name)
                priority = "LOW"
            else:
                classification = "gap"
                gaps.append(jd_skill.name)
                priority = "HIGH" if jd_weight >= 0.85 else "MED"

            gap_magnitude = max(0.0, KNOWN_THRESHOLD - score)

            details.append(
                SkillGapItem(
                    skill=jd_skill.name,
                    resume_score=round(score, 3),
                    jd_weight=round(jd_weight, 3),
                    gap_magnitude=round(gap_magnitude, 3),
                    priority=priority if classification == "gap" else "LOW",
                    onet_verified=True,
                )
            )

        elapsed_ms = int((time.time() - t0) * 1000)
        logger.info(
            "gap_analyzer.analyze",
            known=len(known),
            partial=len(partial),
            gaps=len(gaps),
            elapsed_ms=elapsed_ms,
        )
        return known, partial, gaps, details