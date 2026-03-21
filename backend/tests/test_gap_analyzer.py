"""
Focused unit tests for the GapAnalyzer cosine similarity logic.
"""
import pytest
from app.agent.tools.gap_analyzer import GapAnalyzer, KNOWN_THRESHOLD, PARTIAL_THRESHOLD
from app.models.schemas import ExtractedSkill


def skill(name: str, source: str = "resume") -> ExtractedSkill:
    return ExtractedSkill(name=name, confidence=0.9, source=source, entity_type="TECH")


def test_exact_match_is_known():
    analyzer = GapAnalyzer()
    known, partial, gaps, _ = analyzer.analyze(
        [skill("Python")],
        [skill("Python", "jd")],
    )
    assert "Python" in known
    assert "Python" not in gaps


def test_completely_missing_skill_is_gap():
    analyzer = GapAnalyzer()
    known, partial, gaps, _ = analyzer.analyze(
        [skill("Python"), skill("SQL")],
        [skill("Kubernetes", "jd")],
    )
    assert "Kubernetes" in gaps


def test_priority_assignment():
    analyzer = GapAnalyzer()
    _, _, _, details = analyzer.analyze(
        [skill("Python")],
        [
            ExtractedSkill(name="Kubernetes", confidence=0.95,
                           source="jd", entity_type="TECH"),  # HIGH (conf≥0.85)
            ExtractedSkill(name="Terraform", confidence=0.7,
                           source="jd", entity_type="TECH"),  # MED
        ],
    )
    detail_map = {d.skill: d for d in details}
    assert detail_map["Kubernetes"].priority == "HIGH"
    assert detail_map["Terraform"].priority in ("MED", "HIGH")


def test_empty_resume_all_gaps():
    analyzer = GapAnalyzer()
    jd_skills = [skill("Python", "jd"), skill("Docker", "jd")]
    _, _, gaps, _ = analyzer.analyze([], jd_skills)
    assert set(gaps) == {"Python", "Docker"}


def test_empty_jd_returns_empty():
    analyzer = GapAnalyzer()
    known, partial, gaps, details = analyzer.analyze(
        [skill("Python")], []
    )
    assert known == [] and partial == [] and gaps == [] and details == []