"""
Unit tests for agent tools.
"""
import pytest
from app.agent.tools.gap_analyzer import GapAnalyzer
from app.agent.tools.graph_builder import GraphBuilder
from app.agent.tools.hallucination_guard import HallucinationGuard
from app.agent.reasoning_trace import ReasoningTracer
from app.models.schemas import ExtractedSkill, SkillGapItem, PathwayNode


def make_skill(name: str, source: str = "resume", conf: float = 0.9) -> ExtractedSkill:
    return ExtractedSkill(name=name, confidence=conf, source=source, entity_type="TECH")


def test_gap_analyzer_identifies_gaps():
    analyzer = GapAnalyzer()
    resume = [make_skill("Python"), make_skill("Git"), make_skill("SQL")]
    jd = [
        make_skill("Python", "jd"),
        make_skill("Kubernetes", "jd"),
        make_skill("System Design", "jd"),
    ]
    known, partial, gaps, details = analyzer.analyze(resume, jd)
    assert "Python" in known
    assert len(gaps) >= 1
    assert len(details) == len(jd)


def test_graph_builder_produces_valid_graph():
    builder = GraphBuilder()
    known = ["Python", "Git"]
    partial = ["Docker"]
    gaps = ["Kubernetes", "System Design"]
    details = [
        SkillGapItem(skill="Kubernetes", resume_score=0.1, jd_weight=0.9,
                     gap_magnitude=0.68, priority="HIGH", onet_verified=True),
        SkillGapItem(skill="System Design", resume_score=0.2, jd_weight=0.85,
                     gap_magnitude=0.58, priority="HIGH", onet_verified=True),
    ]
    nodes, edges = builder.build(known, partial, gaps, details, "Software Engineer")
    assert len(nodes) > 0
    assert len(edges) > 0
    node_ids = {n.id for n in nodes}
    for src, tgt in edges:
        assert src in node_ids, f"Edge source '{src}' not in nodes"
        assert tgt in node_ids, f"Edge target '{tgt}' not in nodes"


def test_reasoning_tracer():
    tracer = ReasoningTracer()
    import time
    with tracer.step("Test Step", "input summary"):
        tracer.add_detail("detail one")
        tracer.add_detail("detail two")
        tracer.set_output("output summary", confidence=0.92)
        time.sleep(0.01)
    steps = tracer.to_schema()
    assert len(steps) == 1
    assert steps[0].step_name == "Test Step"
    assert steps[0].confidence == 0.92
    assert len(steps[0].details) == 2
    assert steps[0].duration_ms >= 10


def test_hallucination_guard_accepts_known_skills():
    guard = HallucinationGuard()
    nodes = [
        PathwayNode(id="n1", label="Python", node_type="gap", days=3,
                    priority="HIGH", x=100, y=100),
        PathwayNode(id="n2", label="Docker", node_type="skill", days=2,
                    priority="MED", x=200, y=100),
    ]
    source_texts = ["Python Docker Kubernetes REST APIs Git CI/CD"]
    report = guard.verify(nodes, source_texts)
    # Python and Docker are in source text → should have 0 violations
    assert report.violations == 0


def test_hallucination_guard_flags_invented_skill():
    guard = HallucinationGuard()
    nodes = [
        PathwayNode(id="n1", label="MagicFramework9000XYZ", node_type="gap",
                    days=2, priority="HIGH", x=100, y=100),
    ]
    report = guard.verify(nodes, source_texts=["Python Docker"])
    assert report.violations >= 1
    assert "MagicFramework9000XYZ" in report.flagged_items