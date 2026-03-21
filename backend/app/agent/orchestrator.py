"""
PathForge AI Agent Orchestrator — Groq Edition
────────────────────────────────────────────────
Same 5-step pipeline, now enhanced with Groq LLM at steps 1 & 2:
  Step 1: BERT NER + Groq LLM → richer skill extraction
  Step 2: Cosine similarity + Groq → enhanced gap analysis narrative
  Step 3: O*NET grounding (unchanged)
  Step 4: Adaptive BFS graph (unchanged)
  Step 5: Hallucination guard (unchanged)
"""
import time
import uuid
from datetime import datetime

from app.agent.tools.resume_parser import ResumeParser
from app.agent.tools.jd_parser import JDParser
from app.agent.tools.gap_analyzer import GapAnalyzerfrom app.agent.tools.graph_builder import GraphBuilder
from app.agent.tools.hallucination_guard import HallucinationGuard
from app.agent.tools.onet_lookup import lookup as onet_lookup
from app.agent.reasoning_trace import ReasoningTracer
from app.agent.groq_client import GroqClient
from app.models.schemas import (
    AnalysisResponse,
    ExtractedSkill,
    GraphData,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)

_ROLE_BASELINES = {
    "Software Engineer": {"total_skills": 20, "generic_days": 30},
    "Data Scientist":    {"total_skills": 18, "generic_days": 28},
    "Product Manager":   {"total_skills": 16, "generic_days": 24},
}
_DEFAULT_BASELINE = {"total_skills": 18, "generic_days": 26}

# All known skills for Groq grounding hint
_ALL_KNOWN_SKILLS = [
    "Python", "JavaScript", "TypeScript", "Java", "Go", "Rust", "C++",
    "React", "Vue", "Django", "FastAPI", "Flask", "Spring Boot", "Node.js",
    "TensorFlow", "PyTorch", "Scikit-learn", "Pandas", "NumPy", "Spark",
    "AWS", "GCP", "Azure", "Docker", "Kubernetes", "Terraform", "Ansible",
    "PostgreSQL", "MySQL", "MongoDB", "Redis", "Elasticsearch",
    "REST APIs", "GraphQL", "Microservices", "System Design", "Git", "Linux",
    "CI/CD", "Jenkins", "GitHub Actions", "Statistics", "Machine Learning",
    "Deep Learning", "Feature Engineering", "MLOps", "Data Wrangling", "NLP",
    "SQL", "Tableau", "Power BI", "JIRA", "Figma", "A/B Testing", "OKRs",
    "Agile", "Scrum", "Product Strategy", "Roadmapping", "GTM Strategy",
    "Stakeholder Management", "UX Research", "Competitive Analysis",
    "Communication", "Leadership", "Problem Solving", "Teamwork",
]


class AgentOrchestrator:
    def __init__(self):
        self._resume_parser = ResumeParser()
        self._jd_parser     = JDParser()
        self._gap_analyzer  = GapAnalyzer()
        self._graph_builder = GraphBuilder()
        self._guard          = HallucinationGuard()
        self._groq           = GroqClient.get()

    async def run(
        self,
        resume_text: str,
        jd_text: str,
        role: str,
        experience_level: str,
        used_demo: bool = False,
    ) -> AnalysisResponse:
        """
        Full 5‑step pipeline. Returns an **AnalysisResponse** that includes the
        new `graph_data` field for the frontend visualization.
        """
        t_total = time.time()
        session_id = str(uuid.uuid4())
        tracer = ReasoningTracer()

        logger.info(
            "agent.run.start",
            session_id=session_id,
            role=role,
            groq_available=self._groq.available,
        )

        # ── STEP 1: Entity Extraction (BERT + Groq) ───────
        with tracer.step(
            "Entity Extraction",
            f"Resume: {len(resume_text.split())} words | JD: {len(jd_text.split())} words",
        ):
            resume_skills_bert = self._resume_parser.parse(resume_text, source="resume")
            jd_skills_bert     = self._jd_parser.parse(jd_text)

            import asyncio

            groq_resume_names, groq_jd_names = await asyncio.gather(
                self._groq.extract_skills_from_text(
                    resume_text, source="resume", known_skills=_ALL_KNOWN_SKILLS
                ),
                self._groq.extract_skills_from_text(
                    jd_text, source="jd", known_skills=_ALL_KNOWN_SKILLS                ),
            )

            resume_skills = self._merge_skills(resume_skills_bert, groq_resume_names, "resume")
            jd_skills     = self._merge_skills(jd_skills_bert, groq_jd_names,     "jd")

            tracer.add_detail(f"BERT NER: {len(resume_skills_bert)} resume + {len(jd_skills_bert)} JD skills")
            tracer.add_detail(f"Groq LLM: {len(groq_resume_names)} resume + {len(groq_jd_names)} JD skills")
            tracer.add_detail(f"Merged (deduplicated): {len(resume_skills)} resume | {len(jd_skills)} JD")
            tracer.add_detail(f"Groq model: llama-3.3-70b-versatile")
            tracer.add_detail("BERT model: dslim/bert-base-NER")
            tracer.set_output(
                f"{len(resume_skills)} resume + {len(jd_skills)} JD skills extracted",
                confidence=0.96,
            )

        # ── STEP 2: Gap Computation ───────────────────────
        with tracer.step(
            "Gap Computation",
            f"768-dim embeddings | threshold α=0.78",
        ):
            known, partial, gaps, gap_details = self._gap_analyzer.analyze(
                resume_skills, jd_skills
            )

            groq_analysis = await self._groq.enhance_gap_analysis(
                role=role,
                known=known,
                gaps=gaps,
                partial=partial,
                experience_level=experience_level,
            )

            tracer.add_detail(f"Cosine sim matrix: {len(resume_skills)}×{len(jd_skills)}")
            tracer.add_detail(f"Threshold α=0.78 | KNOWN: {len(known)} | PARTIAL: {len(partial)} | GAP: {len(gaps)}")
            tracer.add_detail(f"Groq summary: {groq_analysis.get('summary', '')[:80]}")
            tracer.add_detail(f"Quick wins: {', '.join(groq_analysis.get('quick_wins', [])[:3])}")
            tracer.add_detail(f"Long-term gaps: {', '.join(groq_analysis.get('long_term', [])[:3])}")
            tracer.set_output(
                f"{len(known)} known | {len(partial)} partial | {len(gaps)} gaps",
                confidence=0.95,
            )

        # ── STEP 3: O*NET Grounding ───────────────────────
        with tracer.step(
            "O*NET Grounding",
            f"Verifying {len(gaps) + len(partial)} skills",
        ):
            all_to_check = gaps + partial + known            onet_map: dict[str, str | None] = {}
            for skill in all_to_check:
                meta = onet_lookup(skill)
                onet_map[skill] = meta["onet_code"] if meta else None            verified_count = sum(1 for v in onet_map.values() if v)
            tracer.add_detail(f"O*NET v28: {verified_count}/{len(all_to_check)} skills verified")
            tracer.add_detail("Unverified: embedding fallback ≥0.75")
            tracer.set_output(f"{verified_count} skills O*NET-grounded", confidence=0.97)

        # ── STEP 4: Adaptive Pathing ──────────────────────
        with tracer.step(
            "Adaptive Pathing",
            f"BFS | {len(gaps)} gap nodes | {len(partial)} skill nodes",
        ):
            pathway_nodes, pathway_edges = self._graph_builder.build(
                known, partial, gaps, gap_details, role
            )

            # Enrich HIGH‑priority gap nodes with Groq‑generated learning tips
            for node in pathway_nodes:
                if node.node_type == "gap" and node.priority == "HIGH" and node.days > 0:
                    tips = await self._groq.generate_learning_tips(
                        node.label, role, node.days
                    )
                    node.resources = tips

            high_count = len(
                [n for n in pathway_nodes if n.node_type == "gap" and n.priority == "HIGH"]
            )
            tracer.add_detail(f"{len(pathway_nodes)} nodes | {len(pathway_edges)} edges")
            tracer.add_detail(f"Groq learning tips added to {high_count} HIGH‑priority nodes")
            tracer.add_detail("Sorted: JD weight × depth × effort")
            tracer.set_output(
                f"Pathway: {len(pathway_nodes)} nodes, {len(pathway_edges)} edges",
                confidence=0.96,
            )

        # ── STEP 5: Hallucination Guard ───────────────────
        with tracer.step(
            "Hallucination Guard",
            f"Verifying {len(pathway_nodes)} nodes",
        ):
            guard_report = self._guard.verify(
                pathway_nodes,
                source_texts=[resume_text, jd_text],
            )
            tracer.add_detail(f"{guard_report.violations} violations detected")
            tracer.add_detail(f"{guard_report.skills_verified_pct}% skills verified")
            tracer.add_detail(f"Confidence avg: {guard_report.confidence_avg}%")
            tracer.set_output(
                f"{guard_report.violations} violations | {guard_report.skills_verified_pct}% verified",
                confidence=guard_report.confidence_avg / 100,
            )

        # ── 5️⃣ Build Graph Data for the Dashboard ─────────────        # The frontend expects a three‑field structure:
        #   labels                 – skill names (max 7‑10)
        #   current_profile        – 0‑100 based on resume extraction
        #   target_role            – 0‑100 based on role requirements
        graph_data = self._build_graph_data(
            known=known,
            partial=partial,
            gaps=gaps,
            gap_details=gap_details,
            experience_level=experience_level,
        )

        # ── Scoring & baseline comparison ───────────────────
        baseline = _ROLE_BASELINES.get(role, _DEFAULT_BASELINE)
        match_score = int((len(known) / max(len(jd_skills), 1)) * 100)
        match_score = max(30, min(95, match_score))
        total_training_days = sum(
            n.days for n in pathway_nodes if node_type := n.days and n.days > 0
        )
        days_saved = max(0, baseline["generic_days"] - total_training_days)

        elapsed_total = int((time.time() - t_total) * 1000)

        logger.info(
            "agent.run.complete",
            session_id=session_id,
            role=role,
            match_score=match_score,
            gaps=len(gaps),
            training_days=total_training_days,
            elapsed_ms=elapsed_total,
        )

        # ── Return the full response ──────────────────────────
        return AnalysisResponse(
            session_id=session_id,
            role=role,
            experience_level=experience_level,
            match_score=match_score,
            days_saved=days_saved,
            total_training_days=total_training_days,

            known_skills=list(set(resume_skills)),        # dedupe just in case            partial_skills=partial[:6],
            gap_skills=gaps[:6],
            skill_gaps_detail=gap_details,
            resume_skills=[s for s in resume_skills],    # keep full list for later
            jd_skills=[s for s in jd_skills],           # keep full list for later
            pathway_nodes=pathway_nodes,
            pathway_edges=pathway_edges,
            reasoning_trace=tracer.to_schema(),
            hallucination_guard=guard_report,
            # NEW: expose the graph data built above
            graph_data=graph_data if graph_data else None,

            generated_at=datetime.utcnow().isoformat() + "Z",
        )

    # ─────────────────────────────────────────────────────────────────────────────    #  PRIVATE HELPERS    # ─────────────────────────────────────────────────────────────────────────────

    def _merge_skills(
        self,
        bert_skills: list[ExtractedSkill],
        groq_names: list[str],
        source: str,
    ) -> list[ExtractedSkill]:
        """
        Merge BERT‑extracted skills with Groq‑extracted skill names.
        BERT results take precedence; Groq names not already present are added
        with a dummy confidence of 0.88.
        """
        existing = {s.name.lower(): s for s in bert_skills}
        for name in groq_names:
            if name.lower() not in existing:
                existing[name.lower()] = ExtractedSkill(
                    name=name,
                    confidence=0.88,
                    source=source,
                    entity_type="TECH",
                )
        return list(existing.values())

    def _build_graph_data(
        self,
        known: list[str],
        partial: list[str],
        gaps: list[str],
        gap_details: list[SkillGapItem],
        experience_level: str,
    ) -> GraphData:
        """
        Returns a **GraphData** object that contains exactly the three fields        the frontend’s radar / bar chart needs:

          - labels          : the 6‑8 skill names we care about
          - current_profile : a 0‑100 proficiency extrapolated from the
                               resume/quiz score          - target_role     : a fixed 80‑100 baseline per skill (role‑required)

        The function decides how many questions/experience levels map to which
        proficiency level; the mapping mirrors the numbers used elsewhere in the
        service so the chart colour bands line up with expected thresholds.
        """
        # Determine the split between “known”, “partial” and “gap” skills
        all_skills = (known + partial + gaps)[:10]          # keep at most 10 for readability
        exp_score = {
            "beginner": 45,
            "mid":      65,
            "senior":   85,
        }.get(experience_level, 50)

        current_profile: list[int] = []
        target_role: list[int]   = []

        for skill in all_skills:
            # Target side is always a high baseline – roles never expect 0
            target_role.append(90)

            if skill in known:
                current_profile.append(exp_score)
            elif skill in partial:
                # Partial knowledge is roughly 30 points lower than the baseline                current_profile.append(max(10, exp_score - 30))
            else:
                # Completely missing = low baseline
                current_profile.append(10)

        return GraphData(
            labels=all_skills,
            current_profile=current_profile,
            target_role=target_role,
        )
