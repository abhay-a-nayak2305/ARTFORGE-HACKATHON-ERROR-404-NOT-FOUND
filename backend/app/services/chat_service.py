"""
Context-Aware Chat Service — Groq Edition
──────────────────────────────────────────
Hybrid approach:
  1. Rule-based intent router handles common questions instantly (0ms latency)
  2. Groq LLM handles complex / open-ended questions with full context grounding
  3. All responses are grounded to the stored AnalysisResponse
"""
from app.models.schemas import AnalysisResponse, ChatRequest, ChatResponse
from app.agent.groq_client import GroqClient
from app.utils.logger import get_logger

logger = get_logger(__name__)

# In-memory session store (swap for Redis in production)
_session_cache: dict[str, AnalysisResponse] = {}


def store_session(session_id: str, analysis: AnalysisResponse) -> None:
    _session_cache[session_id] = analysis


def get_session(session_id: str) -> AnalysisResponse | None:
    return _session_cache.get(session_id)


class ChatService:
    def __init__(self):
        self._groq = GroqClient.get()

    # ── Sync version (kept for backward compat) ───────────
    def respond(self, req: ChatRequest) -> ChatResponse:
        import asyncio
        return asyncio.get_event_loop().run_until_complete(self.respond_async(req))

    # ── Async version (preferred) ─────────────────────────
    async def respond_async(self, req: ChatRequest) -> ChatResponse:
        analysis = get_session(req.session_id)
        if not analysis:
            return ChatResponse(
                reply="Session expired or not found. Please run a new analysis.",
                confidence=1.0,
            )

        q = req.message.lower()

        # ── Try rule-based first (fast, deterministic) ────
        rule_reply, context_used = self._rule_route(q, analysis)

        if rule_reply:
            return ChatResponse(
                reply=rule_reply,
                context_used=context_used,
                confidence=0.99,
            )

        # ── Groq for complex / open-ended questions ───────
        if self._groq.available:
            trainable = [n for n in analysis.pathway_nodes if n.days > 0]
            pathway_context = {
                "role": analysis.role,
                "match_score": analysis.match_score,
                "total_training_days": analysis.total_training_days,
                "days_saved": analysis.days_saved,
                "known_skills": analysis.known_skills,
                "gap_skills": analysis.gap_skills,
                "partial_skills": analysis.partial_skills,
                "module_names": [n.label for n in trainable],
                "violations": analysis.hallucination_guard.violations,
                "confidence": analysis.hallucination_guard.confidence_avg,
            }
            history = [{"role": m.role, "content": m.content} for m in req.history]
            try:
                groq_reply = await self._groq.chat_response(
                    question=req.message,
                    pathway_context=pathway_context,
                    chat_history=history,
                )
                if groq_reply:
                    return ChatResponse(
                        reply=groq_reply,
                        context_used=["groq_llm", "pathway_context"],
                        confidence=0.93,
                    )
            except Exception as e:
                logger.warning("chat_service.groq_failed: %s", e)

        # ── Final fallback: generic summary ───────────────
        return ChatResponse(
            reply=(
                f"Your {analysis.role} pathway has {len(analysis.gap_skills)} gaps "
                f"({', '.join(analysis.gap_skills[:3])}…), "
                f"{analysis.total_training_days} training days, "
                f"match score {analysis.match_score}%. "
                f"Days saved vs generic onboarding: {analysis.days_saved}."
            ),
            context_used=["fallback"],
            confidence=0.85,
        )

    def _rule_route(
        self, q: str, a: AnalysisResponse
    ) -> tuple[str | None, list[str]]:
        """
        Fast rule-based responses for common questions.
        Returns (reply, context_keys) or (None, []) if no rule matches.
        """
        trainable = [n for n in a.pathway_nodes if n.days > 0]
        total_days = sum(n.days for n in trainable)
        gap_nodes = [n for n in a.pathway_nodes if n.node_type == "gap"]
        high_pri = [n for n in gap_nodes if n.priority == "HIGH"]

        if any(k in q for k in ["long", "duration", "how many days", "how many weeks"]):
            weeks = total_days // 5
            return (
                f"Your {a.role} pathway spans {total_days} training days (~{weeks} working weeks). "
                f"Without PathForge, generic onboarding would take {total_days + a.days_saved} days.",
                ["total_training_days", "days_saved"],
            )

        if any(k in q for k in ["gap", "miss", "lack", "need to learn"]):
            return (
                f"You have {len(a.gap_skills)} critical gap(s): {', '.join(a.gap_skills[:5])}. "
                f"These were identified via BERT NER + cosine similarity (α=0.78) "
                f"and verified against O*NET v28.",
                ["gap_skills"],
            )

        if any(k in q for k in ["score", "match", "percent", "ready", "%"]):
            return (
                f"Your current role-readiness score is {a.match_score}%. "
                f"Completing all {len(gap_nodes)} gap modules will bring you to 100%.",
                ["match_score"],
            )

        if any(k in q for k in ["start", "begin", "first", "initial"]):
            first = high_pri[0].label if high_pri else (gap_nodes[0].label if gap_nodes else "N/A")
            return (
                f"Start with '{first}' — it's a HIGH-priority gap directly required by the JD. "
                f"Complete all {len(high_pri)} HIGH-priority modules before moving to MED/LOW.",
                ["pathway_nodes"],
            )

        if any(k in q for k in ["hard", "difficult", "tough", "complex", "challenging"]):
            hardest = max(trainable, key=lambda n: n.days, default=None)
            if hardest:
                return (
                    f"The most intensive module is '{hardest.label}' at {hardest.days} days "
                    f"({hardest.node_type.upper()}, {hardest.priority} priority).",
                    ["pathway_nodes"],
                )

        if any(k in q for k in ["save", "skip", "redund", "effici", "wast"]):
            return (
                f"PathForge saves you {a.days_saved} days by skipping "
                f"{len(a.known_skills)} skills you already know: {', '.join(a.known_skills[:4])}…",
                ["days_saved", "known_skills"],
            )

        if any(k in q for k in ["algo", "bert", "bfs", "how does", "how do you", "work"]):
            return (
                "PathForge pipeline: (1) BERT NER + Groq LLaMA-3.3-70B extract skills → "
                "(2) 768-dim MiniLM embeddings + cosine similarity identify gaps → "
                "(3) O*NET v28 grounds every skill → "
                "(4) Adaptive BFS orders modules by JD weight × depth → "
                "(5) Hallucination Guard verifies all outputs.",
                ["reasoning_trace"],
            )

        if any(k in q for k in ["confid", "accur", "trust", "reliable", "halluc"]):
            g = a.hallucination_guard
            return (
                f"AI confidence: {g.confidence_avg}%. "
                f"{g.skills_verified_pct}% of skills verified against O*NET. "
                f"Hallucination violations: {g.violations}. "
                f"All outputs grounded to your uploaded documents.",
                ["hallucination_guard"],
            )

        # No rule matched → let Groq handle it
        return None, []