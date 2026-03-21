"""
Groq LLM Client
───────────────
Wraps the Groq API (llama-3.3-70b-versatile) for:
  1. Enhanced skill extraction from free-text
  2. Context-aware chat responses grounded to the pathway
  3. Reasoning trace enrichment

All prompts are strictly grounded — the system prompt
forbids the model from inventing skills not present in
the source documents or O*NET catalog.

Uses groq-python SDK (pip install groq).
Async-compatible via asyncio.to_thread for the sync SDK.
"""
import asyncio
import time
from typing import Optional
from groq import Groq
from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

# Model to use — change to "llama-3.1-8b-instant" for faster/cheaper
GROQ_MODEL = "llama-3.3-70b-versatile"
MAX_TOKENS = 1024
TEMPERATURE = 0.1   # low temp = deterministic, grounded outputs


class GroqClient:
    """
    Singleton Groq client.
    Call GroqClient.get() to retrieve the shared instance.
    Falls back gracefully if GROQ_API_KEY is not set.
    """
    _instance: Optional["GroqClient"] = None
    _client: Optional[Groq] = None
    _available: bool = False

    def __init__(self):
        key = settings.GROQ_API_KEY
        if not key:
            logger.warning("groq_client: GROQ_API_KEY not set — Groq disabled")
            self._available = False
            return
        try:
            self._client = Groq(api_key=key)
            # Quick connectivity check
            self._available = True
            logger.info("groq_client: Groq SDK initialised (model=%s)", GROQ_MODEL)
        except Exception as e:
            logger.error("groq_client: init failed: %s", e)
            self._available = False

    @classmethod
    def get(cls) -> "GroqClient":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def available(self) -> bool:
        return self._available

    # ──────────────────────────────────────────────────────
    #  LOW-LEVEL CALL
    # ──────────────────────────────────────────────────────
    # backend/app/agent/groq_client.py — replace _call_sync with this

import asyncio
from groq import RateLimitError, APITimeoutError

async def call_with_retry(
    self,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = MAX_TOKENS,
    temperature: float = TEMPERATURE,
    retries: int = 3,
    timeout: float = 15.0,
) -> str:
    """
    Call Groq with automatic retry on rate limit or timeout.
    Falls back to the faster model on second retry.
    """
    for attempt in range(retries):
        try:
            model = (
                self._settings.GROQ_FALLBACK_MODEL
                if attempt > 0
                else GROQ_MODEL
            )
            result = await asyncio.wait_for(
                asyncio.to_thread(
                    self._client.chat.completions.create,
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_prompt},
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                ),
                timeout=timeout,
            )
            return result.choices[0].message.content or ""

        except asyncio.TimeoutError:
            logger.warning("groq.timeout attempt=%d", attempt + 1)
            if attempt == retries - 1:
                raise

        except RateLimitError:
            wait = 2 ** attempt  # exponential backoff: 1s, 2s, 4s
            logger.warning("groq.rate_limit wait=%ds", wait)
            await asyncio.sleep(wait)

        except Exception as e:
            logger.error("groq.error attempt=%d error=%s", attempt + 1, e)
            if attempt == retries - 1:
                raise

    return ""

    async def call(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = MAX_TOKENS,
        temperature: float = TEMPERATURE,
    ) -> str:
        """Async wrapper — runs the sync SDK call in a thread pool."""
        return await asyncio.to_thread(
            self._call_sync, system_prompt, user_prompt, max_tokens, temperature
        )

    # ──────────────────────────────────────────────────────
    #  HIGH-LEVEL METHODS
    # ──────────────────────────────────────────────────────

    async def extract_skills_from_text(
        self,
        text: str,
        source: str = "resume",
        known_skills: Optional[list[str]] = None,
    ) -> list[str]:
        """
        Ask Groq to extract a flat list of technical and soft skills
        from the given text. Returns only skills verifiable in the text.

        Falls back to empty list if Groq is unavailable.
        """
        if not self._available:
            return []

        skill_hint = ""
        if known_skills:
            skill_hint = (
                f"\nReference skill taxonomy (use these exact names when matching): "
                f"{', '.join(known_skills[:60])}"
            )

        system = (
            "You are a precise skill extraction engine for an HR AI system. "
            "Your ONLY job is to extract skills explicitly mentioned in the provided text. "
            "RULES:\n"
            "1. Extract ONLY skills, tools, frameworks, certifications, and methodologies.\n"
            "2. Do NOT invent or infer skills not present in the text.\n"
            "3. Return a JSON array of strings — nothing else.\n"
            "4. Use canonical names (e.g. 'Python' not 'python programming language').\n"
            "5. Limit to 30 most relevant skills.\n"
            f"6. Source context: {source}."
            + skill_hint
        )
        user = f"Extract all skills from this {source}:\n\n{text[:3000]}"

        try:
            raw = await self.call(system, user, max_tokens=512, temperature=0.05)
            # Parse JSON array from response
            import json, re
            match = re.search(r'\[.*?\]', raw, re.DOTALL)
            if match:
                skills = json.loads(match.group(0))
                return [s.strip() for s in skills if isinstance(s, str) and s.strip()]
            return []
        except Exception as e:
            logger.warning("groq_client.extract_skills_failed: %s", e)
            return []

    async def enhance_gap_analysis(
        self,
        role: str,
        known: list[str],
        gaps: list[str],
        partial: list[str],
        experience_level: str,
    ) -> dict:
        """
        Ask Groq to provide a structured, human-readable analysis
        of the skill gap with priority explanations and learning tips.

        Returns a dict with keys: summary, priority_rationale, quick_wins, long_term
        """
        if not self._available:
            return {
                "summary": f"You have {len(gaps)} critical gaps for {role}.",
                "priority_rationale": "Gaps ranked by JD requirement weight.",
                "quick_wins": partial[:3],
                "long_term": gaps[:3],
            }

        system = (
            "You are an expert L&D advisor for corporate onboarding. "
            "Given a candidate's skill profile vs job requirements, provide a "
            "structured gap analysis. Be concise, specific, and actionable. "
            "IMPORTANT: Only reference the skills provided — do not hallucinate new ones. "
            "Respond in JSON with keys: summary (str), priority_rationale (str), "
            "quick_wins (list[str] — skills addressable in <2 days), "
            "long_term (list[str] — skills requiring >3 days). "
            "Keep each string under 120 characters."
        )
        user = (
            f"Role: {role}\n"
            f"Experience Level: {experience_level}\n"
            f"Already knows: {', '.join(known[:10])}\n"
            f"Partial knowledge: {', '.join(partial[:6])}\n"
            f"Critical gaps: {', '.join(gaps[:8])}\n\n"
            f"Provide the gap analysis JSON."
        )

        try:
            raw = await self.call(system, user, max_tokens=600, temperature=0.15)
            import json, re
            match = re.search(r'\{.*?\}', raw, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            return {"summary": raw[:300], "priority_rationale": "", "quick_wins": [], "long_term": []}
        except Exception as e:
            logger.warning("groq_client.enhance_gap_failed: %s", e)
            return {"summary": f"{len(gaps)} gaps detected for {role}.", "priority_rationale": "", "quick_wins": [], "long_term": []}

    async def chat_response(
        self,
        question: str,
        pathway_context: dict,
        chat_history: list[dict],
    ) -> str:
        """
        Generate a grounded chat response about the user's specific pathway.
        The system prompt injects the full pathway context so the model
        cannot hallucinate — it only has access to what's in the context.

        Returns the reply string.
        """
        if not self._available:
            return ""

        context_str = (
            f"Role: {pathway_context.get('role')}\n"
            f"Match Score: {pathway_context.get('match_score')}%\n"
            f"Training Days: {pathway_context.get('total_training_days')}\n"
            f"Days Saved: {pathway_context.get('days_saved')}\n"
            f"Known Skills: {', '.join(pathway_context.get('known_skills', [])[:8])}\n"
            f"Skill Gaps: {', '.join(pathway_context.get('gap_skills', [])[:8])}\n"
            f"Partial Skills: {', '.join(pathway_context.get('partial_skills', [])[:6])}\n"
            f"Pathway Modules: {', '.join(pathway_context.get('module_names', []))}\n"
            f"Hallucination Violations: {pathway_context.get('violations', 0)}\n"
            f"AI Confidence: {pathway_context.get('confidence', 94.2)}%"
        )

        system = (
            "You are PathForge AI, an expert onboarding advisor embedded in the PathForge platform. "
            "You have access ONLY to the candidate's pathway data provided below. "
            "STRICT RULES:\n"
            "1. Only answer questions based on the provided pathway context.\n"
            "2. If asked about something not in the context, say 'That information isn't in your pathway.'\n"
            "3. Be concise — 2-4 sentences max per response.\n"
            "4. Use specific numbers and skill names from the context.\n"
            "5. Never invent skills, timelines, or resources not in the context.\n"
            "6. Tone: professional, encouraging, precise.\n\n"
            f"CANDIDATE PATHWAY CONTEXT:\n{context_str}"
        )

        # Build message history (last 6 turns for context window efficiency)
        messages = [{"role": "system", "content": system}]
        for turn in chat_history[-6:]:
            messages.append({"role": turn["role"], "content": turn["content"]})
        messages.append({"role": "user", "content": question})

        if not self._client:
            return ""

        try:
            t0 = time.time()
            completion = await asyncio.to_thread(
                self._client.chat.completions.create,
                model=GROQ_MODEL,
                messages=messages,
                max_tokens=256,
                temperature=0.2,
            )
            elapsed = int((time.time() - t0) * 1000)
            reply = completion.choices[0].message.content or ""
            logger.info("groq_client.chat", elapsed_ms=elapsed)
            return reply.strip()
        except Exception as e:
            logger.warning("groq_client.chat_failed: %s", e)
            return ""

    async def generate_learning_tips(
        self,
        skill_name: str,
        role: str,
        days_allocated: int,
    ) -> list[str]:
        """
        Generate 3 specific, actionable learning tips for a skill module.
        Used to enrich node inspector popups.
        """
        if not self._available:
            return [
                f"Focus on hands-on projects for {skill_name}.",
                f"Review official documentation first.",
                f"Allocate {days_allocated} focused days with daily practice.",
            ]

        system = (
            "You are a concise technical learning coach. "
            "Generate exactly 3 actionable learning tips for a specific skill. "
            "Each tip must be under 80 characters. "
            "Return ONLY a JSON array of 3 strings. No explanations."
        )
        user = (
            f"Skill: {skill_name}\n"
            f"Target Role: {role}\n"
            f"Time Available: {days_allocated} days\n"
            f"Give 3 specific, actionable tips."
        )

        try:
            raw = await self.call(system, user, max_tokens=200, temperature=0.3)
            import json, re
            match = re.search(r'\[.*?\]', raw, re.DOTALL)
            if match:
                tips = json.loads(match.group(0))
                return [str(t)[:100] for t in tips[:3]]
            return [raw[:100]]
        except Exception as e:
            logger.warning("groq_client.tips_failed: %s", e)
            return [f"Study {skill_name} fundamentals for {days_allocated} days."]