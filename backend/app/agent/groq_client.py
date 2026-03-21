"""
Groq LLM Client
───────────────
FIXES:
1. _call_sync was never defined — every Groq call crashed with AttributeError.
2. call_with_retry was a dangling function outside the class body.
3. RateLimitError / APITimeoutError don't exist in groq==0.9.0 — replaced
   with generic Exception + response status code checking.
4. self._settings didn't exist — was referencing module-level `settings`.
5. Duplicate mid-file imports removed; all imports are now at the top.
"""

import asyncio
import json
import re
import time
import logging
from typing import Optional

from groq import Groq
from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_FALLBACK_MODEL = "llama-3.1-8b-instant"
MAX_TOKENS = 1024
TEMPERATURE = 0.1


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
    # LOW-LEVEL SYNC CALL (runs in thread pool via asyncio.to_thread)
    # ──────────────────────────────────────────────────────

    def _call_sync(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = MAX_TOKENS,
        temperature: float = TEMPERATURE,
        model: str = GROQ_MODEL,
    ) -> str:
        """
        Synchronous Groq API call. Always called via asyncio.to_thread
        so it never blocks the event loop.
        FIX: This method was completely missing in the original file,
        causing AttributeError on every single Groq call.
        """
        if not self._client:
            return ""
        result = self._client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return result.choices[0].message.content or ""

    # ──────────────────────────────────────────────────────
    # ASYNC CALL WITH RETRY + FALLBACK MODEL
    # ──────────────────────────────────────────────────────

    async def call(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = MAX_TOKENS,
        temperature: float = TEMPERATURE,
        retries: int = 3,
        timeout: float = 30.0,
    ) -> str:
        """
        Async Groq call with automatic retry and fallback model.

        FIX: The original call() delegated to self._call_sync which didn't
        exist. The original call_with_retry() was a dangling module-level
        function (not a method), referenced self._settings (doesn't exist),
        and imported RateLimitError/APITimeoutError which don't exist in
        groq==0.9.0. All of that is replaced here cleanly.

        Retry strategy:
          - Attempt 1: primary model (llama-3.3-70b-versatile), full timeout
          - Attempt 2+: fallback model (llama-3.1-8b-instant), shorter timeout
          - Exponential backoff between retries: 1s, 2s, 4s
        """
        if not self._available or not self._client:
            return ""

        for attempt in range(retries):
            # Switch to faster/cheaper fallback model after first failure
            model = GROQ_MODEL if attempt == 0 else GROQ_FALLBACK_MODEL
            # Reduce timeout on retries to fail faster
            attempt_timeout = timeout if attempt == 0 else timeout * 0.6

            try:
                logger.info(
                    "groq.call attempt=%d model=%s max_tokens=%d",
                    attempt + 1, model, max_tokens,
                )
                result = await asyncio.wait_for(
                    asyncio.to_thread(
                        self._call_sync,
                        system_prompt,
                        user_prompt,
                        max_tokens,
                        temperature,
                        model,
                    ),
                    timeout=attempt_timeout,
                )
                return result

            except asyncio.TimeoutError:
                logger.warning("groq.timeout attempt=%d timeout=%.1fs", attempt + 1, attempt_timeout)
                if attempt == retries - 1:
                    logger.error("groq.all_retries_timed_out")
                    return ""

            except Exception as e:
                err_str = str(e).lower()
                # groq==0.9.0 surfaces rate limit as an HTTP exception with
                # "rate_limit" in the message — catch generically
                if "rate_limit" in err_str or "429" in err_str:
                    wait = 2 ** attempt  # 1s, 2s, 4s
                    logger.warning("groq.rate_limit wait=%ds attempt=%d", wait, attempt + 1)
                    await asyncio.sleep(wait)
                elif "401" in err_str or "invalid_api_key" in err_str:
                    logger.error("groq.invalid_api_key — check GROQ_API_KEY env var on Render")
                    self._available = False
                    return ""
                else:
                    logger.error("groq.error attempt=%d error=%s", attempt + 1, e)
                    if attempt == retries - 1:
                        return ""
                    await asyncio.sleep(1.5 ** attempt)

        return ""

    # ──────────────────────────────────────────────────────
    # HIGH-LEVEL METHODS
    # ──────────────────────────────────────────────────────

    async def extract_skills_from_text(
        self,
        text: str,
        source: str = "resume",
        known_skills: Optional[list[str]] = None,
    ) -> list[str]:
        """
        Extract skills from free text. Returns empty list if Groq unavailable.
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
        Return structured gap analysis with priority explanations.
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
            match = re.search(r'\{.*?\}', raw, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            return {"summary": raw[:300], "priority_rationale": "", "quick_wins": [], "long_term": []}
        except Exception as e:
            logger.warning("groq_client.enhance_gap_failed: %s", e)
            return {
                "summary": f"{len(gaps)} gaps detected for {role}.",
                "priority_rationale": "",
                "quick_wins": [],
                "long_term": [],
            }

    async def chat_response(
        self,
        question: str,
        pathway_context: dict,
        chat_history: list[dict],
    ) -> str:
        """
        Generate a grounded chat response about the user's specific pathway.
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

        # Build full message list including history (last 6 turns)
        messages = [{"role": "system", "content": system}]
        for turn in chat_history[-6:]:
            messages.append({"role": turn["role"], "content": turn["content"]})
        messages.append({"role": "user", "content": question})

        if not self._client:
            return ""

        try:
            t0 = time.time()
            # Chat needs the full messages list, so call _call_sync directly
            # with a custom messages param via to_thread
            completion = await asyncio.wait_for(
                asyncio.to_thread(
                    lambda: self._client.chat.completions.create(
                        model=GROQ_MODEL,
                        messages=messages,
                        max_tokens=256,
                        temperature=0.2,
                    )
                ),
                timeout=20.0,
            )
            elapsed = int((time.time() - t0) * 1000)
            reply = completion.choices[0].message.content or ""
            logger.info("groq_client.chat elapsed_ms=%d", elapsed)
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
        """
        if not self._available:
            return [
                f"Focus on hands-on projects for {skill_name}.",
                "Review official documentation first.",
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
            "Give 3 specific, actionable tips."
        )

        try:
            raw = await self.call(system, user, max_tokens=200, temperature=0.3)
            match = re.search(r'\[.*?\]', raw, re.DOTALL)
            if match:
                tips = json.loads(match.group(0))
                return [str(t)[:100] for t in tips[:3]]
            return [raw[:100]]
        except Exception as e:
            logger.warning("groq_client.tips_failed: %s", e)
            return [f"Study {skill_name} fundamentals for {days_allocated} days."]
