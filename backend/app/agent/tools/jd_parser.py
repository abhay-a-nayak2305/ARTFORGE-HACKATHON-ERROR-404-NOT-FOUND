"""
Job Description Parser
──────────────────────
Extracts required skills, weights them by JD emphasis signals
(e.g., "Required:" section gets weight 1.0, "Nice-to-have:" gets 0.5),
and returns a list of ExtractedSkill with source="jd".
"""
import re
from app.agent.tools.resume_parser import ResumeParser
from app.models.schemas import ExtractedSkill
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Section weight map: section heading → importance multiplier
_SECTION_WEIGHTS: dict[str, float] = {
    "required": 1.0,
    "must have": 1.0,
    "mandatory": 1.0,
    "qualifications": 0.9,
    "responsibilities": 0.85,
    "preferred": 0.6,
    "nice to have": 0.5,
    "nice-to-have": 0.5,
    "bonus": 0.4,
    "desirable": 0.4,
}

_SECTION_RE = re.compile(
    r"(" + "|".join(re.escape(k) for k in _SECTION_WEIGHTS) + r")\s*[:\-–]",
    flags=re.IGNORECASE,
)


class JDParser:
    def __init__(self):
        self._resume_parser = ResumeParser()

    def parse(self, jd_text: str) -> list[ExtractedSkill]:
        """
        Parse a job description into weighted ExtractedSkill objects.
        Skills in "Required" sections get higher JD weight.
        """
        sections = self._split_sections(jd_text)
        all_skills: dict[str, ExtractedSkill] = {}

        for section_name, section_text, weight in sections:
            raw_skills = self._resume_parser.parse(section_text, source="jd")
            for skill in raw_skills:
                if skill.name not in all_skills:
                    # Store with section weight embedded in confidence
                    adjusted_conf = round(min(skill.confidence * weight, 1.0), 3)
                    all_skills[skill.name] = ExtractedSkill(
                        name=skill.name,
                        confidence=adjusted_conf,
                        source="jd",
                        entity_type=skill.entity_type,
                    )
                else:
                    # Keep highest confidence
                    existing = all_skills[skill.name]
                    adjusted_conf = round(min(skill.confidence * weight, 1.0), 3)
                    if adjusted_conf > existing.confidence:
                        all_skills[skill.name] = ExtractedSkill(
                            name=skill.name,
                            confidence=adjusted_conf,
                            source="jd",
                            entity_type=skill.entity_type,
                        )

        logger.info("jd_parser.parse", total_jd_skills=len(all_skills))
        return list(all_skills.values())

    def _split_sections(
        self, text: str
    ) -> list[tuple[str, str, float]]:
        """
        Split JD text into named sections with their weights.
        Returns list of (section_name, text, weight) tuples.
        """
        sections: list[tuple[str, str, float]] = []
        matches = list(_SECTION_RE.finditer(text))

        if not matches:
            # No section headers found — treat whole text as "required"
            return [("required", text, 1.0)]

        # Text before first match
        if matches[0].start() > 0:
            sections.append(("preamble", text[: matches[0].start()], 0.85))

        for i, m in enumerate(matches):
            section_name = m.group(1).lower()
            weight = _SECTION_WEIGHTS.get(section_name, 0.75)
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            sections.append((section_name, text[start:end], weight))

        return sections