"""
O*NET Lookup Tool
─────────────────
Loads the O*NET skills CSV (or falls back to the bundled role_competencies.json)
and maps skill names to O*NET SOC codes + descriptions for grounding.
"""
import json
import csv
import os
from pathlib import Path
from functools import lru_cache
from app.utils.logger import get_logger

logger = get_logger(__name__)

BASE_DIR = Path(__file__).resolve().parents[4]  # project root


@lru_cache(maxsize=1)
def load_onet_skills() -> dict[str, dict]:
    """
    Returns dict: skill_name_lower → {onet_code, title, description}
    Tries CSV first, falls back to role_competencies.json.
    """
    csv_path = BASE_DIR / "data" / "onet_skills.csv"
    if csv_path.exists():
        return _load_from_csv(csv_path)
    logger.warning("O*NET CSV not found at %s; using bundled fallback", csv_path)
    return _load_from_json()


def _load_from_csv(path: Path) -> dict[str, dict]:
    skills: dict[str, dict] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("Element Name", "").strip()
            if not name:
                continue
            skills[name.lower()] = {
                "onet_code": row.get("Element ID", ""),
                "title": name,
                "description": row.get("Description", ""),
                "category": row.get("Scale Name", ""),
            }
    logger.info("Loaded %d O*NET skills from CSV", len(skills))
    return skills


def _load_from_json() -> dict[str, dict]:
    json_path = BASE_DIR / "data" / "role_competencies.json"
    if not json_path.exists():
        logger.error("role_competencies.json not found at %s", json_path)
        return {}
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    skills: dict[str, dict] = {}
    for role, competencies in data.items():
        for skill_name in competencies.get("skills", []):
            skills[skill_name.lower()] = {
                "onet_code": f"FALLBACK-{role[:3].upper()}",
                "title": skill_name,
                "description": f"Core competency for {role}",
                "category": role,
            }
    return skills


def lookup(skill_name: str) -> dict | None:
    """Return O*NET metadata for a skill name, or None if not found."""
    db = load_onet_skills()
    return db.get(skill_name.lower())


def verify_skills(skill_names: list[str]) -> tuple[list[str], list[str]]:
    """
    Returns (verified, unverified) lists.
    Verified = found in O*NET database.
    """
    db = load_onet_skills()
    verified = [s for s in skill_names if s.lower() in db]
    unverified = [s for s in skill_names if s.lower() not in db]
    return verified, unverified