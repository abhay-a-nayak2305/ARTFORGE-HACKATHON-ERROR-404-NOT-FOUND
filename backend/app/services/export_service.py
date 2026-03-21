"""
Export Service
──────────────
Generates downloadable report formats from an AnalysisResponse.
All heavy lifting (PDF, HTML, TXT) is done server-side so the
frontend download buttons can optionally use the API endpoint
instead of the client-side jsPDF.

Endpoints produced:
  GET /api/export/{session_id}?format=json
  GET /api/export/{session_id}?format=txt
  GET /api/export/{session_id}?format=html
"""
import json
from datetime import datetime
from app.models.schemas import AnalysisResponse


class ExportService:

    def to_json(self, analysis: AnalysisResponse) -> str:
        """Pretty-printed JSON export."""
        return json.dumps(analysis.model_dump(), indent=2)

    def to_txt(self, analysis: AnalysisResponse) -> str:
        """Plain text report matching the frontend TXT download format."""
        d = analysis
        trainable = [n for n in d.pathway_nodes if n.days > 0]
        total = sum(n.days for n in trainable)
        now = datetime.utcnow().strftime("%B %d, %Y")
        line = lambda c, n: c * n

        t = ""
        t += line("=", 60) + "\n"
        t += "  PATHFORGE — AI ADAPTIVE ONBOARDING REPORT\n"
        t += line("=", 60) + "\n\n"
        t += f"Generated  : {now}\n"
        t += f"Target Role: {d.role}\n"
        t += f"Match Score: {d.match_score}%\n"
        t += f"Confidence : {d.hallucination_guard.confidence_avg}%\n\n"

        t += line("-", 60) + "\n"
        t += "  EXECUTIVE SUMMARY\n"
        t += line("-", 60) + "\n"
        t += f"Training Duration : {total} days\n"
        t += f"Days Saved        : {d.days_saved} days\n"
        t += f"Critical Gaps     : {len(d.gap_skills)}\n"
        t += f"Role Readiness    : {d.match_score}%\n\n"

        t += line("-", 60) + "\n"
        t += "  SKILL GAP ANALYSIS\n"
        t += line("-", 60) + "\n"
        t += f"[PROFICIENT]\n  {', '.join(d.known_skills)}\n\n"
        t += f"[PARTIAL]\n  {', '.join(d.partial_skills)}\n\n"
        t += f"[GAPS — CRITICAL]\n  {', '.join(d.gap_skills)}\n\n"

        t += line("-", 60) + "\n"
        t += "  ADAPTIVE LEARNING PATHWAY\n"
        t += line("-", 60) + "\n"
        day = 1
        for i, n in enumerate(trainable):
            pri = f" [{n.priority}]" if n.priority else ""
            t += (
                f"  {str(i+1).zfill(2)}. "
                f"Day {str(day).zfill(2)} → Day {str(day + n.days - 1).zfill(2)}  |  "
                f"{n.node_type.upper().ljust(8)}  |  {n.label}{pri}\n"
            )
            day += n.days

        t += "\n" + line("-", 60) + "\n"
        t += "  AI REASONING TRACE\n"
        t += line("-", 60) + "\n"
        for step in d.reasoning_trace:
            t += f"[STEP {str(step.step_number).zfill(2)}] {step.step_name.upper()}\n"
            for detail in step.details:
                t += f"  · {detail}\n"
            t += f"  → {step.output_summary} (conf: {step.confidence:.0%})\n\n"

        t += line("-", 60) + "\n"
        t += "  HALLUCINATION GUARD\n"
        t += line("-", 60) + "\n"
        g = d.hallucination_guard
        t += f"  Status         : {'ACTIVE — 0 violations' if g.violations == 0 else 'VIOLATIONS FOUND'}\n"
        t += f"  Skills verified: {g.skills_verified_pct}%\n"
        t += f"  Confidence avg : {g.confidence_avg}%\n"
        t += f"  False positives: {g.false_positives}\n\n"

        t += line("=", 60) + "\n"
        t += "  PathForge · AI Adaptive Onboarding Engine v4.0\n"
        t += line("=", 60) + "\n"
        return t

    def to_html(self, analysis: AnalysisResponse) -> str:
        """
        Minimal styled HTML report (server-side version).
        The frontend also generates a richer client-side HTML;
        this version is used for the /api/export endpoint.
        """
        d = analysis
        trainable = [n for n in d.pathway_nodes if n.days > 0]
        total = sum(n.days for n in trainable)
        rows = ""
        day = 1
        for i, n in enumerate(trainable):
            rows += (
                f"<tr><td>{i+1}</td><td>{n.node_type.upper()}</td>"
                f"<td>{n.label}</td><td>Day {day}–{day+n.days-1}</td>"
                f"<td>{n.days}d</td><td>{n.priority or '—'}</td></tr>"
            )
            day += n.days

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>PathForge Report — {d.role}</title>
<style>
  body{{font-family:system-ui,sans-serif;background:#0a0f1e;color:#e2e8f0;padding:40px;}}
  h1{{color:#00e5b8;letter-spacing:.1em;}} h2{{color:#e8ff47;font-size:1rem;margin-top:2rem;}}
  table{{width:100%;border-collapse:collapse;margin-top:10px;}}
  th{{background:rgba(255,255,255,.07);padding:8px;text-align:left;font-size:.75rem;color:#94a3b8;}}
  td{{padding:7px 8px;border-bottom:1px solid rgba(255,255,255,.05);font-size:.82rem;}}
  .badge{{display:inline-block;padding:2px 8px;border-radius:4px;font-size:.7rem;font-weight:700;}}
  .gap{{background:rgba(247,37,133,.15);color:#f72585;}}
  .skill{{background:rgba(0,150,255,.15);color:#0096ff;}}
  .stat{{display:inline-block;background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.1);
         border-radius:10px;padding:14px 20px;margin:6px;min-width:110px;text-align:center;}}
  .stat-n{{font-size:2rem;font-weight:900;color:#00e5b8;}}
  .stat-l{{font-size:.65rem;color:#64748b;letter-spacing:.1em;text-transform:uppercase;}}
</style>
</head>
<body>
<h1>⬡ PATHFORGE — {d.role} Roadmap</h1>
<p style="color:#64748b;font-size:.8rem;">Generated {d.generated_at[:10]} · AI Confidence {d.hallucination_guard.confidence_avg}%</p>

<div style="margin:20px 0;">
  <div class="stat"><div class="stat-n">{len(d.gap_skills)}</div><div class="stat-l">Skill Gaps</div></div>
  <div class="stat"><div class="stat-n">{total}d</div><div class="stat-l">Training</div></div>
  <div class="stat"><div class="stat-n">{d.days_saved}d</div><div class="stat-l">Saved</div></div>
  <div class="stat"><div class="stat-n">{d.match_score}%</div><div class="stat-l">Match</div></div>
</div>

<h2>GAPS</h2>
<p>{' '.join(f'<span class="badge gap">{s}</span>' for s in d.gap_skills)}</p>
<h2>KNOWN</h2>
<p>{' '.join(f'<span class="badge skill">{s}</span>' for s in d.known_skills)}</p>

<h2>LEARNING PATHWAY</h2>
<table>
  <thead><tr><th>#</th><th>Type</th><th>Module</th><th>Schedule</th><th>Days</th><th>Priority</th></tr></thead>
  <tbody>{rows}</tbody>
</table>
<p style="color:#334155;font-size:.7rem;margin-top:40px;text-align:center;">
  PathForge · AI Adaptive Onboarding Engine · v4.0
</p>
</body></html>"""