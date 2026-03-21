"""
Reasoning Trace Logger
──────────────────────
Records each agent step's input, output, timing, and confidence
to produce the full XAI reasoning trace returned to the frontend.
"""
import time
from dataclasses import dataclass, field
from app.models.schemas import TraceStep
from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class _StepRecord:
    step_number: int
    step_name: str
    input_summary: str
    details: list[str] = field(default_factory=list)
    output_summary: str = ""
    confidence: float = 1.0
    start_time: float = field(default_factory=time.time)
    end_time: float = 0.0


class ReasoningTracer:
    """
    Context-manager-friendly tracer.
    Usage:
        tracer = ReasoningTracer()
        with tracer.step("Entity Extraction", "847 resume tokens"):
            ...
            tracer.add_detail("Found 12 technical skills")
            tracer.set_output("Extracted 16 skills", confidence=0.95)
    """

    def __init__(self):
        self._steps: list[_StepRecord] = []
        self._active: _StepRecord | None = None

    class _StepCtx:
        def __init__(self, tracer: "ReasoningTracer", record: _StepRecord):
            self._tracer = tracer
            self._record = record

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self._record.end_time = time.time()
            self._tracer._active = None

    def step(self, name: str, input_summary: str) -> "_StepCtx":
        record = _StepRecord(
            step_number=len(self._steps) + 1,
            step_name=name,
            input_summary=input_summary,
            start_time=time.time(),
        )
        self._steps.append(record)
        self._active = record
        return self._StepCtx(self, record)

    def add_detail(self, detail: str) -> None:
        if self._active:
            self._active.details.append(detail)

    def set_output(self, summary: str, confidence: float = 1.0) -> None:
        if self._active:
            self._active.output_summary = summary
            self._active.confidence = confidence

    def to_schema(self) -> list[TraceStep]:
        result = []
        for r in self._steps:
            elapsed = int((r.end_time - r.start_time) * 1000)
            result.append(TraceStep(
                step_number=r.step_number,
                step_name=r.step_name,
                input_summary=r.input_summary,
                output_summary=r.output_summary or "Completed",
                confidence=round(r.confidence, 3),
                duration_ms=elapsed,
                details=r.details,
            ))
        return result