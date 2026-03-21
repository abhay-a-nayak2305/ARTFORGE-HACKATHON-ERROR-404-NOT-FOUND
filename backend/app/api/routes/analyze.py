"""
POST /api/analyze  — now uses Groq LLM for enhanced reasoning
"""
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import JSONResponse
from typing import Optional, Annotated
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.orchestrator import AgentOrchestrator
from app.services.analysis_service import AnalysisService
from app.services.chat_service import store_session
from app.dependencies import get_db, get_orchestrator
from app.utils.file_handler import (
    extract_text, get_demo_resume, get_demo_jd, validate_file_size,
)
from app.config import get_settings
from app.agent.orchestrator import AgentOrchestrator
from app.utils.logger import get_logger

router = APIRouter(prefix="/api", tags=["analysis"])
logger = get_logger(__name__)
settings = get_settings()


@router.post("/analyze")
async def analyze(
    role: Annotated[str, Form()],
    experience_level: Annotated[str, Form()],
    use_demo: Annotated[bool, Form()] = False,
    resume_file: Optional[UploadFile] = File(None),
    jd_file: Optional[UploadFile] = File(None),
    db: AsyncSession = Depends(get_db),
    orchestrator: AgentOrchestrator = Depends(get_orchestrator),
):
    # ── Resolve resume text ───────────────────────────────
    if use_demo or resume_file is None:
        resume_text = get_demo_resume()
        used_demo = True
    else:
        raw = await resume_file.read()
        err = validate_file_size(len(raw), settings.MAX_FILE_SIZE_MB)
        if err:
            raise HTTPException(status_code=413, detail=err)
        try:
            resume_text = extract_text(raw, resume_file.filename or "resume.txt")
        except ValueError as e:
            raise HTTPException(status_code=415, detail=str(e))
        used_demo = False

    # ── Resolve JD text ───────────────────────────────────
    if use_demo or jd_file is None:
        jd_text = get_demo_jd(role)
    else:
        raw = await jd_file.read()
        err = validate_file_size(len(raw), settings.MAX_FILE_SIZE_MB)
        if err:
            raise HTTPException(status_code=413, detail=err)
        try:
            jd_text = extract_text(raw, jd_file.filename or "jd.txt")
        except ValueError as e:
            raise HTTPException(status_code=415, detail=str(e))

    if not resume_text.strip() or not jd_text.strip():
        raise HTTPException(
            status_code=422,
            detail="Could not extract text from one or both documents.",
        )

    logger.info(
        "analyze.request",
        role=role,
        experience_level=experience_level,
        use_demo=used_demo,
    )

    service = AnalysisService(orchestrator)
    try:
        result = await service.run_analysis(
            resume_text=resume_text,
            jd_text=jd_text,
            role=role,
            experience_level=experience_level,
            used_demo=used_demo,
            db=db,
        )
    except Exception as e:
        logger.error("analyze.error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")

    return JSONResponse(content=result.model_dump())


@router.get("/export/{session_id}")
async def export_report(
    session_id: str,
    format: str = "json",
    db: AsyncSession = Depends(get_db),
    orchestrator: AgentOrchestrator = Depends(get_orchestrator),
):
    from app.services.export_service import ExportService
    service = AnalysisService(orchestrator)
    analysis = await service.fetch_from_db(session_id, db)
    if not analysis:
        raise HTTPException(status_code=404, detail="Session not found")

    exporter = ExportService()
    if format == "txt":
        from fastapi.responses import PlainTextResponse
        return PlainTextResponse(
            content=exporter.to_txt(analysis),
            media_type="text/plain",
            headers={"Content-Disposition": f'attachment; filename="PathForge_{session_id[:8]}.txt"'},
        )
    elif format == "html":
        from fastapi.responses import HTMLResponse
        return HTMLResponse(
            content=exporter.to_html(analysis),
            headers={"Content-Disposition": f'attachment; filename="PathForge_{session_id[:8]}.html"'},
        )
    else:
        return JSONResponse(
            content=analysis.model_dump(),
            headers={"Content-Disposition": f'attachment; filename="PathForge_{session_id[:8]}.json"'},
        )

@router.post("/analyze/stream")
async def analyze_stream(
    role: Annotated[str, Form()],
    experience_level: Annotated[str, Form()],
    use_demo: Annotated[bool, Form()] = False,
    resume_file: Optional[UploadFile] = File(None),
    jd_file: Optional[UploadFile] = File(None),
    orchestrator: AgentOrchestrator = Depends(get_orchestrator)
):
    async def event_generator():
        steps = [
            ("BERT NER",          "Extracting skills from resume…"),
            ("Groq LLM",          "Running LLM skill extraction…"),
            ("Gap Computation",   "Computing cosine similarity…"),
            ("O*NET Grounding",   "Verifying against O*NET…"),
        ]

        for i, (name, msg) in enumerate(steps):
            yield f"data: {json.dumps({'step': i+1,'name':name,'message':msg})}\n\n"
            await asyncio.sleep(0.1)

        # Now call the real orchestrator
        result = await orchestrator.run_analysis(
            role=role,
            experience_level=experience_level,
            resume_file=resume_file,
            jd_file=jd_file,
            use_demo=use_demo
        )

        yield f"data: {json.dumps({'done': True, 'result': result.model_dump()})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
