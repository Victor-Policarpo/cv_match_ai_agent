import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.database.connection import get_db
from app.schemas.analysis import AnalysisRequest, AnalysisResponse
from app.services.analysis_service import (
    AnalysisService,
    GeminiAnalysisError,
    NoResumeEvidenceError,
    ResumeNotFoundError,
)

router = APIRouter(prefix="/api/v1", tags=["Analysis"])


@router.post("/analysis", response_model=AnalysisResponse)
def analyze_cv(
    request: AnalysisRequest,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> AnalysisResponse:
    try:
        resume_id = uuid.UUID(request.resume_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="resume_id inválido") from exc
    try:
        return AnalysisService(settings).analyze(db, resume_id, request.job_description)
    except ResumeNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Currículo não encontrado") from exc
    except NoResumeEvidenceError as exc:
        raise HTTPException(status_code=422, detail="Currículo sem trechos para recuperação") from exc
    except GeminiAnalysisError as exc:
        raise HTTPException(
            status_code=exc.http_status,
            detail={"category": exc.category.value, "message": exc.public_detail},
        ) from exc
