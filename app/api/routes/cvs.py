import uuid

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.database.connection import get_db
from app.models import Resume, ResumeChunk
from app.schemas.resume import ResumeUploadResponse
from app.services.chunking_service import chunk_text
from app.services.embedding_service import GeminiEmbeddingService
from app.services.pdf_service import extract_pdf_text

router = APIRouter(prefix="/api/v1/cvs", tags=["CVs"])


@router.post("/upload", response_model=ResumeUploadResponse, status_code=201)
def upload_cv(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> ResumeUploadResponse:
    text = extract_pdf_text(file)
    chunks = chunk_text(text, settings.chunk_size, settings.chunk_overlap)
    try:
        vectors = GeminiEmbeddingService(settings).embed_texts(chunks, "RETRIEVAL_DOCUMENT")
    except Exception as exc:
        raise HTTPException(status_code=502, detail="Falha ao gerar embeddings para o currículo") from exc
    resume = Resume(id=uuid.uuid4(), filename=file.filename or "resume.pdf")
    resume.chunks = [ResumeChunk(chunk_index=i, content=chunk, embedding=vector) for i, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True))]
    try:
        db.add(resume)
        db.commit()
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=503, detail="Não foi possível salvar o currículo no banco") from exc
    return ResumeUploadResponse(resume_id=str(resume.id), filename=resume.filename, chunks_created=len(chunks))
