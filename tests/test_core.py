from tempfile import SpooledTemporaryFile
from unittest.mock import Mock
from uuid import uuid4

import pytest
from fastapi import HTTPException, UploadFile

from app.core.config import Settings
from app.schemas.analysis import AnalysisDraft, AnalysisResponse, RequirementEvidence, RequirementStatus
from app.services.analysis_service import (
    AnalysisService,
    GeminiAnalysisError,
    GeminiFailureCategory,
    _classify_gemini_error,
)
from app.services.chunking_service import chunk_text
from app.services.pdf_service import extract_pdf_text


def settings(**overrides) -> Settings:
    values = {
        "gemini_api_key": "test-key",
        "database_url": "postgresql+psycopg://postgres:postgres@localhost:5433/cv_match",
        "_env_file": None,
    }
    values.update(overrides)
    return Settings(**values)


def upload_file(filename: str, content: bytes) -> UploadFile:
    stream = SpooledTemporaryFile()
    stream.write(content)
    stream.seek(0)
    return UploadFile(filename=filename, file=stream)


def test_settings_validate_and_keep_model_values():
    config = settings(gemini_model="gemini-3.7-flash", gemini_embedding_model="gemini-embedding-001")
    assert config.gemini_model == "gemini-3.7-flash"
    assert config.gemini_embedding_model == "gemini-embedding-001"
    assert config.embedding_dimensions == 3072


def test_settings_reject_invalid_overlap():
    with pytest.raises(ValueError, match="CHUNK_OVERLAP"):
        settings(chunk_size=200, chunk_overlap=200)


def test_chunking_respects_overlap_and_size():
    text = " ".join(f"w{i}" for i in range(100))
    chunks = chunk_text(text, chunk_size=50, chunk_overlap=10)
    assert len(chunks) > 1
    assert all(len(chunk) <= 50 for chunk in chunks)
    assert chunks[0][-10:] == chunks[1][:10]


def test_chunking_empty_text():
    assert chunk_text(" \n ", 100, 10) == []


def test_pdf_upload_rejects_non_pdf():
    upload = upload_file("resume.pdf", b"not a pdf")
    with pytest.raises(HTTPException) as err:
        extract_pdf_text(upload)
    assert err.value.status_code == 415


def test_pdf_upload_rejects_wrong_extension():
    upload = upload_file("resume.txt", b"%PDF-valid")
    with pytest.raises(HTTPException) as err:
        extract_pdf_text(upload)
    assert err.value.status_code == 415


def test_pdf_upload_extracts_text(monkeypatch):
    from app.services import pdf_service

    page = Mock()
    page.extract_text.return_value = "Experiência profissional com Python."
    monkeypatch.setattr(pdf_service, "PdfReader", lambda _stream: Mock(pages=[page]))
    upload = upload_file("resume.PDF", b"%PDF-valid")
    assert extract_pdf_text(upload) == "Experiência profissional com Python."


def test_analysis_response_validation_and_explicit_score():
    response = AnalysisResponse(
        compatibility_score=75,
        summary="Boa aderência.",
        strengths=["Python"],
        missing_requirements=["Kubernetes"],
        requirements_analysis=[
            RequirementEvidence(requirement="Python", status=RequirementStatus.FOUND, evidence="Desenvolveu APIs em Python."),
            RequirementEvidence(requirement="Kubernetes", status=RequirementStatus.PARTIAL, evidence="Menciona containers Docker."),
        ],
        recommendations=[],
        score_method="Teste",
    )
    assert AnalysisService._score([r.model_dump() for r in response.requirements_analysis]) == 75


def test_matched_requirement_requires_evidence():
    with pytest.raises(ValueError, match="evidência"):
        RequirementEvidence(requirement="Python", status="found", evidence="")


def test_gemini_generation_error_is_logged_without_secrets(monkeypatch, caplog):
    from app.services import analysis_service as module

    service = AnalysisService.__new__(AnalysisService)
    service.settings = settings()
    service.client = Mock()
    service.embeddings = Mock()
    service.embeddings.embed_texts.return_value = [[0.1] * 3072]
    monkeypatch.setattr(module, "retrieve_chunks", lambda *_args: [Mock(id=1, content="private resume sentence")])
    service.client.models.generate_content.side_effect = RuntimeError(
        "API key=test-key; job=Uma vaga de engenharia backend com Python e PostgreSQL; cv=private resume sentence"
    )
    db = Mock()
    db.get.return_value = object()

    with pytest.raises(GeminiAnalysisError) as err:
        service.analyze(db, uuid4(), "Uma vaga de engenharia backend com Python e PostgreSQL")
    assert err.value.category == GeminiFailureCategory.UNKNOWN
    assert "RuntimeError" in caplog.text
    assert "Traceback:" in caplog.text
    assert "test-key" not in caplog.text
    assert "Uma vaga de engenharia backend" not in caplog.text
    assert "private resume sentence" not in caplog.text


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        (401, GeminiFailureCategory.AUTHENTICATION),
        (403, GeminiFailureCategory.AUTHENTICATION),
        (404, GeminiFailureCategory.MODEL_NOT_FOUND),
        (400, GeminiFailureCategory.INVALID_REQUEST),
        (422, GeminiFailureCategory.INVALID_REQUEST),
        (429, GeminiFailureCategory.RATE_LIMITED),
        (503, GeminiFailureCategory.UNAVAILABLE),
    ],
)
def test_gemini_http_errors_are_classified(code, expected):
    from google.genai.errors import APIError

    exc = APIError(code, {"error": {"message": "provider error"}})
    assert _classify_gemini_error(exc) == expected


def test_google_genai_accepts_pydantic_response_schema():
    from google.genai import types

    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=AnalysisDraft,
        temperature=0.1,
    )
    assert config.response_schema is AnalysisDraft


def test_upload_persists_chunks_and_embeddings(monkeypatch):
    from app.api.routes import cvs

    monkeypatch.setattr(cvs, "extract_pdf_text", lambda _file: "Python backend developer")

    class FakeEmbeddings:
        def __init__(self, _settings):
            pass

        def embed_texts(self, chunks, _task):
            return [[0.25] * 3072 for _ in chunks]

    monkeypatch.setattr(cvs, "GeminiEmbeddingService", FakeEmbeddings)
    db = Mock()
    db.commit.return_value = None
    result = cvs.upload_cv(
        upload_file("resume.pdf", b"%PDF-valid"),
        db=db,
        settings=settings(chunk_size=100, chunk_overlap=10),
    )
    stored = db.add.call_args.args[0]
    assert result.chunks_created == 1
    assert stored.chunks[0].content == "Python backend developer"
    assert len(stored.chunks[0].embedding) == 3072
    db.commit.assert_called_once()


def test_retrieval_uses_postgresql_cosine_distance():
    from app.services.retrieval_service import retrieve_chunks
    from sqlalchemy.dialects import postgresql

    db = Mock()
    db.scalars.return_value.all.return_value = []
    retrieve_chunks(db, uuid4(), [0.1] * 3072, 5)
    statement = db.scalars.call_args.args[0]
    sql = str(statement.compile(dialect=postgresql.dialect()))
    assert "<=>" in sql
    assert "LIMIT" in sql
