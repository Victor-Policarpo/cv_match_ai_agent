import json
import logging
import re
import traceback
import uuid
from enum import StrEnum

from google import genai
from google.genai import types
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.models import Resume
from app.schemas.analysis import AnalysisDraft, AnalysisResponse, RequirementStatus
from app.services.embedding_service import GeminiEmbeddingService
from app.services.retrieval_service import retrieve_chunks

logger = logging.getLogger(__name__)


class GeminiFailureCategory(StrEnum):
    AUTHENTICATION = "authentication"
    MODEL_NOT_FOUND = "model_not_found"
    INVALID_REQUEST = "invalid_request"
    RATE_LIMITED = "rate_limited"
    UNAVAILABLE = "unavailable"
    TIMEOUT = "timeout"
    INVALID_RESPONSE = "invalid_response"
    UNKNOWN = "unknown"


_FAILURE_DETAILS = {
    GeminiFailureCategory.AUTHENTICATION: (502, "Falha de autenticação ou permissão no Gemini"),
    GeminiFailureCategory.MODEL_NOT_FOUND: (502, "O modelo configurado não foi encontrado ou não está disponível"),
    GeminiFailureCategory.INVALID_REQUEST: (422, "O Gemini rejeitou os parâmetros da solicitação"),
    GeminiFailureCategory.RATE_LIMITED: (429, "Limite de requisições/cota do Gemini atingido"),
    GeminiFailureCategory.UNAVAILABLE: (503, "O serviço Gemini está indisponível"),
    GeminiFailureCategory.TIMEOUT: (504, "O Gemini excedeu o tempo limite"),
    GeminiFailureCategory.INVALID_RESPONSE: (502, "O Gemini retornou uma resposta estruturada inválida"),
    GeminiFailureCategory.UNKNOWN: (502, "Falha na geração da análise pelo Gemini"),
}


class ResumeNotFoundError(Exception):
    pass


class NoResumeEvidenceError(Exception):
    pass


class GeminiAnalysisError(Exception):
    def __init__(self, category: GeminiFailureCategory):
        self.category = category
        self.http_status, self.public_detail = _FAILURE_DETAILS[category]
        super().__init__(self.public_detail)


def _classify_gemini_error(exc: Exception) -> GeminiFailureCategory:
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    class_name = type(exc).__name__.lower()
    message = str(getattr(exc, "message", "") or exc).lower()
    if code in (401, 403) or "authenticationerror" in class_name:
        return GeminiFailureCategory.AUTHENTICATION
    if code == 404 or "notfounderror" in class_name or re.search(
        r"model.{0,50}(not found|does not exist|not available|not supported)", message
    ):
        return GeminiFailureCategory.MODEL_NOT_FOUND
    if code == 429 or "ratelimiterror" in class_name:
        return GeminiFailureCategory.RATE_LIMITED
    if code in (400, 422) or "badrequesterror" in class_name:
        return GeminiFailureCategory.INVALID_REQUEST
    if code is not None and code >= 500:
        return GeminiFailureCategory.UNAVAILABLE
    if "timeout" in class_name or "timed out" in message:
        return GeminiFailureCategory.TIMEOUT
    if isinstance(exc, (ValueError, TypeError)):
        return GeminiFailureCategory.INVALID_REQUEST
    if "validationerror" in class_name or isinstance(exc, json.JSONDecodeError):
        return GeminiFailureCategory.INVALID_RESPONSE
    if any(term in class_name for term in ("connectionerror", "connecterror", "networkerror")):
        return GeminiFailureCategory.UNAVAILABLE
    return GeminiFailureCategory.UNKNOWN


def _redact_log_text(value: str, settings: Settings, sensitive_texts: list[str]) -> str:
    for secret in [settings.gemini_api_key.get_secret_value(), *sensitive_texts]:
        if secret and len(secret) >= 4:
            value = value.replace(secret, "[REDACTED]")
    value = re.sub(r"AIza[0-9A-Za-z_-]{20,}", "[REDACTED_API_KEY]", value)
    value = re.sub(r"(?i)(authorization\s*[:=]\s*bearer\s+)\S+", r"\1[REDACTED]", value)
    value = re.sub(r"(?i)(api[_ -]?key\s*[:=]\s*)[^\s,;\"']+", r"\1[REDACTED]", value)
    value = re.sub(r"(?i)(postgres(?:ql)?(?:\+\w+)?://)[^\s@]+@", r"\1[REDACTED]@", value)
    return value


def _log_gemini_failure(
    exc: Exception,
    category: GeminiFailureCategory,
    settings: Settings,
    sensitive_texts: list[str],
) -> None:
    formatted_traceback = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    safe_message = _redact_log_text(str(exc), settings, sensitive_texts)
    safe_traceback = _redact_log_text(formatted_traceback, settings, sensitive_texts)
    logger.error(
        "Gemini request failed | category=%s | exception_type=%s | message=%s\nTraceback:\n%s",
        category.value,
        type(exc).__name__,
        safe_message,
        safe_traceback,
    )


class AnalysisService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = genai.Client(api_key=settings.gemini_api_key.get_secret_value())
        self.embeddings = GeminiEmbeddingService(settings)

    def analyze(self, db: Session, resume_id: uuid.UUID, job_description: str) -> AnalysisResponse:
        if db.get(Resume, resume_id) is None:
            raise ResumeNotFoundError
        try:
            query_vector = self.embeddings.embed_texts([job_description], "RETRIEVAL_QUERY")[0]
        except Exception as exc:
            raise GeminiAnalysisError("Falha ao gerar embedding da vaga") from exc
        chunks = retrieve_chunks(db, resume_id, query_vector, self.settings.top_k)
        if not chunks:
            raise NoResumeEvidenceError
        evidence = [{"chunk_id": item.id, "text": item.content} for item in chunks]
        prompt = f"""Analise a compatibilidade da vaga com o currículo usando APENAS as evidências recuperadas.
Não invente competências. Use status found somente com evidência explícita, partial para evidência incompleta,
not_found quando não houver evidência. Ausência no texto não prova ausência da habilidade.
strengths e requirements_analysis devem citar somente fatos evidenciados. recommendations devem ser construtivas.
O campo compatibility_score será calculado pelo servidor e não deve ser incluído.

DESCRIÇÃO DA VAGA:
{job_description}

EVIDÊNCIAS DO CURRÍCULO (JSON):
{json.dumps(evidence, ensure_ascii=False)}"""
        sensitive_texts = [job_description, *(chunk.content for chunk in chunks)]
        try:
            response = self.client.models.generate_content(
                model=self.settings.gemini_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=AnalysisDraft,
                    temperature=0.1,
                ),
            )
        except Exception as exc:
            category = _classify_gemini_error(exc)
            _log_gemini_failure(exc, category, self.settings, sensitive_texts)
            raise GeminiAnalysisError(category) from exc

        try:
            draft = AnalysisDraft.model_validate(json.loads(response.text or ""))
            payload = draft.model_dump()
            payload["strengths"] = [
                f"{item.requirement}: {item.evidence}"
                for item in draft.requirements_analysis
                if item.status in (RequirementStatus.FOUND, RequirementStatus.PARTIAL)
            ]
            payload["missing_requirements"] = [
                item.requirement
                for item in draft.requirements_analysis
                if item.status == RequirementStatus.NOT_FOUND
            ]
            payload["compatibility_score"] = self._score(payload["requirements_analysis"])
            payload["score_method"] = "Média por requisito: found=1, partial=0.5, not_found=0; arredondada para 0–100. Requisitos recebem peso igual."
            return AnalysisResponse.model_validate(payload)
        except Exception as exc:
            _log_gemini_failure(
                exc,
                GeminiFailureCategory.INVALID_RESPONSE,
                self.settings,
                sensitive_texts,
            )
            raise GeminiAnalysisError(GeminiFailureCategory.INVALID_RESPONSE) from exc

    @staticmethod
    def _score(requirements: list[dict]) -> int:
        if not requirements:
            return 0
        weights = {RequirementStatus.FOUND: 1.0, RequirementStatus.PARTIAL: 0.5, RequirementStatus.NOT_FOUND: 0.0}
        return round(100 * sum(weights[RequirementStatus(item["status"])] for item in requirements) / len(requirements))
