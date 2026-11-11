from fastapi import FastAPI

from app.api.routes.analysis import router as analysis_router
from app.api.routes.cvs import router as cvs_router

app = FastAPI(title="CV Match Agent", version="0.1.0", description="RAG para análise de compatibilidade entre currículo e vaga")
app.include_router(cvs_router)
app.include_router(analysis_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
