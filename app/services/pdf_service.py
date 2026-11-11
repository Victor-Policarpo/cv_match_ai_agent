from io import BytesIO

from fastapi import HTTPException, UploadFile
from pypdf import PdfReader


def extract_pdf_text(file: UploadFile) -> str:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=415, detail="Envie um arquivo PDF")
    content = file.file.read()
    if not content.startswith(b"%PDF-"):
        raise HTTPException(status_code=415, detail="O conteúdo enviado não é um PDF válido")
    try:
        reader = PdfReader(BytesIO(content))
        text = "\n".join(page.extract_text() or "" for page in reader.pages).strip()
    except Exception as exc:
        raise HTTPException(status_code=422, detail="Não foi possível extrair texto do PDF") from exc
    if not text:
        raise HTTPException(status_code=422, detail="O PDF não contém texto extraível")
    return text
