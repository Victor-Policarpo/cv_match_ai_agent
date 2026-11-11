from pydantic import BaseModel


class ResumeUploadResponse(BaseModel):
    resume_id: str
    filename: str
    chunks_created: int
