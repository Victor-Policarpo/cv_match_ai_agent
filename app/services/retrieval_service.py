import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import ResumeChunk


def retrieve_chunks(db: Session, resume_id: uuid.UUID, query_embedding: list[float], top_k: int) -> list[ResumeChunk]:
    distance = ResumeChunk.embedding.cosine_distance(query_embedding)
    statement = (
        select(ResumeChunk)
        .where(ResumeChunk.resume_id == resume_id)
        .order_by(distance)
        .limit(top_k)
    )
    return list(db.scalars(statement).all())
