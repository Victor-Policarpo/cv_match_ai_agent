"""Create resume and vector chunk tables."""
from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

from app.core.config import get_settings

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    dimensions = get_settings().embedding_dimensions
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "resumes",
        sa.Column("id", sa.Uuid(as_uuid=True), primary_key=True),
        sa.Column("filename", sa.String(length=255), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_table(
        "resume_chunks",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("resume_id", sa.Uuid(as_uuid=True), sa.ForeignKey("resumes.id", ondelete="CASCADE"), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(dimensions), nullable=False),
    )
    op.create_index("ix_resume_chunks_resume_id", "resume_chunks", ["resume_id"])


def downgrade() -> None:
    op.drop_index("ix_resume_chunks_resume_id", table_name="resume_chunks")
    op.drop_table("resume_chunks")
    op.drop_table("resumes")
