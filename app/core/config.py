from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8", extra="ignore")

    gemini_api_key: SecretStr
    gemini_model: str = "gemini-3.7-flash"
    gemini_embedding_model: str = "gemini-embedding-001"
    database_url: str
    top_k: int = Field(default=5, ge=1, le=100)
    chunk_size: int = Field(default=1000, ge=100)
    chunk_overlap: int = Field(default=200, ge=0)
    embedding_dimensions: int = Field(default=3072, ge=128, le=3072)

    @model_validator(mode="after")
    def validate_chunking(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP precisa ser menor que CHUNK_SIZE")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
