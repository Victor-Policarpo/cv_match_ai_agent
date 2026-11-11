from google import genai
from google.genai import types

from app.core.config import Settings


class GeminiEmbeddingService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = genai.Client(api_key=settings.gemini_api_key.get_secret_value())

    def embed_texts(self, texts: list[str], task_type: str) -> list[list[float]]:
        if not texts:
            return []
        response = self.client.models.embed_content(
            model=self.settings.gemini_embedding_model,
            contents=texts,
            config=types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=self.settings.embedding_dimensions,
            ),
        )
        vectors = [item.values for item in (response.embeddings or [])]
        if len(vectors) != len(texts) or any(vector is None for vector in vectors):
            raise RuntimeError("Gemini retornou uma quantidade inválida de embeddings")
        return [list(vector) for vector in vectors]
