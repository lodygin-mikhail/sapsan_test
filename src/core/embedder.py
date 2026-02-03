import logging

from langchain_openai import OpenAIEmbeddings

from src.core.config import settings


logger = logging.getLogger(__name__)

class Embedder:
    """Обёртка над OpenAI Embeddings."""

    def __init__(self):
        logger.info("Инициализация Embedder")
        self.client = OpenAIEmbeddings(
            base_url=settings.BASE_LLM_URL,
            model=settings.EMBEDDING_MODEL,
            api_key=settings.API_KEY,
        )

    async def aembed_query(self, query: str):

        return await self.client.aembed_query(query)

    async def aembed_documents(self, documents: list[str]):

        return await self.client.aembed_documents(documents)
