import logging

from langchain_openai import ChatOpenAI

from src.core.config import settings

logger = logging.getLogger(__name__)


def get_llm(model=settings.LLM_MODEL):
    """
    Фабрика LLM клиента.
    """
    logger.info("Создание LLM клиента: %s", model)
    return ChatOpenAI(
        base_url=settings.BASE_LLM_URL,
        model=model,
        api_key=settings.API_KEY,
    )
