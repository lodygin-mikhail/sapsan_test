from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings

BASE_DIR = Path(__file__).parents[2]


class Settings(BaseSettings):
    REDIS_URL: str
    BASE_LLM_URL: str
    API_KEY: str
    LLM_MODEL: str
    EMBEDDING_MODEL: str
    RERANKER_MODEL: str
    QDRANT_URL: str
    COLLECTION_NAME: str

    class Config:
        env_file = BASE_DIR / ".env"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()