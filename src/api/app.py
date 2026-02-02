import logging

from fastapi import FastAPI

from src.api.lifespan import lifespan
from src.api.routers import health_router, ingestion_router, generation_router, retrieval_router


logger = logging.getLogger(__name__)

def create_app() -> FastAPI:
    """
    Фабрика FastAPI-приложения.

    Инициализирует приложение, подключает роутеры
    и настраивает lifespan.
    """
    logger.info("Инициализация FastAPI приложения")

    try:
        app = FastAPI(
            lifespan=lifespan,
            title="RAG система",
        )

        app.include_router(health_router)
        app.include_router(ingestion_router)
        app.include_router(generation_router)
        app.include_router(retrieval_router)

        logger.info("FastAPI приложение успешно создано")
        return app

    except Exception:
        logger.exception("Ошибка при создании FastAPI приложения")
        raise

